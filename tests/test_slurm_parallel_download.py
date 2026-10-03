"""Real local HTTPS/curl proof of bounded concurrent image transfer lifetime."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from integrations.slurm_s3_job import download_parts


@unittest.skipUnless(shutil.which('curl') and shutil.which('openssl'), 'requires curl and openssl')
class ParallelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        cert = self.root / 'cert.pem'
        key = self.root / 'key.pem'
        config = self.root / 'openssl.cnf'
        config.write_text('[req]\ndistinguished_name=dn\nx509_extensions=ext\nprompt=no\n'
                          '[dn]\nCN=localhost\n[ext]\nsubjectAltName=IP:127.0.0.1\n')
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
                        '-config', str(config), '-keyout', str(key), '-out', str(cert)],
                       check=True, capture_output=True, timeout=15)
        owner = self
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.delay = 0.01
        self.block = b'qualified-input' * 1024

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                with owner.lock:
                    owner.active += 1
                    owner.peak = max(owner.peak, owner.active)
                try:
                    if self.path == '/fail':
                        self.send_error(404)
                        return
                    self.send_response(200)
                    self.send_header('Content-Length', str(len(owner.block) * 40))
                    self.end_headers()
                    for _ in range(40):
                        self.wfile.write(owner.block)
                        self.wfile.flush()
                        time.sleep(owner.delay)
                except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                    pass
                finally:
                    with owner.lock:
                        owner.active -= 1

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
        self.server.socket = context.wrap_socket(self.server.socket, server_side=True)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.env = dict(os.environ, CURL_CA_BUNDLE=str(cert))
        self.base = f'https://127.0.0.1:{self.server.server_port}'
        self.parts = [(self.base + f'/part{i}', self.root / f'part{i}') for i in range(4)]

    def test_real_parallel_bytes_and_bounded_concurrency(self):
        with patch.dict(os.environ, self.env):
            download_parts(self.parts)
        expected = hashlib.sha256(self.block * 40).hexdigest()
        for _, path in self.parts:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
        self.assertGreaterEqual(self.peak, 2)
        self.assertLessEqual(self.peak, 4)

    def test_failed_transfer_rejects_batch_without_leaking_url(self):
        self.parts[1] = (self.base + '/fail', self.parts[1][1])
        with patch.dict(os.environ, self.env), self.assertRaises(RuntimeError) as caught:
            download_parts(self.parts)
        self.assertNotIn(self.base, str(caught.exception))
        sizes = {p.name: p.stat().st_size for _, p in self.parts if p.exists()}
        time.sleep(0.15)
        self.assertEqual(sizes, {p.name: p.stat().st_size for _, p in self.parts if p.exists()})

    def test_signal_stops_all_writers_before_return(self):
        self.delay = 0.05
        script = self.root / 'interrupt.py'
        script.write_text('''import json,signal,sys
from pathlib import Path
from integrations.slurm_s3_job import download_parts,JobSignal
def stop(signum,frame):raise JobSignal(signum)
signal.signal(signal.SIGUSR1,stop)
try:download_parts([(u,Path(p)) for u,p in json.loads(sys.argv[1])])
except JobSignal:print('DOWNLOAD_STOPPED',flush=True)
''')
        env = dict(self.env, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        process = subprocess.Popen([sys.executable, str(script), json.dumps([(u, str(p)) for u, p in self.parts])],
                                   env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 10
            while not self.active:
                if process.poll() is not None or time.monotonic() > deadline:
                    self.fail('Real curl transfer did not start')
                time.sleep(0.01)
            process.send_signal(signal.SIGUSR1)
            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertIn(b'DOWNLOAD_STOPPED', stdout)
            sizes = {p.name: p.stat().st_size for _, p in self.parts if p.exists()}
            time.sleep(0.3)
            self.assertEqual(sizes, {p.name: p.stat().st_size for _, p in self.parts if p.exists()})
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
