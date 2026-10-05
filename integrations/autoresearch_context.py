"""Deterministic context transport readable by the provider's rootless builder."""

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


def write_tar(root, stream):
    root = Path(root).resolve()
    entries = [root, *sorted(root.rglob('*'))]
    if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in entries):
        raise ValueError('Build context must contain only regular files and directories')
    with tarfile.open(fileobj=stream, mode='w|', format=tarfile.GNU_FORMAT) as archive:
        for path in entries:
            name = '.' if path == root else './' + str(path.relative_to(root))
            member = archive.gettarinfo(str(path), name)
            member.uid = member.gid = member.mtime = 0
            member.uname = member.gname = ''
            # Rootless BuildKit must read files created under a private umask.
            member.mode = 0o755 if path.is_dir() or member.mode & 0o111 else 0o644
            if path.is_file():
                with path.open('rb') as source:
                    archive.addfile(member, source)
            else:
                archive.addfile(member)


def pack_context(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError('Write the archive outside its input tree')
    with output.open('xb') as destination:
        compressor = subprocess.Popen(['zstd', '-q', '-3', '-c'], stdin=subprocess.PIPE,
                                      stdout=destination)
        try:
            write_tar(root, compressor.stdin)
        finally:
            compressor.stdin.close()
            code = compressor.wait()
        if code:
            raise RuntimeError('Context compression failed; partial archive retained')
    with output.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    return dict(path=str(output), size_bytes=output.stat().st_size, sha256=digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(pack_context(args.root, args.output)))


if __name__ == '__main__':
    main()
