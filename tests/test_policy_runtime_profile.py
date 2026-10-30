import json
from types import SimpleNamespace

from integrations import policy_runtime_profile as profile


def test_gpu_sample_preserves_existing_csv_and_adds_hardware(tmp_path, monkeypatch):
    values = ['2026/10/07 00:00:00.000','GPU-123','68515 MiB','42 %','NVIDIA H100 80GB HBM3',
              '580.95.05','P0','1800 MHz','2619 MHz','350 W','700 W','53','0x0000000000000000']
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=', '.join(values)+'\n')
    monkeypatch.setattr(profile.subprocess,'run',run)
    profile.telemetry(tmp_path,'GPU-123')
    assert (tmp_path/'gpu.csv').read_text() == ', '.join(values[:4])+'\n'
    report = json.loads((tmp_path/'hardware.jsonl').read_text())
    assert report['gpu']['clocks.current.sm'] == '1800 MHz'
    assert report['gpu']['uuid'] == 'GPU-123'
    assert report['cpu']['effective_cpus'] > 0
    assert len(calls) == 1


def test_cpu_quota_caps_affinity(monkeypatch):
    from pathlib import Path
    exists, read = Path.exists, Path.read_text
    files = {'/sys/fs/cgroup/cpu.max':'250000 100000', '/sys/fs/cgroup/cpu.stat':'nr_throttled 42',
             '/sys/fs/cgroup/cpuset.cpus.effective':'0-7'}
    monkeypatch.setattr(profile.os,'sched_getaffinity',lambda pid:set(range(8)),raising=False)
    monkeypatch.setattr(Path,'exists',lambda p:str(p) in files or exists(p))
    monkeypatch.setattr(Path,'read_text',lambda p,*a,**kw:files[str(p)] if str(p) in files else read(p,*a,**kw))
    result = profile.cpu_snapshot()
    assert result['effective_cpus'] == 2.5
    assert result['cgroup']['cpu.stat'] == 'nr_throttled 42'
