"""Admission for the native contiguous minibatch schedule (no GPU claim)."""
import ast
from configparser import ConfigParser
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def helpers():
    tree = ast.parse((ROOT / "integrations/puffer_coworld_frozen_transfer.py").read_text())
    names = {"install_minibatch_rotation", "validate_minibatch_rotation"}
    module = ast.Module(body=[x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names], type_ignores=[])
    scope = {"Path": Path, "ConfigParser": ConfigParser}
    exec(compile(module, "trainer", "exec"), scope)
    return scope


def settings(replay=0.5, minibatch=8192, graphs=-1):
    config = ConfigParser()
    config.read_dict({"base": {"cudagraphs": graphs}, "vec": {"total_agents": 4096},
                      "train": {"horizon": 128, "minibatch_size": minibatch, "replay_ratio": replay}})
    return config


def test_schedule_native_compilation(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    target = source / "pufferl.cu"
    target.write_text("        int dest_off = (mb * mb_segs) % n_rows;")
    helpers()["install_minibatch_rotation"](tmp_path)
    fragment = target.read_text()
    cpp = '''#include <cstdio>
#include <cassert>
#include <cstdint>
#include <algorithm>
struct Hyper { bool cudagraphs=false; } h, *hypers=&h;
struct Puff { int epoch=0; } p, *pufferl=&p;
int main() {
  int Nmb=64, mb_segs=64, n_rows=4096;
  for (float replay : {0.1f, 0.5f, 1.0f, 2.0f}) {
    int total_minibatches=replay*4096*128/8192;
    int visits[64]={};
    for(p.epoch=0;p.epoch<64;p.epoch++) {
      for(int mb=0;mb<total_minibatches;mb++) {
''' + fragment + '''
        assert(dest_off>=0 && dest_off+Nmb<=n_rows && dest_off%Nmb==0);
        visits[dest_off/Nmb]++;
      }
    }
    for(int i=0;i<64;i++) assert(visits[i]==total_minibatches);
  }
}'''
    (tmp_path / "schedule.cpp").write_text(cpp)
    subprocess.run(["c++", "-std=c++17", str(tmp_path / "schedule.cpp"), "-o", str(tmp_path / "schedule")], check=True)
    subprocess.run([str(tmp_path / "schedule")], check=True)
    with pytest.raises(ValueError, match="anchor"):
        helpers()["install_minibatch_rotation"](tmp_path)


@pytest.mark.parametrize("replay", [.1, .5, 1, 2])
def test_valid_settings(replay):
    helpers()["validate_minibatch_rotation"](settings(replay))


@pytest.mark.parametrize("config", [settings(graphs=0), settings(graphs=1), settings(minibatch=8193),
                                      settings(minibatch=128*63), settings(replay=0.001)])
def test_invalid_settings(config):
    with pytest.raises(ValueError):
        helpers()["validate_minibatch_rotation"](config)
