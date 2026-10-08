import inspect
import json
from pathlib import Path
import pytest
from integrations.policy_execution import execute

@pytest.mark.parametrize('steps,seconds,has_config',[(2621440,480,True),(2097152,481,True),(2097152,480,False)])
def test_diagnostic_exception_cannot_admit_unbounded_training(tmp_path,steps,seconds,has_config):
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(total_timesteps=steps,overrides={'vec.total_agents':4096,'train.horizon':128})))
    with pytest.raises(ValueError):
        execute('unused',[],source=tmp_path,output=tmp_path/'output',sampler={},name='train',
                seconds=seconds,training_config=config if has_config else None,diagnostic_profile=True)
    assert not (tmp_path/'output').exists()

def test_production_execution_keeps_strict_default():
    assert inspect.signature(execute).parameters['diagnostic_profile'].default is False
