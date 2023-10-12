from pathlib import Path
import pytest

from ..temp_st.core.configuration import *

config_file_contents = """---
train_size: 51200
test_size: 21952
epochs: 300
checkpoint_file: "bg_train.pt"
continuing: False

generator:
  parameter: Bg
  batch_size: 64
  phi_range: 
    min: 0.00003
    max: 0.02
    num: 224
    geom: True
  nw_range: 
    min: 100
    max: 100000
    num: 224
    geom: True
  eta_sp_dist: [1, 1000000]
  bg_dist: [0.36, 1.55]
  bth_dist: [0.22, 0.82]
  pe_dist: [3.2, 13.5]

adam:
  learning_rate: 0.001
  betas: [0.7, 0.9]
  epsilon: 0.000000001
  weight_decay: 0.0

...
"""

class testConfig:
    def __init__(self, tmp_path: Path):
        filepath = tmp_path / ""
        self.config = getConfig(filepath)
        self.generator_config = self.config.generator_config
        self.adam_config = self.config.adam_config
    
    def test_getConfig():
        pass
