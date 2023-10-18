from pathlib import Path

import pytest

import psst


@pytest.fixture
def yaml_config() -> str:
    return """---
run:
  num_epochs: 3
  num_samples_train: 512000
  num_samples_test: 219500

generator:
  parameter: Bg
  batch_size: 64
  phi_range:
    min: 1e-2
    max: 1e3
    num: 224
    log_scale: True
  nw_range:
    min: 1e-2
    max: 1e3
    num: 224
    log_scale: True
  visc_range:
    min: 1e-2
    max: 1e3
    log_scale: True
  bg_range:
    min: 1e-2
    max: 1e3
  bth_range:
    min: 1e-2
    max: 1e3
  pe_range:
    min: 1e-2
    max: 1e3

adam:
  lr: 1e-3
  betas: [0.7, 0.9]
  eps: 1e-9
  weight_decay: 0.0
...
"""


@pytest.fixture
def json_config() -> str:
    return """{
    "run": {
        "num_epochs": 3,
        "num_samples_train": 512000,
        "num_samples_test": 219500
    },
    "generator": {
        "parameter": "Bg",
        "batch_size": 64,
        "phi_range": {
            "min": 0.01,
            "max": 1000.0,
            "num": 224,
            "log_scale": true
        },
        "nw_range": {
            "min": 0.01,
            "max": 1000.0,
            "num": 224,
            "log_scale": true
        },
        "visc_range": {
            "min": 0.01,
            "max": 1000.0,
            "log_scale": true
        },
        "bg_range": {
            "min": 0.01,
            "max": 1000.0
        },
        "bth_range": {
            "min": 0.01,
            "max": 1000.0
        },
        "pe_range": {
            "min": 0.01,
            "max": 1000.0
        }
    },
    "adam": {
        "lr": 0.001,
        "betas": [
            0.7,
            0.9
        ],
        "eps": 1e-09,
        "weight_decay": 0.0
    }
}
"""


def test_yaml_config(tmp_path: Path, yaml_config):
    filepath = tmp_path / "test_config.yaml"
    filepath.write_text(yaml_config)

    config = psst.loadConfig(filepath)

    assert config.run_config.num_epochs == 3
    assert config.run_config.num_samples_train == 512000
    assert config.run_config.num_samples_test == 219500
    assert config.run_config.checkpoint_filename == "chk.pt"
    assert config.run_config.checkpoint_frequency == 0

    assert config.adam_config.lr == 1e-3
    assert len(config.adam_config.betas) == 2
    assert config.adam_config.betas[0] == 0.7
    assert config.adam_config.betas[1] == 0.9
    assert config.adam_config.eps == 1e-9
    assert config.adam_config.weight_decay == 0

    gen = config.generator_config
    assert gen.parameter == "Bg"
    assert gen.batch_size == 64

    ranges = {
        gen.phi_range,
        gen.nw_range,
        gen.visc_range,
        gen.bg_range,
        gen.bth_range,
        gen.pe_range,
    }
    assert all(r.min == 0.01 for r in ranges)
    assert all(r.max == 1000 for r in ranges)

    ranges1 = {gen.phi_range, gen.nw_range, gen.visc_range}
    assert all(r.log_scale for r in ranges1)
    assert all(not r.log_scale for r in (ranges - ranges1))

    ranges1 = {gen.phi_range, gen.nw_range}
    assert all(r.num == 224 for r in ranges1)
    assert all(r.num == 0 for r in (ranges - ranges1))


def test_json_config(tmp_path: Path, json_config):
    filepath = tmp_path / "test_config.yaml"
    filepath.write_text(json_config)

    config = psst.loadConfig(filepath)

    assert config.run_config.num_epochs == 3
    assert config.run_config.num_samples_train == 512000
    assert config.run_config.num_samples_test == 219500
    assert config.run_config.checkpoint_filename == "chk.pt"
    assert config.run_config.checkpoint_frequency == 0

    assert config.adam_config.lr == 1e-3
    assert len(config.adam_config.betas) == 2
    assert config.adam_config.betas[0] == 0.7
    assert config.adam_config.betas[1] == 0.9
    assert config.adam_config.eps == 1e-9
    assert config.adam_config.weight_decay == 0

    gen = config.generator_config
    assert gen.parameter == "Bg"
    assert gen.batch_size == 64

    ranges = {
        gen.phi_range,
        gen.nw_range,
        gen.visc_range,
        gen.bg_range,
        gen.bth_range,
        gen.pe_range,
    }
    assert all(r.min == 0.01 for r in ranges)
    assert all(r.max == 1000 for r in ranges)

    ranges1 = {gen.phi_range, gen.nw_range, gen.visc_range}
    assert all(r.log_scale for r in ranges1)
    assert all(not r.log_scale for r in (ranges - ranges1))

    ranges1 = {gen.phi_range, gen.nw_range}
    assert all(r.num == 224 for r in ranges1)
    assert all(r.num == 0 for r in (ranges - ranges1))
