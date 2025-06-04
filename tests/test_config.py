"""
Tests for configuration module.
"""

import os

import pytest

from nirs_tomato.config import (
    DataConfig,
    ExperimentConfig,
    FeatureSelectionConfig,
    FeatureSelectionMethod,
    ModelConfig,
    ModelType,
    TransformType,
)


def test_experiment_config_creation():
    """Test creation of experiment configuration."""
    config = ExperimentConfig(
        name="test_experiment",
        data=DataConfig(data_path="data/test.csv", target_column="Brix"),
    )
    assert config.name == "test_experiment"
    assert config.data.data_path == "data/test.csv"
    assert config.data.target_column == "Brix"


def test_experiment_config_from_yaml(tmp_path):
    """Test loading configuration from YAML file."""
    # Create temporary YAML file
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        f.write(
            """
name: yaml_test
data:
  data_path: data/yaml_test.csv
  target_column: Brix
  transform: snv
model:
  model_type: pls
  pls_n_components: 5
        """
        )

    # Load configuration
    config = ExperimentConfig.from_yaml(str(config_path))

    # Check values
    assert config.name == "yaml_test"
    assert config.data.data_path == "data/yaml_test.csv"
    assert config.data.transform == TransformType.SNV
    assert config.model.model_type == ModelType.PLS
    assert config.model.pls_n_components == 5


def test_experiment_config_to_yaml(tmp_path):
    """Test saving configuration to YAML file."""
    # Create configuration
    config = ExperimentConfig(
        name="save_test",
        data=DataConfig(
            data_path="data/save_test.csv",
            target_column="Brix",
            transform=TransformType.MSC,
        ),
        model=ModelConfig(model_type=ModelType.RF, rf_n_estimators=50),
    )

    # Save to YAML
    config_path = tmp_path / "saved_config.yaml"
    config.to_yaml(str(config_path))

    # Check the file exists
    assert os.path.exists(config_path)

    # Read the raw content to verify serialization
    with open(config_path, "r") as f:
        content = f.read()
        # Check that the file contains the enums (serialized format might vary)
        assert "TransformType.MSC" in content
        assert "ModelType.RF" in content
        assert "rf_n_estimators: 50" in content


def test_field_validation():
    """Test field validations in configs."""
    # Test SavGolConfig window_length validation
    from nirs_tomato.config import SavGolConfig

    # Valid window length (odd number)
    valid_config = SavGolConfig(window_length=11)
    assert valid_config.window_length == 11

    # Invalid window length (even number)
    with pytest.raises(ValueError):
        SavGolConfig(window_length=10)

    # Test ModelConfig test_size validation
    with pytest.raises(ValueError):
        ModelConfig(test_size=0.05)  # Too small

    with pytest.raises(ValueError):
        ModelConfig(test_size=0.6)  # Too large


def test_experiment_name_generation():
    """Test experiment name generation based on config."""
    config = ExperimentConfig(
        name="test",
        data=DataConfig(
            data_path="test.csv",
            target_column="Brix",
            transform=TransformType.SNV,
            savgol={
                "enabled": True,
                "window_length": 15,
                "polyorder": 2,
                "deriv": 0,
            },
        ),
        feature_selection=FeatureSelectionConfig(
            method=FeatureSelectionMethod.VIP, n_features=20
        ),
        model=ModelConfig(model_type=ModelType.PLS, tune_hyperparams=True),
    )

    generated_name = config.get_experiment_name()
    assert "pls" in generated_name
    assert "snv" in generated_name
    assert "sg15" in generated_name
    assert "vip20" in generated_name
    assert "tuned" in generated_name


def test_file_not_found_error():
    """Test file not found error when loading config."""
    with pytest.raises(FileNotFoundError):
        ExperimentConfig.from_yaml("non_existent_config.yaml")
