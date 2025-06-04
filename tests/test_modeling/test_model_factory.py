"""
Tests for model factory module.
"""

import lightgbm as lgb
import pytest
import xgboost as xgb
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from nirs_tomato.modeling.model_factory import create_model


def test_create_pls_model():
    """Test creation of PLS regression model."""
    model = create_model(model_type="pls", n_components=5)

    assert model is not None
    assert isinstance(model, PLSRegression)
    assert model.n_components == 5


def test_create_svr_model():
    """Test creation of SVR model."""
    model = create_model(model_type="svr", kernel="rbf", C=10.0)

    assert model is not None
    assert isinstance(model, SVR)
    assert model.kernel == "rbf"
    assert model.C == 10.0


def test_create_rf_model():
    """Test creation of Random Forest model."""
    model = create_model(model_type="rf", n_estimators=50, max_depth=10)

    assert model is not None
    assert isinstance(model, RandomForestRegressor)
    assert model.n_estimators == 50
    assert model.max_depth == 10


def test_create_xgb_model():
    """Test creation of XGBoost model."""
    model = create_model(model_type="xgb", n_estimators=50, max_depth=5)

    assert model is not None
    assert isinstance(model, xgb.XGBRegressor)
    assert model.n_estimators == 50
    assert model.max_depth == 5


def test_create_lgbm_model():
    """Test creation of LightGBM model."""
    model = create_model(model_type="lgbm", n_estimators=50, num_leaves=31)

    assert model is not None
    assert isinstance(model, lgb.LGBMRegressor)
    assert model.n_estimators == 50
    assert model.num_leaves == 31


def test_invalid_model_type():
    """Test error handling for invalid model type."""
    with pytest.raises(ValueError):
        create_model(model_type="invalid_model_type")


def test_default_parameters():
    """Test default parameters are applied correctly."""
    # Create model without specifying parameters
    model = create_model(model_type="rf")

    assert model is not None
    assert isinstance(model, RandomForestRegressor)
    assert model.n_estimators == 100  # Default value
