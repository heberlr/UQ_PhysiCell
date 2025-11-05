"""
Test script for ABC CalibrationContext to verify basic functionality.
This tests the import and initialization without requiring a full PhysiCell setup.
"""

import sys
import os
import logging
import tempfile
import numpy as np
import pytest

# Add the path to import uq_physicell
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that the new CalibrationContext can be imported."""
    from uq_physicell.abc import CalibrationContext
    from pyabc import Distribution, RV
    # Basic sanity assertions: imports resolved
    assert CalibrationContext is not None
    assert Distribution is not None
    assert RV is not None

def test_initialization():
    """Test CalibrationContext initialization with minimal configuration."""
    from uq_physicell.abc import CalibrationContext
    from pyabc import Distribution, RV

    # Setup minimal configuration
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Synthetic observed data
    obs_data = {
        'QoI1': np.array([1.0, 1.1, 1.2, 1.3, 1.4]),
        'QoI2': np.array([0.2, 0.21, 0.22, 0.23, 0.24])
    }

    obs_data_columns = {
        'QoI1': 'QoI1_data',
        'QoI2': 'QoI2_data'
    }

    model_config = {
        'ini_path': 'ini_file_path.ini',
        'struc_name': 'model_struc_name',
    }

    qoi_functions = {
        'QoI1': 'lambda df: df["QoI1"].values',
        'QoI2': 'lambda df: df["QoI2"].values'
    }

    distance_functions = {
        'QoI1': {'function': 'euclidean', 'weight': 1.0},
        'QoI2': {'function': 'euclidean', 'weight': 1.0}
    }

    lb1 = 0.0; ub1 = 5.0
    lb2 = 0.0; ub2 = 10.0; loc2 = 5.0; scale2 = 2.0 # lb and ub are bounds, loc is mean, scale is stddev
    prior = Distribution(
        param1 = RV('uniform', lb1, ub1-lb1),
        param2 = RV('truncnorm', lb2, ub2, loc2, scale2)
    )

    abc_options = {
        'max_populations': 5,
        'max_simulations': 50,
        'sampler': 'multicore',
        'num_workers': 2,
        'mode': 'local'
    }

    # Create temporary database file
    db_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name

        # Initialize CalibrationContext
        calib_context = CalibrationContext(
            db_path=db_path,
            obsData=obs_data,
            obsData_columns=obs_data_columns,
            model_config=model_config,
            qoi_functions=qoi_functions,
            distance_functions=distance_functions,
            prior=prior,
            abc_options=abc_options,
            logger=logger
        )

        # Sanity-check a few attributes
        assert hasattr(calib_context, 'db_path')
        assert list(calib_context.qoi_functions.keys()) == ['QoI1', 'QoI2']
        # prior is a pyabc Distribution — ensure parameter names property exists if available
        if hasattr(calib_context.prior, 'get_parameter_names'):
            assert callable(calib_context.prior.get_parameter_names)
        assert hasattr(calib_context, 'sampler_type')
        assert hasattr(calib_context, 'num_workers')

    finally:
        # Cleanup
        if db_path and os.path.exists(db_path):
            try:
                os.unlink(db_path)
            except Exception:
                pass

def test_configuration_validation():
    """Test that configuration validation works correctly."""
    from uq_physicell.abc import CalibrationContext
    from pyabc import Distribution, RV

    # Test with missing required keys
    invalid_model_config = {
        'numReplicates': 2
        # Missing 'config_file' and 'model_name'
    }

    with pytest.raises(ValueError):
        CalibrationContext(
            db_path="dummy.db",
            obsData={'dummy': [1, 2, 3]},
            obsData_columns={'dummy': 'dummy'},
            model_config=invalid_model_config,  # Invalid config
            qoi_functions={'dummy': 'lambda x: x'},
            distance_functions={'dummy': {'function': 'euclidean', 'weight': 1.0}},
            prior= Distribution(param1 = RV('uniform', 0, 1.0)),
            abc_options={'max_populations': 5}
        )

def main():
    """Run all tests."""
    print("🧪 Testing ABC CalibrationContext")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Initialization Test", test_initialization),
        ("Validation Test", test_configuration_validation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        result = test_func()
        results.append(result)
        print(f"{'✅ PASSED' if result else '❌ FAILED'}: {test_name}")
    
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! CalibrationContext is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)