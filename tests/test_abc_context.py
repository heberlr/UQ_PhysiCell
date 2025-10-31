"""
Test script for ABC CalibrationContext to verify basic functionality.
This tests the import and initialization without requiring a full PhysiCell setup.
"""

import sys
import os
import logging
import tempfile
import numpy as np

# Add the path to import uq_physicell
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that the new CalibrationContext can be imported."""
    try:
        from uq_physicell.abc import CalibrationContext, run_abc_calibration
        print("✅ Successfully imported CalibrationContext and run_abc_calibration")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_initialization():
    """Test CalibrationContext initialization with minimal configuration."""
    try:
        from uq_physicell.abc import CalibrationContext
        
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
            'config_file': 'dummy_config.xml',
            'model_name': 'test_model',
            'numReplicates': 2
        }
        
        qoi_functions = {
            'QoI1': 'lambda df: df["QoI1"].values',
            'QoI2': 'lambda df: df["QoI2"].values'
        }
        
        distance_functions = {
            'QoI1': {'function': 'euclidean', 'weight': 1.0},
            'QoI2': {'function': 'euclidean', 'weight': 1.0}
        }
        
        search_space = {
            'param1': {'type': 'real', 'lower_bound': 0.1, 'upper_bound': 1.0},
            'param2': {'type': 'real', 'lower_bound': 0.5, 'upper_bound': 2.0}
        }
        
        abc_options = {
            'max_populations': 5,
            'max_simulations': 50,
            'sampler': 'multicore',
            'num_workers': 2,
            'mode': 'local'
        }
        
        # Create temporary database file
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
            search_space=search_space,
            abc_options=abc_options,
            logger=logger
        )
        
        print("✅ CalibrationContext initialized successfully")
        print(f"   Database: {calib_context.db_path}")
        print(f"   QoIs: {list(calib_context.qoi_functions.keys())}")
        print(f"   Parameters: {list(calib_context.search_space.keys())}")
        print(f"   Sampler: {calib_context.sampler_type}")
        print(f"   Workers: {calib_context.num_workers}")
        
        # Cleanup
        try:
            os.unlink(db_path)
        except:
            pass
            
        return True
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_validation():
    """Test that configuration validation works correctly."""
    try:
        from uq_physicell.abc import CalibrationContext
        
        # Test with missing required keys
        invalid_model_config = {
            'numReplicates': 2
            # Missing 'config_file' and 'model_name'
        }
        
        try:
            CalibrationContext(
                db_path="dummy.db",
                obsData={'dummy': [1, 2, 3]},
                obsData_columns={'dummy': 'dummy'},
                model_config=invalid_model_config,  # Invalid config
                qoi_functions={'dummy': 'lambda x: x'},
                distance_functions={'dummy': {'function': 'euclidean', 'weight': 1.0}},
                search_space={'param1': {'type': 'real', 'lower_bound': 0, 'upper_bound': 1}},
                abc_options={'max_populations': 5}
            )
            print("❌ Validation should have failed but didn't")
            return False
        except ValueError as e:
            print(f"✅ Validation correctly caught error: {e}")
            return True
            
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False

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