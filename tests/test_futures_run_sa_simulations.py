import unittest
import pickle
from unittest.mock import patch
from uq_physicell.SA_script import run_sa_simulations

# Define the mock function at the module level
def mock_run_replicate_serializable(ini_path, struc_name, sampleID, replicateID, ParametersXML, ParametersRules, qois_dic=None, drop_columns=[], custom_summary_function=None):
    return sampleID, replicateID, pickle.dumps({"out1": 1.0, "out2": 2.0})

class MockPhysiCellModel:
    """A serializable mock object for PhysiCell_Model."""
    def __init__(self, ini_filePath, strucName):
        self.numReplicates = 2
        self.XML_parameters_variable = {'param_xml1': 'param1'}
        self.parameters_rules_variable = {'param_rule1': 'param2'}
        self.output_folder = '/tmp/test_output'

    def info(self):
        return "Mocked PhysiCell Model Info"

class TestRunSASimulationsFutures(unittest.TestCase):
    @patch('uq_physicell.SA_script.PhysiCell_Model', new=MockPhysiCellModel)
    @patch('uq_physicell.SA_script.check_existing_sa')
    @patch('uq_physicell.SA_script.create_db_structure')
    @patch('uq_physicell.SA_script.insert_metadata')
    @patch('uq_physicell.SA_script.insert_inputs')
    @patch('uq_physicell.SA_script.insert_output')
    @patch('uq_physicell.SA_script.run_replicate_serializable', new=mock_run_replicate_serializable)
    def test_run_sa_simulations_futures(self, mock_insert_inputs, mock_insert_metadata, mock_insert_output,
                                        mock_create_db_structure, mock_check_existing_sa):
        # Mock check_existing_sa
        mock_check_existing_sa.return_value = (False, [], [], [])

        # Input parameters
        ini_filePath = "test.ini"
        strucName = "test_structure"
        SA_type = "test_type"
        SA_method = "test_method"
        SA_sampler = "test_sampler"
        param_names = ["param1", "param2"]
        ref_values = [1.0, 1.0]
        bounds = [(0.0, 2.0), (0.5, 1.5)]
        perturbations = [1, 0.5]
        dic_samples = {0: {"param1": 1.0, "param2": 2.0},
                       1: {"param1": 1.25, "param2": 0.75},
                       2: {"param1": 1.75, "param2": 1.25}}
        qois_dic = {}
        db_file = "/tmp/test_db.db"
        use_futures = True
        num_workers = 2

        # Run the function
        run_sa_simulations(ini_filePath, strucName, SA_type, SA_method, SA_sampler, param_names, ref_values, bounds,
                           perturbations, dic_samples, qois_dic, db_file, use_mpi=False, use_futures=use_futures, num_workers=num_workers)

        # Assertions
        print("Asserting database structure creation...")
        mock_create_db_structure.assert_called_once_with(db_file)
        print("Asserting metadata insertion...")
        mock_insert_metadata.assert_called_once()
        print("Asserting output insertion...")
        mock_insert_output.assert_called()

if __name__ == "__main__":
    unittest.main()