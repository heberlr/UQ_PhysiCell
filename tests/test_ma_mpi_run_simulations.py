from mpi4py import MPI
import unittest
import pickle
from unittest.mock import MagicMock, patch
from uq_physicell.model_analysis.ma_context import run_simulations, ModelAnalysisContext


# Create a serializable mock model class
class SerializableMockModel:
    def __init__(self, ini_filePath=None, strucName=None):
        self.numReplicates = 2
        self.XML_parameters_variable = {'param_xml1': 'param1'}
        self.parameters_rules_variable = {'param_rule1': 'param2'}
        self.output_folder = '/tmp/test_output'
    
    def info(self):
        return "Mocked PhysiCell Model Info"


# Create a serializable mock function
def mock_run_replicate_func(ini_path, struc_name, sampleID, replicateID, ParametersXML, ParametersRules, qois_dic=None, drop_columns=[], custom_summary_function=None):
    return sampleID, replicateID, pickle.dumps({"out1": 1.0, "out2": 2.0})


def mock_run_replicate_serializable_func(ini_path, struc_name, sampleID, replicateID, ParametersXML, ParametersRules, qois_dic=None, drop_columns=[], custom_summary_function=None, return_binary_output=False):
    return sampleID, replicateID, pickle.dumps({"out1": 1.0, "out2": 2.0})


class TestRunSASimulationsMPI(unittest.TestCase):
    @patch('uq_physicell.model_analysis.ma_context.PhysiCell_Model', new=SerializableMockModel)
    @patch('uq_physicell.model_analysis.ma_context.check_simulations_db')
    @patch('uq_physicell.model_analysis.ma_context.create_structure')
    @patch('uq_physicell.model_analysis.ma_context.insert_metadata')
    @patch('uq_physicell.model_analysis.ma_context.insert_param_space')
    @patch('uq_physicell.model_analysis.ma_context.insert_qois')
    @patch('uq_physicell.model_analysis.ma_context.insert_samples')
    @patch('uq_physicell.model_analysis.ma_context.insert_output')
    @patch('uq_physicell.model_analysis.ma_context.run_replicate', new=mock_run_replicate_func)
    @patch('uq_physicell.model_analysis.ma_context.run_replicate_serializable', new=mock_run_replicate_serializable_func)
    def test_run_sa_simulations_mpi(self, mock_insert_output, mock_insert_samples,
                                    mock_insert_qoi, mock_insert_param_space, mock_insert_metadata, 
                                    mock_create_structure, mock_check_simulations_db):
        # Mock MPI environment
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        print(f"Rank {rank}: Starting test on {size} MPI processes")

        # Mock check_simulations_db
        mock_check_simulations_db.return_value = (False, [], [], [])

        # Input parameters
        model_config = {"ini_path": "test.ini", "struc_name": "test_structure"}
        sampler = "LHS"
        params_dict = {
            "names": ["param1", "param2"],
            "samples": {0: {"param1": 1.0, "param2": 2.0},
                       1: {"param1": 1.25, "param2": 0.75},
                       2: {"param1": 1.75, "param2": 1.25}}
        }
        qois_dic = None
        db_file = "test_db.db"
        
        # Create mock context
        mock_context = ModelAnalysisContext(db_file, model_config, sampler, params_dict, qois_dic)
        mock_context.dic_samples = params_dict["samples"]
        mock_context.cancelled = lambda: False

        # Run the function
        print(f"Rank {rank}: Calling run_simulations")
        run_simulations(mock_context)
        print(f"Rank {rank}: Finished run_simulations")

        # Assertions
        if rank == 0:
            # Master rank: Check setup tasks
            print("Rank 0: Asserting database structure creation...")
            mock_create_structure.assert_called_once_with(db_file)
            print("Rank 0: Asserting metadata insertion...")
            mock_insert_metadata.assert_called_once()
            print("Rank 0: Asserting parameter space insertion...")
            mock_insert_param_space.assert_called_once()
            print("Rank 0: Asserting insert qoi...")
            mock_insert_qoi.assert_called_once()
            print("Rank 0: Asserting insert samples...")
            mock_insert_samples.assert_called()
            print("Rank 0: Asserting insert outputs...")
            # Since mocks should succeed, insert_output should be called
            self.assertTrue(mock_insert_output.called, "insert_output should have been called after successful simulations")
            print("Rank 0: insert_output was called successfully")

        print(f"Rank {rank}: Test completed successfully.")

if __name__ == "__main__":
    unittest.main()
    # how to call: mpirun -n 4 python tests/test_mpi_run_sa_simulations.py
    # or mpiexec -n 4 python tests/test_mpi_run_sa_simulations.py