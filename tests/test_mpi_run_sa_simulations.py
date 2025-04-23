from mpi4py import MPI
import unittest
import pickle
from unittest.mock import MagicMock, patch
from uq_physicell.SA_script import run_sa_simulations


class TestRunSASimulationsMPI(unittest.TestCase):
    @patch('uq_physicell.SA_script.PhysiCell_Model')
    @patch('uq_physicell.SA_script.check_existing_sa')
    @patch('uq_physicell.SA_script.create_db_structure')
    @patch('uq_physicell.SA_script.insert_metadata')
    @patch('uq_physicell.SA_script.insert_inputs')
    @patch('uq_physicell.SA_script.insert_output')
    @patch('uq_physicell.SA_script.run_replicate')
    def test_run_sa_simulations_mpi(self, mock_run_replicate, mock_insert_inputs, mock_insert_metadata, mock_insert_outputs,
                                    mock_create_db_structure, mock_check_existing_sa, mock_PhysiCell_Model):
        # Mock MPI environment
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        print(f"Rank {rank}: Starting test on {size} MPI processes")

        # Mock PhysiCell_Model
        mock_model_instance = MagicMock()
        mock_model_instance.numReplicates = 2
        mock_model_instance.XML_parameters_variable = {'param_xml1': 'param1'}
        mock_model_instance.parameters_rules_variable = {'param_rule1': 'param2'}
        mock_model_instance.output_folder = '/tmp/test_output'
        mock_PhysiCell_Model.return_value = mock_model_instance

        # Mock check_existing_sa
        mock_check_existing_sa.return_value = (False, [], [], [])

        # Mock run_replicate
        mock_run_replicate.return_value = (0, 1, pickle.dumps({"out1": 1.0, "out2": 2.0}))

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
        db_file = "test_db.db"
        use_mpi = True

        # Run the function
        print(f"Rank {rank}: Calling run_sa_simulations")
        run_sa_simulations(ini_filePath, strucName, SA_type, SA_method, SA_sampler, param_names, ref_values, bounds,
                           perturbations, dic_samples, qois_dic, db_file, use_mpi=use_mpi, use_futures=False)
        print(f"Rank {rank}: Finished run_sa_simulations")

        # Assertions
        if rank == 0:
            # Master rank: Check setup tasks
            print("Rank 0: Asserting database structure creation...")
            mock_create_db_structure.assert_called_once_with(db_file)
            print("Rank 0: Asserting metadata insertion...")
            mock_insert_metadata.assert_called_once()
            print("Rank 0: Asserting input parameters insertion...")
            mock_insert_outputs.assert_called_once()
            print("Rank 0: Asserting insert outputs...")

        else:
            # Worker ranks: Check simulation execution
            print(f"Rank {rank}: Asserting run_replicate calls...")
            mock_run_replicate.assert_called()

        print(f"Rank {rank}: Test completed successfully.")

if __name__ == "__main__":
    unittest.main()
    # how to call: mpirun -n 4 python tests/test_mpi_run_sa_simulations.py
    # or mpiexec -n 4 python tests/test_mpi_run_sa_simulations.py