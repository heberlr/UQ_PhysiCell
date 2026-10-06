"""Unit tests for uq_physicell.utils.pc_studio.

Builds small synthetic pcdl-4-style TimeStep objects (only the ``data`` dict the
adapter reads), so no PhysiCell output folder is required.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

import xml.etree.ElementTree as ET

import pickle
import sqlite3

from uq_physicell.database.ma_db import create_structure, insert_metadata, insert_output, insert_param_space, insert_samples
from uq_physicell.pc_model import _sha256_file
from uq_physicell.utils.pc_studio import (StudioFrameLoader, check_database, config_status, database_summary,
                                         find_reference_config, model_config_info, timestep_to_studio_data,
                                         ini_fixed_parameters, output_differences, verify_config,
                                         xml_differences)


# ─── helpers ────────────────────────────────────────────────────────────────

def _fake_timestep(time=0.0, xmlfile='output00000000.xml', substrates=('oxygen', 'drug'), microenv=True):
    """2x2x1 mesh, 2 cell types, pcdl column naming."""
    substrates = list(substrates) if microenv else []
    x, y, z = np.array([-10.0, 10.0]), np.array([-10.0, 10.0]), np.array([0.0])
    grid = np.array(np.meshgrid(x, y, z, indexing='xy'))
    coords = np.array([[-10.0, 10.0, -10.0, 10.0], [-10.0, -10.0, 10.0, 10.0], [0.0, 0.0, 0.0, 0.0]])

    df_cell = pd.DataFrame({
        'position_x': [1.0, 2.0, 3.0],
        'previous_velocity_000': [0.1, 0.2, 0.3],
        'cell_type': ['tumor', 'immune', 'tumor'],
        'cycle_model': ['flow_cytometry_separated_cycle_model'] * 2 + ['apoptosis_death_model'],
        'current_phase': ['G0G1_phase', 'S_phase', 'apoptotic'],
        'dead': [False, False, True],
        'tumor_attack_rates': [0.0, 0.5, 0.0],
        'immune_attack_rates': [0.0, 0.0, 0.0],
        # pcdl-derived columns that Studio does not have
        'voxel_i': [0, 1, 0], 'mesh_center_m': [-10.0, 10.0, -10.0], 'time': [time] * 3,
        'xmlfile': [xmlfile] * 3, 'position_vectorlength': [1.0, 2.0, 3.0],
    }, index=pd.Index([0, 1, 5], name='ID'))
    if microenv:
        for i, sub in enumerate(substrates):
            df_cell[f'{sub}_secretion_rates'] = [float(i)] * 3
            df_cell[sub] = [9.0] * 3  # concentration at the cell's voxel (pcdl-derived)
            df_cell[f'{sub}_decay_rate'] = [0.1] * 3
    else:
        df_cell['secretion_rates_0'] = [0.0] * 3
        df_cell['secretion_rates_1'] = [1.0] * 3

    df_conc = pd.DataFrame({
        'voxel_i': [0, 1, 0, 1], 'voxel_j': [0, 0, 1, 1], 'voxel_k': [0, 0, 0, 0],
        **{sub: np.arange(4, dtype=float) + 10 * i for i, sub in enumerate(substrates)},
    })
    df_sub = pd.DataFrame({'decay_rate': [0.1] * len(substrates), 'diffusion_coefficient': [1000.0] * len(substrates)},
                          index=pd.Index(substrates, name='substrate'))
    data = {
        'metadata': {'multicellds_version': 'MultiCellDS_2', 'pcdl_version': 'pcdl_4.1.7',
                     'physicell_version': 'PhysiCell_1.14.2', 'created': 'now', 'current_time': time,
                     'time_units': 'min', 'current_runtime': 1.0, 'runtime_units': 'sec', 'spatial_unit': 'micron',
                     'ds_unit': {'position_x': 'micron', 'previous_velocity': 'micron/min', 'oxygen': 'mmHg'}},
        'mesh': {'mnp_grid': grid, 'mnp_axis': [x, y, z], 'mnp_range': [], 'ijk_range': [], 'ijk_axis': [],
                 'xyz_range': [(-20.0, 20.0), (-20.0, 20.0), (-10.0, 10.0)], 'mnp_coordinate': coords, 'volume': 8000.0, 'mnp_spacing': [20.0, 20.0, 20.0]},
        'substrate': {'ds_substrate': {str(i): s for i, s in enumerate(substrates)}, 'ls_substarte': substrates,
                      'df_substarte': df_sub, 'df_conc': df_conc},
        'cell': {'ds_celltype': {'0': 'tumor', '1': 'immune'}, 'ls_celltype': ['tumor', 'immune'],
                 'df_cell': df_cell, 'ls_cellattr': [], 'dei_graph': {}},
    }
    return SimpleNamespace(data=data, get_cell_df=lambda: df_cell)


# ─── timestep_to_studio_data ────────────────────────────────────────────────

class TestTimestepToStudioData:
    def test_top_level_layout(self):
        data = timestep_to_studio_data(_fake_timestep(), microenv=True)
        assert set(data) == {'metadata', 'mesh', 'continuum_variables', 'discrete_cells'}
        assert data['metadata']['spatial_units'] == 'micron'
        assert 'pcdl_version' not in data['metadata']
        np.testing.assert_array_equal(data['mesh']['volumes'], np.full(4, 8000.0))

    def test_no_continuum_key_when_microenv_not_requested(self):
        assert 'continuum_variables' not in timestep_to_studio_data(_fake_timestep(), microenv=False)

    def test_cell_columns_renamed_and_derived_dropped(self):
        cells = timestep_to_studio_data(_fake_timestep(), microenv=False)['discrete_cells']['data']
        assert list(cells)[0] == 'ID'
        np.testing.assert_array_equal(cells['ID'], [0.0, 1.0, 5.0])
        assert 'previous_velocity_0' in cells and 'previous_velocity_000' not in cells
        # per-substrate / per-cell-type columns follow Studio's index labels
        assert {'secretion_rates_0', 'secretion_rates_1', 'attack_rates_0', 'attack_rates_1'} <= set(cells)
        np.testing.assert_array_equal(cells['attack_rates_0'], [0.0, 0.5, 0.0])
        for derived in ('voxel_i', 'mesh_center_m', 'time', 'xmlfile', 'position_vectorlength',
                        'oxygen', 'oxygen_decay_rate', 'oxygen_secretion_rates'):
            assert derived not in cells

    def test_single_substrate_uses_plain_label(self):
        cells = timestep_to_studio_data(_fake_timestep(substrates=('oxygen',)))['discrete_cells']['data']
        assert 'secretion_rates' in cells and 'secretion_rates_0' not in cells

    def test_unindexed_substrate_columns_without_microenv(self):
        cells = timestep_to_studio_data(_fake_timestep(microenv=False), microenv=False)['discrete_cells']['data']
        np.testing.assert_array_equal(cells['secretion_rates_1'], [1.0, 1.0, 1.0])

    def test_categorical_columns_back_to_codes(self):
        cells = timestep_to_studio_data(_fake_timestep(), microenv=False)['discrete_cells']['data']
        np.testing.assert_array_equal(cells['cell_type'], [0.0, 1.0, 0.0])
        np.testing.assert_array_equal(cells['cycle_model'], [6.0, 6.0, 100.0])
        np.testing.assert_array_equal(cells['current_phase'], [4.0, 10.0, 100.0])
        np.testing.assert_array_equal(cells['dead'], [0.0, 0.0, 1.0])
        assert all(arr.dtype == np.float64 for arr in cells.values())

    def test_units(self):
        units = timestep_to_studio_data(_fake_timestep(), microenv=False)['discrete_cells']['units']
        assert units['ID'] == 'none'
        assert units['position_x'] == 'micron'
        assert units['previous_velocity_0'] == 'micron/min'  # falls back to the vector's base name

    def test_substrate_grid_indexed_j_i_k(self):
        cont = timestep_to_studio_data(_fake_timestep(), microenv=True)['continuum_variables']
        assert set(cont) == {'oxygen', 'drug'}
        field = cont['drug']['data']
        assert field.shape == (2, 2, 1)
        # df_conc rows: (i, j) = (0,0)->10, (1,0)->11, (0,1)->12, (1,1)->13
        np.testing.assert_array_equal(field[:, :, 0], [[10.0, 11.0], [12.0, 13.0]])
        assert cont['oxygen']['units'] == 'mmHg'
        assert cont['oxygen']['decay_rate']['value'] == 0.1
        assert cont['oxygen']['diffusion_coefficient']['value'] == 1000.0

    def test_microenv_requested_but_not_stored_warns(self):
        with pytest.warns(UserWarning, match='no microenvironment data'):
            data = timestep_to_studio_data(_fake_timestep(microenv=False), microenv=True)
        assert data['continuum_variables'] == {}

    def test_graph_placeholders(self):
        graph = timestep_to_studio_data(_fake_timestep(), microenv=False, graph=True)['discrete_cells']['graph']
        assert graph == {'neighbor_cells': {}, 'attached_cells': {}}

    def test_rejects_non_pcdl4_object(self):
        with pytest.raises(TypeError, match='pcdl 4 TimeStep'):
            timestep_to_studio_data(SimpleNamespace(data={'discrete_cells': {}}))


# ─── StudioFrameLoader ──────────────────────────────────────────────────────

class TestStudioFrameLoader:
    def _loader(self, n=3):
        return StudioFrameLoader([_fake_timestep(time=60.0 * i, xmlfile=f'output{i:08d}.xml') for i in range(n)])

    def test_list_frames(self):
        assert self._loader().list_frames() == ['output00000000.xml', 'output00000001.xml', 'output00000002.xml']

    def test_call_with_studio_signature(self):
        loader = self._loader()
        data = loader('output00000002.xml', '/some/output/dir', microenv=False, graph=False)
        assert data['metadata']['current_time'] == 120.0

    def test_frame_index_aliases(self):
        loader = self._loader()
        assert loader.frame_index('/abs/path/output00000001.xml') == 1
        assert loader.frame_index('initial.xml') == 0
        assert loader.frame_index('final.xml') == 2
        with pytest.raises(KeyError):
            loader.frame_index('output00000099.xml')

    def test_frame_name_falls_back_to_position_without_cells(self):
        ts = _fake_timestep()
        ts.data['cell']['df_cell'] = ts.data['cell']['df_cell'].iloc[0:0]
        assert StudioFrameLoader([ts]).list_frames() == ['output00000000.xml']

    def test_rejects_empty_list(self):
        with pytest.raises(ValueError):
            StudioFrameLoader([])

    def test_from_database(self):
        df = pd.DataFrame({'SampleID': [3], 'ReplicateID': [0], 'Data': [[_fake_timestep(), _fake_timestep(time=60.0)]]})
        with patch('uq_physicell.database.ma_db.load_output', return_value=df) as mock_load:
            loader = StudioFrameLoader.from_database('ma.db', sample_id=3, replicate_id=0)
        mock_load.assert_called_once_with('ma.db', sample_ids=[3], replicate_ids=[0])
        assert len(loader) == 2

    def test_from_database_rejects_qoi_output(self):
        df = pd.DataFrame({'SampleID': [3], 'ReplicateID': [0], 'Data': [pd.DataFrame({'q': [1.0]})]})
        with patch('uq_physicell.database.ma_db.load_output', return_value=df):
            with pytest.raises(ValueError, match='does not store raw MCDS output'):
                StudioFrameLoader.from_database('ma.db', sample_id=3, replicate_id=0)

    def test_from_database_missing_row(self):
        with patch('uq_physicell.database.ma_db.load_output', return_value=pd.DataFrame()):
            with pytest.raises(ValueError, match='No output'):
                StudioFrameLoader.from_database('ma.db', sample_id=3, replicate_id=0)


# ─── index folder / database helpers ────────────────────────────────────────

_CONFIG_XML = """<PhysiCell_settings><microenvironment_setup>
  <variable name="oxygen" units="mmHg" ID="0"><physical_parameter_set>
    <diffusion_coefficient units="micron^2/min">100000.0</diffusion_coefficient>
    <decay_rate units="1/min">0.1</decay_rate></physical_parameter_set></variable>
</microenvironment_setup></PhysiCell_settings>"""


class TestWriteIndexFolder:
    def _write(self, tmp_path, **kw):
        loader = StudioFrameLoader([_fake_timestep(time=60.0 * i, xmlfile=f'output{i:08d}.xml', **kw) for i in range(2)])
        config = tmp_path / 'settings.xml'
        config.write_text(_CONFIG_XML)
        out = tmp_path / 'view'
        loader.write_index_folder(str(out), config_file=str(config))
        return out

    def test_files_written(self, tmp_path):
        out = self._write(tmp_path)
        assert sorted(p.name for p in out.iterdir()) == [
            'PhysiCell_settings.xml', 'final.xml', 'initial.xml', 'output00000000.xml', 'output00000001.xml']

    def test_fields_studio_parses(self, tmp_path):
        root = ET.parse(self._write(tmp_path) / 'output00000001.xml').getroot()
        assert float(root.find('.//current_time').text) == 60.0
        assert root.find('.//microenvironment//domain//mesh//x_coordinates').text.split() == ['-10', '10']
        assert [v.get('name') for v in root.find('.//microenvironment//domain//variables')] == ['oxygen', 'drug']
        types = root.findall('.//cellular_information//cell_populations//cell_population//custom//simplified_data//cell_types//type')
        assert [(t.get('ID'), t.text) for t in types] == [('0', 'tumor'), ('1', 'immune')]

    def test_substrates_from_config_without_microenv(self, tmp_path):
        root = ET.parse(self._write(tmp_path, microenv=False) / 'initial.xml').getroot()
        assert [v.get('name') for v in root.find('.//microenvironment//domain//variables')] == ['oxygen']


class TestFindReferenceConfig:
    def test_resolves_config_relative_to_ini(self, tmp_path):
        (tmp_path / 'config').mkdir()
        (tmp_path / 'config' / 'PhysiCell_settings.xml').write_text(_CONFIG_XML)
        (tmp_path / 'model.ini').write_text('[my_model]\nconfigFile_ref = ./config/PhysiCell_settings.xml\n')
        metadata = pd.DataFrame({'Ini_File_Path': ['model.ini'], 'StructureName': ['my_model']})
        with patch('uq_physicell.database.ma_db.load_metadata', return_value=metadata):
            path = find_reference_config(str(tmp_path / 'study.db'))
        assert path == str(tmp_path / 'config' / 'PhysiCell_settings.xml')

    def test_missing_ini_returns_none(self, tmp_path):
        metadata = pd.DataFrame({'Ini_File_Path': ['nope.ini'], 'StructureName': ['m']})
        with patch('uq_physicell.database.ma_db.load_metadata', return_value=metadata):
            assert find_reference_config(str(tmp_path / 'study.db')) is None


# ─── check_database ─────────────────────────────────────────────────────────

class _StoredTimeStep:
    """Picklable stand-in for a pcdl 4 TimeStep (only the data keys pc_studio reads from the database)."""
    def __init__(self, substrates=('oxygen',), celltypes=None, time=0.0, diffusion=100000.0, decay=0.1):
        df_sub = pd.DataFrame({'decay_rate': [decay] * len(substrates),
                               'diffusion_coefficient': [diffusion] * len(substrates)},
                              index=pd.Index(list(substrates), name='substrate'))
        self.time = time
        self.data = {'metadata': {}, 'cell': {'ds_celltype': celltypes or {}},
                     'mesh': {'xyz_range': [(-500.0, 500.0), (-500.0, 500.0), (-10.0, 10.0)],
                              'mnp_spacing': [20.0, 20.0, 20.0]},
                     'substrate': {'ls_substarte': list(substrates), 'df_substarte': df_sub}}

    def get_time(self):
        return self.time


def _ma_db(path, outputs):
    db = str(path)
    create_structure(db)
    insert_metadata(db, 'LHS', 'model.ini', 'my_model')
    for (sample_id, replicate_id), data in outputs.items():
        insert_output(db, sample_id, replicate_id, pickle.dumps(data))
    return db


class TestCheckDatabase:
    def test_raw_mcds_output(self, tmp_path):
        db = _ma_db(tmp_path / 'raw.db', {(0, 0): [_StoredTimeStep()] * 3, (0, 1): [_StoredTimeStep()] * 3})
        ok, msg = check_database(db)
        assert ok, msg
        assert '2 runs stored' in msg and '3 time steps' in msg and 'microenvironment stored' in msg

    def test_raw_output_without_microenv(self, tmp_path):
        ok, msg = check_database(_ma_db(tmp_path / 'raw.db', {(0, 0): [_StoredTimeStep(substrates=())]}))
        assert ok and 'microenvironment not stored' in msg

    def test_qoi_summaries_rejected(self, tmp_path):
        ok, msg = check_database(_ma_db(tmp_path / 'qoi.db', {(0, 0): pd.DataFrame({'q': [1.0]})}))
        assert not ok and 'QoI summaries' in msg

    def test_cell_dataframes_rejected(self, tmp_path):
        ok, msg = check_database(_ma_db(tmp_path / 'cells.db', {(0, 0): [pd.DataFrame({'ID': [0]})]}))
        assert not ok and 'drop_columns' in msg

    def test_run_without_data_rejected(self, tmp_path):
        db = _ma_db(tmp_path / 'raw.db', {(0, 0): [_StoredTimeStep()]})
        with sqlite3.connect(db) as conn:
            conn.execute('INSERT INTO Output (SampleID, ReplicateID, Data) VALUES (1, 0, NULL)')
        ok, msg = check_database(db)
        assert not ok and '1 of 2 stored runs have no data' in msg

    def test_no_runs_yet(self, tmp_path):
        ok, msg = check_database(_ma_db(tmp_path / 'empty.db', {}))
        assert not ok and 'No simulations' in msg

    def test_bo_database_rejected(self, tmp_path):
        db = str(tmp_path / 'bo.db')
        with sqlite3.connect(db) as conn:
            conn.execute('CREATE TABLE Metadata (BO_Method TEXT)')
        ok, msg = check_database(db)
        assert not ok and 'Bayesian optimization' in msg

    def test_missing_file(self, tmp_path):
        assert check_database(str(tmp_path / 'nope.db'))[0] is False


class TestModelConfigInfo:
    def _model(self, tmp_path, xml_hash):
        (tmp_path / 'config').mkdir()
        config = tmp_path / 'config' / 'PhysiCell_settings.xml'
        config.write_text(_CONFIG_XML)
        (tmp_path / 'model.ini').write_text('[my_model]\nconfigFile_ref = ./config/PhysiCell_settings.xml\n')
        metadata = pd.DataFrame({'Ini_File_Path': ['model.ini'], 'StructureName': ['my_model'],
                                 'XML_Hash': [_sha256_file(str(config)) if xml_hash == 'same' else xml_hash]})
        with patch('uq_physicell.database.ma_db.load_metadata', return_value=metadata):
            return model_config_info(str(tmp_path / 'study.db')), str(config)

    def test_match(self, tmp_path):
        info, config = self._model(tmp_path, 'same')
        assert info['status'] == 'match' and info['config_file'] == config
        assert info['ini_path'] == str(tmp_path / 'model.ini') and info['section'] == 'my_model'
        assert info['config_ref'] == './config/PhysiCell_settings.xml'

    def test_hash_differs(self, tmp_path):
        # no stored output to compare with here, so only the hash difference is known
        info = self._model(tmp_path, 'f' * 64)[0]
        assert info['status'] == 'edited' and info['differences'] == []

    def test_no_recorded_hash(self, tmp_path):
        assert self._model(tmp_path, None)[0]['status'] == 'unverifiable'

    def test_config_status_for_browsed_file(self, tmp_path):
        info, config = self._model(tmp_path, 'same')
        other = tmp_path / 'other.xml'
        other.write_text(_CONFIG_XML + ' ')
        assert config_status(info['xml_hash'], config) == 'match'
        assert config_status(info['xml_hash'], str(other)) == 'mismatch'
        assert config_status(info['xml_hash'], str(tmp_path / 'missing.xml')) == 'not_found'


class TestDatabaseSummary:
    def _db(self, tmp_path, param_space):
        outputs = {(0, 0): [_StoredTimeStep()], (0, 1): [_StoredTimeStep()], (1, 0): [_StoredTimeStep()]}
        db = _ma_db(tmp_path / 'study.db', outputs)
        insert_samples(db, {0: {'a': 0.1, 'b': 2.0}, 1: {'a': 0.3, 'b': 4.0}, 2: {'a': 0.5, 'b': 6.0}})
        if param_space:
            insert_param_space(db, {'a': {'lower_bound': 0.0, 'upper_bound': 1.0, 'ref_value': None, 'perturbation': None},
                                    'b': {'lower_bound': 1.0, 'upper_bound': 9.0, 'ref_value': None, 'perturbation': None}})
        return db

    def test_counts_and_sampler(self, tmp_path):
        summary = database_summary(self._db(tmp_path, param_space=True))
        assert summary['sampler'] == 'LHS'
        assert (summary['n_samples'], summary['n_samples_run'], summary['n_runs']) == (3, 2, 3)
        assert summary['replicates'] == (1, 2)
        assert [(p['name'], p['lower_bound'], p['upper_bound']) for p in summary['parameters']] == [('a', 0.0, 1.0), ('b', 1.0, 9.0)]

    def test_parameter_names_from_samples_without_parameter_space(self, tmp_path):
        params = database_summary(self._db(tmp_path, param_space=False))['parameters']
        assert sorted(p['name'] for p in params) == ['a', 'b']
        assert all(p['lower_bound'] is None and p['ref_value'] is None for p in params)


class TestOutputDifferences:
    _XML = """<PhysiCell_settings>
  <domain><x_min>-500</x_min><x_max>500</x_max><y_min>-500</y_min><y_max>500</y_max><z_min>-10</z_min>
    <z_max>10</z_max><dx>20</dx><dy>20</dy><dz>20</dz><use_2D>true</use_2D></domain>
  <overall><max_time units="min">{max_time}</max_time></overall>
  <save><full_data><interval units="min">{interval}</interval></full_data></save>
  <microenvironment_setup><variable name="{sub}" units="mmHg" ID="0"><physical_parameter_set>
    <diffusion_coefficient units="micron^2/min">{diffusion}</diffusion_coefficient>
    <decay_rate units="1/min">0.1</decay_rate></physical_parameter_set></variable></microenvironment_setup>
  <cell_definitions><cell_definition name="{cell}" ID="0"/></cell_definitions>
</PhysiCell_settings>"""

    def _setup(self, tmp_path):
        run = [_StoredTimeStep(celltypes={'0': 'tumor'}, time=t) for t in (0.0, 60.0, 120.0)]
        db = _ma_db(tmp_path / 'study.db', {(0, 0): run})
        return db, tmp_path

    def _xml(self, d, name='model.xml', max_time=120, interval=60, sub='oxygen', diffusion=100000.0, cell='tumor'):
        path = d / name
        path.write_text(self._XML.format(max_time=max_time, interval=interval, sub=sub, diffusion=diffusion, cell=cell))
        return str(path)

    def test_same_settings(self, tmp_path):
        db, d = self._setup(tmp_path)
        assert output_differences(db, self._xml(d)) == []

    def test_changed_settings_listed(self, tmp_path):
        db, d = self._setup(tmp_path)
        diffs = output_differences(db, self._xml(d, max_time=7200, interval=30, diffusion=500.0))
        assert ('save/full_data/interval', '30', '60.0') in diffs
        assert ('overall/max_time', '7200', '120 (last stored time)') in diffs
        assert ('substrate oxygen/diffusion_coefficient', '500.0', '100000.0') in diffs

    def test_ini_overrides_applied(self, tmp_path):
        db, d = self._setup(tmp_path)
        xml = self._xml(d, interval=30)
        assert output_differences(db, xml, {'.//save/full_data/interval': '60'}) == []

    def test_other_model(self, tmp_path):
        db, d = self._setup(tmp_path)
        diffs = output_differences(db, self._xml(d, sub='virus', cell='macrophage'))
        assert ('substrate virus', 'defined', None) in diffs and ('substrate oxygen', None, 'stored') in diffs
        assert ('cell type ID 0', 'macrophage', 'tumor') in diffs


class TestVerifyConfig:
    def _setup(self, tmp_path):
        helper = TestOutputDifferences()
        db, d = helper._setup(tmp_path)
        created = helper._xml(d, 'created.xml')
        return helper, db, d, _sha256_file(created)

    def test_same_xml_matches(self, tmp_path):
        helper, db, d, xml_hash = self._setup(tmp_path)
        assert verify_config(db, str(d / 'created.xml'), xml_hash) == ('match', [])

    def test_edited_without_recorded_difference(self, tmp_path):
        helper, db, d, xml_hash = self._setup(tmp_path)
        edited = d / 'edited.xml'
        edited.write_text((d / 'created.xml').read_text() + '<!-- comment -->')
        assert verify_config(db, str(edited), xml_hash) == ('edited', [])

    def test_no_hash_and_nothing_differs_is_unverifiable(self, tmp_path):
        helper, db, d, _ = self._setup(tmp_path)
        assert verify_config(db, str(d / 'created.xml'), None) == ('unverifiable', [])

    def test_recorded_setting_differs(self, tmp_path):
        helper, db, d, xml_hash = self._setup(tmp_path)
        status, diffs = verify_config(db, helper._xml(d, 'other.xml', max_time=7200), xml_hash)
        assert status == 'differs' and diffs == [('overall/max_time', '7200', '120 (last stored time)')]

    def test_names_differ_is_mismatch(self, tmp_path):
        helper, db, d, xml_hash = self._setup(tmp_path)
        assert verify_config(db, helper._xml(d, 'other.xml', cell='macrophage'), xml_hash)[0] == 'mismatch'

    def test_missing_file(self, tmp_path):
        helper, db, d, xml_hash = self._setup(tmp_path)
        assert verify_config(db, str(d / 'missing.xml'), xml_hash) == ('not_found', [])


class TestIniFixedParameters:
    def test_only_unsampled_parameters(self, tmp_path):
        ini = tmp_path / 'model.ini'
        ini.write_text('[m]\nparameters = {\n\t".//save/full_data/interval" : \'60\',\n'
                       '\t".//cell_definitions/x" : [None, \'rate\'],\n\t}\n')
        assert ini_fixed_parameters(str(ini), 'm') == {'.//save/full_data/interval': '60'}

    def test_missing_ini(self, tmp_path):
        assert ini_fixed_parameters(str(tmp_path / 'nope.ini'), 'm') == {}


class TestXMLDifferences:
    _A = """<PhysiCell_settings><overall><max_time units="min">7200</max_time></overall>
  <cell_definitions><cell_definition name="tumor" ID="0"><phenotype><cycle code="5">
    <phase_durations><duration index="0">1440</duration><duration index="1">300</duration></phase_durations>
  </cycle></phenotype></cell_definition></cell_definitions></PhysiCell_settings>"""

    def _files(self, tmp_path, other):
        a, b = tmp_path / 'a.xml', tmp_path / 'b.xml'
        a.write_text(self._A)
        b.write_text(other)
        return str(a), str(b)

    def test_identical(self, tmp_path):
        assert xml_differences(*self._files(tmp_path, self._A)) == []

    def test_changed_values_and_attributes(self, tmp_path):
        other = self._A.replace('7200', '2880').replace('units="min"', 'units="hour"').replace('>300<', '>999<')
        assert xml_differences(*self._files(tmp_path, other)) == [
            ('overall/max_time/@units', 'min', 'hour'),
            ('overall/max_time', '7200', '2880'),
            ("cell_definitions/cell_definition[@name='tumor']/phenotype/cycle/phase_durations/duration[2]", '300', '999'),
        ]

    def test_added_and_removed_settings(self, tmp_path):
        other = self._A.replace('<overall>', '<overall><dt>0.1</dt>').replace('<duration index="1">300</duration>', '')
        diffs = xml_differences(*self._files(tmp_path, other))
        assert ('overall/dt', None, '0.1') in diffs
        assert any(first == '300' and second is None for _, first, second in diffs)
