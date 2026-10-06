"""
Adapter that lets PhysiCell Studio display simulations stored in a UQ-PhysiCell database.

When a model-analysis run stores raw output (no QoI functions), each Output row holds
a pickled list of pcdl 4 ``TimeStep`` objects. PhysiCell Studio, however, ships its own
``pyMCDS`` reader whose ``data`` dict follows the older MultiCellDS layout
(``metadata`` / ``mesh`` / ``continuum_variables`` / ``discrete_cells``).

This module converts a pcdl ``TimeStep`` into that layout, so Studio can stay free of any
pcdl or uq_physicell dependency. Studio registers the loader for one output folder through
an optional hook in its ``pyMCDS.py`` (``set_frame_loader(output_dir, loader)``); every
``pyMCDS(xmlfile, output_dir, ...)`` in that folder then gets its data from the loader.

The folder itself is a "virtual" output folder written by ``write_index_folder``: small
MultiCellDS XMLs (time, mesh, substrates, cell types) and a copy of the model config, so
Studio's own file listing and XML parsing work unchanged while the heavy data stays in the
database::

    from uq_physicell.utils.pc_studio import StudioFrameLoader
    loader = StudioFrameLoader.from_database('ma.db', sample_id=3, replicate_id=0)
    loader.list_frames()                                   # ['output00000000.xml', ...]
    loader.write_index_folder('/tmp/uq_view', config_file='PhysiCell_settings.xml')
    data = loader('output00000005.xml', '/tmp/uq_view', microenv=True, graph=False)
"""
from __future__ import annotations
import configparser
import math
import os
import re
import shutil
import warnings
import xml.etree.ElementTree as ET
from typing import Union

import numpy as np
from pcdl.timestep import ds_cycle_model, ds_death_model, ds_cycle_phase, ds_death_phase

# Cell variables PhysiCell writes as one value per substrate / per cell type.
# Studio labels them '<attr>' (if a single substrate/cell type) or '<attr>_<index>'.
SUBSTRATE_INDEXED_ATTRS = (
    'chemotactic_sensitivities', 'secretion_rates', 'uptake_rates', 'saturation_densities',
    'net_export_rates', 'internalized_total_substrates', 'fraction_released_at_death',
    'fraction_transferred_when_ingested',
)
CELLTYPE_INDEXED_ATTRS = (
    'cell_adhesion_affinities', 'live_phagocytosis_rates', 'attack_rates', 'immunogenicities',
    'fusion_rates', 'transformation_rates',
)

# Columns pcdl derives on load that are not part of the PhysiCell cell output.
PCDL_DERIVED_COLUMNS = {
    'voxel_i', 'voxel_j', 'voxel_k', 'mesh_center_m', 'mesh_center_n', 'mesh_center_p',
    'cell_count_voxel', 'cell_density_micron3', 'time', 'runtime', 'xmlfile',
}

def _inverse(*codecs: dict) -> dict:
    return {label: float(code) for codec in codecs for code, label in codec.items()}

# pcdl decodes these integer codes into labels; Studio expects the numeric codes.
# Dead cells carry their death model in 'cycle_model' and a death phase in 'current_phase'.
_CODEC_COLUMNS = {
    'cycle_model': _inverse(ds_cycle_model, ds_death_model),
    'current_death_model': _inverse(ds_death_model),
    'current_phase': _inverse(ds_cycle_phase, ds_death_phase),
}

_PADDED_INDEX = re.compile(r'^(.*)_(\d{3})$')
_OUTPUT_XML = re.compile(r'^output(\d+)\.xml$')


def _to_float_array(series, mapping: Union[dict, None] = None) -> np.ndarray:
    """Return a column as a float array, mapping string labels back to numeric codes."""
    if series.dtype.kind in 'biuf':
        return series.to_numpy(dtype=np.float64)
    values = series.astype(object)
    if mapping:
        values = values.map(lambda v: mapping.get(v, v))
    out = np.full(len(values), np.nan)
    for i, v in enumerate(values):
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def _indexed_columns(df_columns, attr: str, labels: list) -> list:
    """Find the pcdl columns of a per-substrate/per-cell-type attribute, in index order.

    pcdl names them '<label>_<attr>' when the labels are known (e.g. 'oxygen_secretion_rates'),
    otherwise '<attr>_<index>' (e.g. 'secretion_rates_0').
    """
    named = [f'{label}_{attr}' for label in labels]
    if labels and all(col in df_columns for col in named):
        return named
    pattern = re.compile(rf'^{re.escape(attr)}_(\d+)$')
    numbered = [(int(m.group(1)), col) for col in df_columns if (m := pattern.match(col))]
    return [col for _, col in sorted(numbered)]


def _studio_cell_columns(df_cell, substrates: list, celltypes: list) -> dict:
    """Map pcdl cell-dataframe column names to Studio's ``discrete_cells`` labels."""
    columns = list(df_cell.columns)
    rename = {}
    for attrs, labels in ((SUBSTRATE_INDEXED_ATTRS, substrates), (CELLTYPE_INDEXED_ATTRS, celltypes)):
        for attr in attrs:
            cols = _indexed_columns(columns, attr, labels)
            if len(cols) == 1:
                rename[cols[0]] = attr
            else:
                rename.update({col: f'{attr}_{i}' for i, col in enumerate(cols)})

    # Columns pcdl adds from the microenvironment: the substrate concentration at the
    # cell's voxel, and per-substrate decay/diffusion constants.
    derived = set(PCDL_DERIVED_COLUMNS) | set(substrates)
    derived |= {f'{s}_{p}' for s in substrates for p in ('decay_rate', 'diffusion_coefficient')}

    mapping = {}
    for col in columns:
        if col in rename:
            mapping[col] = rename[col]
        elif col in derived or col.endswith('_vectorlength'):
            continue
        elif (m := _PADDED_INDEX.match(col)):
            mapping[col] = f'{m.group(1)}_{int(m.group(2))}'  # previous_velocity_000 -> previous_velocity_0
        else:
            mapping[col] = col
    return mapping


def timestep_to_studio_data(mcds, microenv: bool = True, graph: bool = False) -> dict:
    """Convert a pcdl 4 ``TimeStep`` into the ``data`` dict of PhysiCell Studio's ``pyMCDS``.

    Args:
        mcds: pcdl ``TimeStep`` (e.g. one element of ``TimeSeries.get_mcds_list()``).
        microenv: Whether to build ``continuum_variables``. If the TimeStep was loaded
            without microenvironment data, an empty dict is returned for it.
        graph: Whether to build ``discrete_cells['graph']`` from the stored cell graphs.

    Returns:
        dict: ``{'metadata', 'mesh', 'discrete_cells'[, 'continuum_variables']}`` matching
        Studio's ``pyMCDS._read_xml`` output.
    """
    src = mcds.data
    if not all(key in src for key in ('metadata', 'mesh', 'cell', 'substrate')):
        raise TypeError(f'Expected a pcdl 4 TimeStep, got {type(mcds).__name__} with data keys {list(src)}.')
    units = src['metadata'].get('ds_unit', {})

    # metadata
    meta = src['metadata']
    data = {'metadata': {
        key: meta[key] for key in ('multicellds_version', 'physicell_version', 'created', 'current_time',
                                   'time_units', 'current_runtime', 'runtime_units') if key in meta
    }}
    data['metadata']['spatial_units'] = meta.get('spatial_unit')

    # mesh: Studio keeps one volume per voxel instead of a single scalar
    mesh = src['mesh']
    data['mesh'] = {key: mesh[key] for key in ('mnp_grid', 'mnp_axis', 'mnp_range', 'ijk_range',
                                               'ijk_axis', 'xyz_range', 'mnp_coordinate')}
    data['mesh']['volumes'] = np.full(mesh['mnp_coordinate'].shape[1], float(mesh['volume']))

    # microenvironment: rebuild meshgrid-shaped arrays indexed [j, i, k]
    substrates = list(src['substrate'].get('ls_substarte', []))
    if microenv:
        data['continuum_variables'] = {}
        df_conc = src['substrate'].get('df_conc')
        if substrates and df_conc is not None and len(df_conc):
            ii = df_conc['voxel_i'].to_numpy(dtype=int)
            jj = df_conc['voxel_j'].to_numpy(dtype=int)
            kk = df_conc['voxel_k'].to_numpy(dtype=int)
            df_sub = src['substrate'].get('df_substarte')
            for sub in substrates:
                field = np.zeros(mesh['mnp_grid'][0].shape)
                field[jj, ii, kk] = df_conc[sub].to_numpy(dtype=np.float64)
                entry = {'units': units.get(sub), 'data': field}
                for param in ('diffusion_coefficient', 'decay_rate'):
                    value = float(df_sub.loc[sub, param]) if df_sub is not None and sub in df_sub.index else np.nan
                    entry[param] = {'value': value, 'units': units.get(f'{sub}_{param}')}
                data['continuum_variables'][sub.replace(' ', '_')] = entry
        else:
            warnings.warn('TimeStep has no microenvironment data (stored with microenv=False); '
                          'substrate views will be empty.', stacklevel=2)

    # cells: dict of float arrays, ID first, same labels Studio derives from the output XML
    df_cell = src['cell']['df_cell']
    celltypes = list(src['cell'].get('ls_celltype', []))
    ds_celltype = src['cell'].get('ds_celltype', {})
    ds_substrate = src['substrate'].get('ds_substrate', {})
    codecs = dict(_CODEC_COLUMNS)
    codecs['cell_type'] = {label: float(code) for code, label in ds_celltype.items()}
    codecs['chemotaxis_index'] = {label: float(code) for code, label in ds_substrate.items()}

    cells = {'ID': df_cell.index.to_numpy(dtype=np.float64)}
    cell_units = {'ID': 'none'}
    for col, label in _studio_cell_columns(df_cell, substrates, celltypes).items():
        if label == 'ID':
            continue
        cells[label] = _to_float_array(df_cell[col], codecs.get(col))
        padded = _PADDED_INDEX.match(col)
        cell_units[label] = units.get(col, units.get(label, units.get(padded.group(1) if padded else label, 'none')))
    data['discrete_cells'] = {'units': cell_units, 'data': cells}

    if graph:
        dei_graph = src['cell'].get('dei_graph', {}) or {}
        data['discrete_cells']['graph'] = {
            'neighbor_cells': dei_graph.get('neighbor_cells', {}),
            'attached_cells': dei_graph.get('attached_cells', {}),
        }
    return data


class StudioFrameLoader:
    """Per-frame loader for PhysiCell Studio, backed by a list of pcdl ``TimeStep`` objects.

    Calling the instance with Studio's ``pyMCDS`` arguments returns the ``data`` dict for
    that frame, so it can be registered as Studio's frame loader. Frames are converted on
    demand, one at a time.

    Args:
        mcds_list: List of pcdl ``TimeStep`` objects ordered by time.
    """

    def __init__(self, mcds_list: list):
        if not isinstance(mcds_list, (list, tuple)) or not mcds_list:
            raise ValueError('mcds_list must be a non-empty list of pcdl TimeStep objects.')
        self.mcds_list = list(mcds_list)
        self._frame_names = [self._frame_name(mcds, i) for i, mcds in enumerate(self.mcds_list)]
        self._index = {name: i for i, name in enumerate(self._frame_names)}

    @staticmethod
    def _frame_name(mcds, position: int) -> str:
        df_cell = getattr(mcds, 'data', {}).get('cell', {}).get('df_cell')
        if df_cell is not None and len(df_cell) and 'xmlfile' in df_cell.columns:
            return os.path.basename(str(df_cell['xmlfile'].iloc[0]))
        return f'output{position:08d}.xml'

    @classmethod
    def from_database(cls, db_file: str, sample_id: int, replicate_id: int) -> 'StudioFrameLoader':
        """Build a loader from the raw output stored for one (SampleID, ReplicateID).

        Raises:
            ValueError: If the row is missing or does not hold a list of pcdl TimeStep objects
                (e.g. the run stored QoI DataFrames instead of raw output).
        """
        from uq_physicell.database.ma_db import load_output
        df_output = load_output(db_file, sample_ids=[sample_id], replicate_ids=[replicate_id])
        if df_output.empty:
            raise ValueError(f'No output for SampleID={sample_id}, ReplicateID={replicate_id} in {db_file}.')
        mcds_list = df_output['Data'].iloc[0]
        problem = _raw_output_problem(mcds_list)
        if problem:
            raise ValueError(f'SampleID={sample_id}, ReplicateID={replicate_id} does not store raw MCDS output '
                             f'({problem}); rerun the analysis without QoI functions.')
        return cls(mcds_list)

    def __len__(self) -> int:
        return len(self.mcds_list)

    def list_frames(self) -> list:
        """Frame file names in time order, as Studio would glob them from an output folder."""
        return list(self._frame_names)

    def frame_index(self, xmlfile: str) -> int:
        """Resolve an output XML name (with or without path) to an index in ``mcds_list``."""
        name = os.path.basename(str(xmlfile))
        if name in self._index:
            return self._index[name]
        if name == 'initial.xml':
            return 0
        if name == 'final.xml':
            return len(self.mcds_list) - 1
        m = _OUTPUT_XML.match(name)
        if m and int(m.group(1)) < len(self.mcds_list):
            return int(m.group(1))
        raise KeyError(f'Frame {name!r} is not available; stored frames: {self._frame_names[0]} .. {self._frame_names[-1]}.')

    def get_mcds(self, xmlfile: str):
        """Return the stored pcdl TimeStep for a frame."""
        return self.mcds_list[self.frame_index(xmlfile)]

    def __call__(self, xmlfile: str, output_path: str = '.', microenv: bool = True, graph: bool = True) -> dict:
        """Studio frame-loader signature: same arguments as Studio's ``pyMCDS(...)``.

        ``output_path`` is accepted for compatibility and ignored.
        """
        return timestep_to_studio_data(self.get_mcds(xmlfile), microenv=microenv, graph=graph)

    def write_index_folder(self, out_dir: str, config_file: Union[str, None] = None) -> str:
        """Write a virtual output folder that PhysiCell Studio can open with the frame loader.

        For every stored frame this writes an index ``outputNNNNNNNN.xml`` (plus ``initial.xml``
        and ``final.xml``) holding time, mesh, substrates and cell types, without the ``.mat``
        data files. If given, ``config_file`` is copied as ``PhysiCell_settings.xml``; it is also
        used for substrate names when the run was stored without microenvironment data.

        Args:
            out_dir: Folder to write into (created if needed).
            config_file: PhysiCell settings XML of the model (optional).

        Returns:
            str: ``out_dir``.
        """
        os.makedirs(out_dir, exist_ok=True)
        if config_file:
            shutil.copyfile(config_file, os.path.join(out_dir, 'PhysiCell_settings.xml'))
        config_substrates = _config_substrates(config_file) if config_file else []
        frames = self.list_frames()
        for i, (name, mcds) in enumerate(zip(frames, self.mcds_list)):
            tree = _index_xml(mcds, name, config_substrates)
            tree.write(os.path.join(out_dir, name), encoding='utf-8', xml_declaration=True)
            if i == 0:
                tree.write(os.path.join(out_dir, 'initial.xml'), encoding='utf-8', xml_declaration=True)
            if i == len(frames) - 1:
                tree.write(os.path.join(out_dir, 'final.xml'), encoding='utf-8', xml_declaration=True)
        return out_dir


def _config_substrates(config_file: str) -> list:
    """Substrates (name, units, diffusion, decay) declared in a PhysiCell settings XML."""
    root = ET.parse(config_file).getroot()
    substrates = []
    for var in root.findall('.//microenvironment_setup/variable'):
        pps = var.find('physical_parameter_set')
        diff = pps.find('diffusion_coefficient') if pps is not None else None
        decay = pps.find('decay_rate') if pps is not None else None
        substrates.append({
            'name': var.get('name'), 'units': var.get('units', 'dimensionless'),
            'diffusion_coefficient': (diff.text.strip() if diff is not None else '0', diff.get('units', '') if diff is not None else ''),
            'decay_rate': (decay.text.strip() if decay is not None else '0', decay.get('units', '') if decay is not None else ''),
        })
    return substrates


def _index_xml(mcds, frame_name: str, config_substrates: list) -> ET.ElementTree:
    """MultiCellDS XML for one frame, without the cell/microenvironment ``.mat`` payload."""
    meta, mesh = mcds.data['metadata'], mcds.data['mesh']
    stem = os.path.splitext(frame_name)[0]
    units = meta.get('ds_unit', {})
    spatial = meta.get('spatial_unit') or 'micron'

    root = ET.Element('MultiCellDS', version='2', type='snapshot/simulation')
    md = ET.SubElement(root, 'metadata')
    software = ET.SubElement(md, 'software')
    sw_name, _, sw_version = str(meta.get('physicell_version', 'PhysiCell_')).partition('_')
    ET.SubElement(software, 'name').text = sw_name or 'PhysiCell'
    ET.SubElement(software, 'version').text = sw_version
    ET.SubElement(md, 'current_time', units=meta.get('time_units') or 'min').text = repr(float(meta['current_time']))
    ET.SubElement(md, 'current_runtime', units=meta.get('runtime_units') or 'sec').text = repr(float(meta.get('current_runtime', 0.0)))
    ET.SubElement(md, 'created').text = str(meta.get('created', ''))
    ET.SubElement(md, 'last_modified').text = str(meta.get('created', ''))

    domain = ET.SubElement(ET.SubElement(root, 'microenvironment'), 'domain', name='microenvironment')
    mesh_node = ET.SubElement(domain, 'mesh', type='Cartesian', uniform='true', regular='true', units=spatial)
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = mesh['xyz_range']
    ET.SubElement(mesh_node, 'bounding_box', type='axis-aligned', units=spatial).text = \
        ' '.join(f'{v:.6f}' for v in (xmin, ymin, zmin, xmax, ymax, zmax))
    for axis, values in zip('xyz', mesh['mnp_axis']):
        ET.SubElement(mesh_node, f'{axis}_coordinates', delimiter=' ').text = ' '.join(f'{v:g}' for v in values)
    ET.SubElement(ET.SubElement(mesh_node, 'voxels', type='matlab'), 'filename').text = 'initial_mesh0.mat'

    variables = ET.SubElement(domain, 'variables')
    stored = list(mcds.data['substrate'].get('ls_substarte', []))
    if stored:
        df_sub = mcds.data['substrate'].get('df_substarte')
        substrates = [{
            'name': sub, 'units': units.get(sub, 'dimensionless'),
            'diffusion_coefficient': (repr(float(df_sub.loc[sub, 'diffusion_coefficient'])) if df_sub is not None else '0',
                                      units.get(f'{sub}_diffusion_coefficient', '')),
            'decay_rate': (repr(float(df_sub.loc[sub, 'decay_rate'])) if df_sub is not None else '0',
                           units.get(f'{sub}_decay_rate', '')),
        } for sub in stored]
    else:
        substrates = config_substrates
    for i, sub in enumerate(substrates):
        var = ET.SubElement(variables, 'variable', name=sub['name'], units=sub['units'], ID=str(i))
        pps = ET.SubElement(var, 'physical_parameter_set')
        ET.SubElement(pps, 'conditions')
        for param in ('diffusion_coefficient', 'decay_rate'):
            value, unit = sub[param]
            ET.SubElement(pps, param, units=unit).text = value
    ET.SubElement(ET.SubElement(domain, 'data', type='matlab'), 'filename').text = f'{stem}_microenvironment0.mat'

    population = ET.SubElement(ET.SubElement(ET.SubElement(root, 'cellular_information'), 'cell_populations'),
                               'cell_population', type='individual')
    simplified = ET.SubElement(ET.SubElement(population, 'custom'), 'simplified_data',
                               type='matlab', source='PhysiCell', data_version='2')
    cell_types = ET.SubElement(simplified, 'cell_types')
    for code, label in sorted(mcds.data['cell'].get('ds_celltype', {}).items(), key=lambda kv: int(kv[0])):
        ET.SubElement(cell_types, 'type', ID=str(code), type=str(code)).text = label
    ET.SubElement(simplified, 'labels')
    ET.SubElement(simplified, 'filename').text = f'{stem}_cells.mat'

    ET.indent(root, space='\t')
    return ET.ElementTree(root)


def _is_pcdl4_timestep(obj) -> bool:
    data = getattr(obj, 'data', None)
    return isinstance(data, dict) and all(key in data for key in ('metadata', 'mesh', 'cell', 'substrate'))


def _raw_output_problem(stored) -> Union[str, None]:
    """Why a stored Output 'Data' value is not a list of pcdl 4 TimeStep objects (None if it is)."""
    if isinstance(stored, list):
        if not stored:
            return 'empty list'
        if all(_is_pcdl4_timestep(item) for item in stored):
            return None
        kinds = sorted({type(item).__name__ for item in stored})
        if kinds == ['DataFrame']:
            return 'cell DataFrames per time step (drop_columns), not MCDS objects'
        return f'list of {", ".join(kinds)}, not pcdl 4 TimeStep objects'
    if stored is None:
        return 'no data'
    if type(stored).__name__ == 'DataFrame':
        return 'QoI summaries (DataFrame)'
    return f'{type(stored).__name__}'


def check_database(db_file: str) -> tuple:
    """Check that a database can be opened in PhysiCell Studio, i.e. it stores the raw
    simulation output (a list of pcdl 4 TimeStep objects per SampleID/ReplicateID).

    Unpickles a single stored run; every other run is only checked for non-empty data.

    Returns:
        tuple: ``(ok, message)``; ``message`` explains the problem, or summarizes the runs when ok.
    """
    import sqlite3
    from uq_physicell.database.ma_db import get_database_type, load_output
    if not os.path.isfile(db_file):
        return False, f'{db_file} does not exist.'
    db_type = get_database_type(db_file)
    if db_type == 'BO':
        return False, 'Bayesian optimization database: it stores objective values, not simulation output.'
    if db_type != 'MA':
        return False, ('Not a UQ-PhysiCell model-analysis database (no Metadata table with a Sampler); '
                       'ABC and data-assimilation databases are not supported.')
    try:
        conn = sqlite3.connect(db_file)
        try:
            n_runs, n_empty = conn.execute(
                'SELECT COUNT(*), SUM(CASE WHEN Data IS NULL OR length(Data) = 0 THEN 1 ELSE 0 END) FROM Output'
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error as e:
        return False, f'Unable to read the Output table: {e}'
    if not n_runs:
        return False, 'No simulations stored yet.'
    if n_empty:
        return False, f'{n_empty} of {n_runs} stored runs have no data.'
    probe = load_output(db_file, load_data=False).iloc[0]
    sample_id, replicate_id = int(probe['SampleID']), int(probe['ReplicateID'])
    stored = load_output(db_file, sample_ids=[sample_id], replicate_ids=[replicate_id])['Data'].iloc[0]
    problem = _raw_output_problem(stored)
    if problem:
        return False, (f'It stores {problem} instead of the raw simulation output. '
                       'Run the analysis without QoI functions to store MCDS objects.')
    has_microenv = bool(stored[0].data['substrate'].get('ls_substarte'))
    return True, (f'{n_runs} runs stored; SampleID={sample_id}, ReplicateID={replicate_id} has {len(stored)} '
                  f'time steps, microenvironment {"stored" if has_microenv else "not stored"}.')


def list_runs(db_file: str):
    """(SampleID, ReplicateID) pairs stored in a model-analysis database, sorted.

    Includes a 'Seed' column (PhysiCell random seed per replicate) when the database records it.
    """
    from uq_physicell.database.ma_db import load_output
    try:
        df = load_output(db_file, load_data=False, load_seed=True)[['SampleID', 'ReplicateID', 'Seed']]
    except Exception:  # databases created before seeds were recorded
        df = load_output(db_file, load_data=False)[['SampleID', 'ReplicateID']]
    return df.sort_values(['SampleID', 'ReplicateID']).reset_index(drop=True)


def database_summary(db_file: str, runs=None) -> dict:
    """Overview of a model-analysis database for display (e.g. PhysiCell Studio).

    Args:
        db_file: Path to the database.
        runs: Output of ``list_runs(db_file)``, if already loaded.

    Returns:
        dict: ``sampler``; ``n_samples`` (defined in Samples), ``n_samples_run`` (with stored
        output), ``n_runs`` (stored simulations), ``replicates`` ((min, max) per sample) and
        ``parameters``: list of dicts with ``name``, ``lower_bound``, ``upper_bound``,
        ``ref_value`` and ``perturbation`` (None where not defined).
    """
    import sqlite3
    from uq_physicell.database.ma_db import load_metadata, load_parameter_space
    runs = list_runs(db_file) if runs is None else runs
    try:
        sampler = load_metadata(db_file).iloc[0].get('Sampler')
    except Exception:
        sampler = None
    conn = sqlite3.connect(db_file)
    try:
        n_samples = conn.execute('SELECT COUNT(DISTINCT SampleID) FROM Samples').fetchone()[0]
        sample_param_names = [r[0] for r in conn.execute('SELECT DISTINCT ParamName FROM Samples')]
    except sqlite3.Error:
        n_samples, sample_param_names = None, []
    finally:
        conn.close()
    keys = ('lower_bound', 'upper_bound', 'ref_value', 'perturbation')
    try:
        parameters = [{'name': row['ParamName'], **{k: row.get(k) for k in keys}}
                      for _, row in load_parameter_space(db_file).iterrows()]
    except Exception:
        parameters = []
    if not parameters:   # e.g. user-defined samples: no parameter space, names from Samples
        parameters = [{'name': name, **{k: None for k in keys}} for name in sample_param_names]
    per_sample = runs.groupby('SampleID').size() if len(runs) else None
    return {
        'sampler': sampler,
        'n_samples': n_samples,
        'n_samples_run': int(runs['SampleID'].nunique()),
        'n_runs': int(len(runs)),
        'replicates': (int(per_sample.min()), int(per_sample.max())) if per_sample is not None else (0, 0),
        'parameters': parameters,
    }


def sample_parameters(db_file: str) -> dict:
    """Input parameter values of every sample, with their bounds when defined.

    Returns:
        dict: ``{SampleID: [(name, value, lower_bound, upper_bound), ...]}``; bounds are None
        when the parameter space does not define them.
    """
    from uq_physicell.database.ma_db import load_parameter_space, load_samples
    samples = load_samples(db_file)
    bounds = {}
    try:
        for _, row in load_parameter_space(db_file).iterrows():
            bounds[row['ParamName']] = (row.get('lower_bound'), row.get('upper_bound'))
    except Exception:
        pass
    table = {}
    for sample_id, params in samples.items():
        table[int(sample_id)] = [(name, value, *bounds.get(name, (None, None)))
                                 for name, value in params.items()]
    return table


def config_status(xml_hash: Union[str, None], config_file: Union[str, None]) -> str:
    """Compare a PhysiCell settings XML with the XML_Hash recorded in a database.

    Returns:
        str: 'match', 'mismatch', 'unverified' (the database has no XML_Hash) or 'not_found'.
    """
    from uq_physicell.pc_model import _sha256_file
    if not config_file or not os.path.isfile(config_file):
        return 'not_found'
    if not xml_hash:
        return 'unverified'
    return 'match' if _sha256_file(config_file) == xml_hash else 'mismatch'


def _stored_run(db_file: str) -> list:
    """TimeStep list of the first stored run (what the database records about the model)."""
    from uq_physicell.database.ma_db import load_output
    probe = load_output(db_file, load_data=False).iloc[0]
    return load_output(db_file, sample_ids=[int(probe['SampleID'])],
                       replicate_ids=[int(probe['ReplicateID'])])['Data'].iloc[0]


def ini_fixed_parameters(ini_path: Union[str, None], section: Union[str, None]) -> dict:
    """XML settings the INI section sets for every run (``parameters`` entries that are not
    sampled), as ``{xpath: value}``; these override the reference XML in each simulation."""
    import ast
    if not ini_path or not section:
        return {}
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read(ini_path)
        parameters = ast.literal_eval(parser[section].get('parameters', '{}'))
    except Exception:
        return {}
    return {xpath: value for xpath, value in parameters.items() if not isinstance(value, list)}


def _same(a, b) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    except (TypeError, ValueError):
        return str(a) == str(b)


def output_differences(db_file: str, config_file: str, fixed_parameters: Union[dict, None] = None) -> list:
    """Settings of a PhysiCell settings XML that disagree with what the stored simulation output
    records: domain bounds and voxel size, save interval, max_time, substrates (names, diffusion
    coefficient, decay rate) and cell types. ``fixed_parameters`` (see ``ini_fixed_parameters``)
    are applied to the XML first, as they are in every run.

    Returns:
        list: ``(setting, xml_value, stored_value)`` for each disagreement; a value is None when the
        setting exists on one side only.
    """
    root = ET.parse(config_file).getroot()
    for xpath, value in (fixed_parameters or {}).items():
        for element in root.findall(xpath):
            element.text = str(value)
    text = lambda path: (root.findtext(path) or '').strip() or None
    run = _stored_run(db_file)
    data = run[0].data
    diffs = []

    def check(setting, xml_value, stored_value):
        if xml_value is not None and stored_value is not None and not _same(xml_value, stored_value):
            diffs.append((setting, str(xml_value), str(stored_value)))

    # domain and mesh
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = data['mesh']['xyz_range']
    for key, stored in zip(('x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'), (xmin, xmax, ymin, ymax, zmin, zmax)):
        check(f'domain/{key}', text(f'.//domain/{key}'), float(stored))
    spacing = data['mesh'].get('mnp_spacing') or []
    is_2d = (text('.//domain/use_2D') or '').lower() == 'true'
    for key, stored in zip(('dx', 'dy', 'dz'), spacing):
        if not (key == 'dz' and is_2d):
            check(f'domain/{key}', text(f'.//domain/{key}'), float(stored))

    # output times
    times = [step.get_time() for step in run]
    if len(times) > 1:
        intervals = np.diff(times)
        check('save/full_data/interval', text('.//save/full_data/interval'), float(np.median(intervals)))
        max_time = text('.//overall/max_time')
        if max_time is not None and abs(float(max_time) - times[-1]) >= float(np.median(intervals)):
            diffs.append(('overall/max_time', max_time, f'{times[-1]:g} (last stored time)'))

    # substrates
    stored_subs = list(data['substrate'].get('ls_substarte', []))
    if stored_subs:
        xml_vars = {v.get('name'): v for v in root.findall('.//microenvironment_setup/variable')}
        df_sub = data['substrate'].get('df_substarte')
        for name in list(xml_vars) + [n for n in stored_subs if n not in xml_vars]:
            if name not in stored_subs or name not in xml_vars:
                diffs.append((f'substrate {name}', 'defined' if name in xml_vars else None,
                              'stored' if name in stored_subs else None))
                continue
            for param in ('diffusion_coefficient', 'decay_rate'):
                xml_value = (xml_vars[name].findtext(f'physical_parameter_set/{param}') or '').strip() or None
                stored = df_sub.loc[name, param] if df_sub is not None and name in df_sub.index else None
                check(f'substrate {name}/{param}', xml_value, stored)

    # cell types
    stored_types = {str(k): str(v) for k, v in data['cell'].get('ds_celltype', {}).items()}
    if stored_types and any(k != v for k, v in stored_types.items()):   # output with cell type names
        xml_types = {cd.get('ID'): cd.get('name') for cd in root.findall('.//cell_definitions/cell_definition')}
        for code in sorted(set(stored_types) | set(xml_types), key=lambda c: int(c) if str(c).isdigit() else c):
            if stored_types.get(code) != xml_types.get(code):
                diffs.append((f'cell type ID {code}', xml_types.get(code), stored_types.get(code)))
    return diffs


def verify_config(db_file: str, config_file: Union[str, None], xml_hash: Union[str, None] = None,
                  fixed_parameters: Union[dict, None] = None) -> tuple:
    """Check a settings XML against a database: its XML_Hash, and the settings the stored output
    records (see ``output_differences``).

    Returns:
        tuple: ``(status, differences)``. status is 'match' (same XML as recorded by XML_Hash),
        'differs' (settings disagree with the stored output), 'mismatch' (cell types or substrates
        disagree), 'edited' (XML_Hash differs but no recorded setting does), 'unverifiable' (no
        XML_Hash and no recorded setting differs) or 'not_found'; differences as returned by
        ``output_differences``.
    """
    status = config_status(xml_hash, config_file)
    if status in ('match', 'not_found'):
        return status, []
    try:
        diffs = output_differences(db_file, config_file, fixed_parameters)
    except Exception:
        return ('edited' if xml_hash else 'unverifiable'), []
    if any(setting.startswith(('cell type', 'substrate ')) and '/' not in setting for setting, _, _ in diffs):
        return 'mismatch', diffs
    if diffs:
        return 'differs', diffs
    return ('edited' if xml_hash else 'unverifiable'), []


def _flatten_xml(root) -> dict:
    """{path: value} for every leaf text and attribute of an XML tree. Elements are addressed by
    their 'name' attribute when they have one (cell definitions, substrates, ...), otherwise by
    position among same-tag siblings."""
    flat = {}

    def visit(element, path):
        for attr, value in element.attrib.items():
            if attr != 'name':
                flat[f'{path}/@{attr}'] = value
        children = list(element)
        if not children:
            text = (element.text or '').strip()
            if text:
                flat[path] = text
        counts = {}
        for child in children:
            counts[child.tag] = counts.get(child.tag, 0) + 1
        seen = {}
        for child in children:
            if 'name' in child.attrib:
                key = f"{child.tag}[@name='{child.get('name')}']"
            elif counts[child.tag] > 1:
                seen[child.tag] = seen.get(child.tag, 0) + 1
                key = f'{child.tag}[{seen[child.tag]}]'
            else:
                key = child.tag
            visit(child, f'{path}/{key}' if path else key)

    visit(root, '')
    return flat


def xml_differences(xml_file: str, other_file: str) -> list:
    """Settings that differ between two PhysiCell settings XML files (e.g. the database's model
    XML and the config loaded in PhysiCell Studio).

    Returns:
        list: ``(path, value, other_value)`` in document order; a value is None when the setting
        exists in only one of the two files.
    """
    first = _flatten_xml(ET.parse(xml_file).getroot())
    other = _flatten_xml(ET.parse(other_file).getroot())
    paths = list(first) + [p for p in other if p not in first]
    return [(p, first.get(p), other.get(p)) for p in paths if first.get(p) != other.get(p)]


def model_config_info(db_file: str) -> dict:
    """Model configuration recorded in a database: INI file, model section and settings XML.

    The INI path and section come from the database Metadata; the XML is the section's
    ``configFile_ref`` resolved against the INI and database folders, and is compared with
    the XML_Hash the database recorded when it was created.

    Returns:
        dict: ``ini_file`` (as recorded), ``ini_path`` (resolved, or None), ``section``,
        ``config_ref`` (as written in the INI, or None), ``config_file`` (resolved, or None),
        ``xml_hash`` (recorded, or None), ``fixed_parameters`` (INI overrides, see
        ``ini_fixed_parameters``), ``status`` and ``differences`` (see ``verify_config``).
    """
    from uq_physicell.database.ma_db import load_metadata
    info = {'ini_file': None, 'ini_path': None, 'section': None, 'config_ref': None,
            'config_file': None, 'xml_hash': None, 'fixed_parameters': {}, 'status': 'not_found',
            'differences': []}
    try:
        metadata = load_metadata(db_file).iloc[0]
    except Exception:
        return info
    info['ini_file'], info['section'] = metadata.get('Ini_File_Path'), metadata.get('StructureName')
    info['xml_hash'] = metadata.get('XML_Hash') or None
    db_dir = os.path.dirname(os.path.abspath(db_file))
    if info['ini_file']:
        candidates = [os.path.join(db_dir, info['ini_file']), info['ini_file']]
        info['ini_path'] = next((os.path.abspath(p) for p in candidates if os.path.isfile(p)), None)
    if info['ini_path'] and info['section']:
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read(info['ini_path'])
            info['config_ref'] = parser[info['section']]['configFile_ref']
        except Exception:
            pass
    if info['config_ref']:
        ini_dir = os.path.dirname(info['ini_path'])
        for base in (ini_dir, db_dir, os.getcwd()):
            candidate = os.path.normpath(os.path.join(base, info['config_ref']))
            if os.path.isfile(candidate):
                info['config_file'] = candidate
                break
    info['fixed_parameters'] = ini_fixed_parameters(info['ini_path'], info['section'])
    info['status'], info['differences'] = verify_config(db_file, info['config_file'], info['xml_hash'],
                                                        info['fixed_parameters'])
    return info


def find_reference_config(db_file: str) -> Union[str, None]:
    """Best-effort path to the model's PhysiCell settings XML recorded in a database
    (``model_config_info(db_file)['config_file']``)."""
    return model_config_info(db_file)['config_file']
