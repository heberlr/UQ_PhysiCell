import pcdl
import numpy as np
import scipy
import pandas as pd
from shutil import rmtree
from typing import Union
import re
import inspect, ast
import textwrap
    
def summ_func_FinalPopLiveDead(outputPath:str,summaryFile:Union[str,None], dic_params:dict, SampleID:int, ReplicateID:int) -> Union[pd.DataFrame,None]:
    """
    Final population of live and dead cells
    
    Args:
        outputPath: str -> Path to the PhysiCell output directory.
        summaryFile: Union[str, None] -> File to store the summary (optional).
        dic_params: dict -> Dictionary of simulation parameters.
        SampleID: int -> Unique identifier for the sample.
        ReplicateID: int -> Unique identifier for the replicate.
    
    Returns:
        pd.DataFrame or None -> DataFrame with the computed QoIs or None if saved to a file.
    """
    # read the last file
    mcds = pcdl.TimeStep('final.xml',outputPath, microenv=False, graph=False, settingxml=None, verbose=False)
    # dataframe of cells
    df_cell = mcds.get_cell_df() 
    # population stats live and dead cells
    live_cells = len(df_cell[ (df_cell['dead'] == False) ] )
    dead_cells = len(df_cell[ (df_cell['dead'] == True) ] )
    # dataframe structure
    data = {'time': mcds.get_time(), 'sampleID': SampleID, 'replicateID': ReplicateID, 'live_cells': live_cells, 'dead_cells': dead_cells, 'run_time_sec': mcds.get_runtime()}
    data_conc = {**data,**dic_params} # concatenate output data and parameters
    df = pd.DataFrame([data_conc])
    # remove replicate output folder
    rmtree( outputPath )
    if (summaryFile): 
        df.to_csv(summaryFile, sep='\t', encoding='utf-8')
        return None
    else: return df

# Population over time of live and dead cells
def summ_func_TimeSeriesPopLiveDead(outputPath:str,summaryFile:Union[str,None], dic_params:dict, SampleID:int, ReplicateID:int) -> Union[pd.DataFrame,None]:
    """
    Population over time of live and dead cells

    Args:
        outputPath: str -> Path to the PhysiCell output directory.
        summaryFile: Union[str, None] -> File to store the summary (optional).
        dic_params: dict -> Dictionary of simulation parameters.
        SampleID: int -> Unique identifier for the sample.
        ReplicateID: int -> Unique identifier for the replicate.
    
    Returns:
        pd.DataFrame or None -> DataFrame with the computed QoIs or None if saved to a file.
    """

    mcds_ts = pcdl.TimeSeries(outputPath, microenv=False, graph=False, settingxml=None, verbose=False)
    for mcds in mcds_ts.get_mcds_list():
        df_cell = mcds.get_cell_df() 
        live_cells = len(df_cell[ (df_cell['dead'] == False) ] )
        dead_cells = len(df_cell[ (df_cell['dead'] == True) ] )
        data = {'time': mcds.get_time(), 'sampleID': SampleID, 'replicateID': ReplicateID, 'live_cells': live_cells, 'dead_cells': dead_cells, 'run_time_sec': mcds.get_runtime()}
        data_conc = {**data,**dic_params} # concatenate output data and parameters
        if ( mcds.get_time() == 0 ): df = pd.DataFrame([data_conc]) # create the dataframe
        else: df.loc[len(df)] = data_conc # append the dictionary to the dataframe
    # remove replicate output folder
    rmtree( outputPath )
    if (summaryFile): 
        df.to_csv(summaryFile, sep='\t', encoding='utf-8')
        return None
    else: return df

def _check_functions_need_microenv(qoi_funcs:dict) -> bool:
    """
    Check if any of the QoI functions require microenvironment data (df_subs).
    
    Args:
        qoi_funcs: dict -> Dictionary of QoI functions
    
    Returns:
        bool: True if any function needs microenvironment data
    """
    # If no functions provided, return True (default to loading microenvironment)
    if not qoi_funcs:
        return True
    
    # If any function needs microenvironment data, return True
    for func in qoi_funcs.values():
        # Check if any parameter name indicates substrate/concentration data
        param_lower = func.__param_name__.lower()
        if 'subs' in param_lower or 'conc' in param_lower or 'micro' in param_lower:
                return True
    # If none need it,
    return False

def safe_call_qoi_function(func: callable, mcds:Union[pcdl.TimeStep,None]=None, list_mcds:Union[list,None]=None ):
    """
    Safely call a QoI function with the appropriate dataframe based on parameter inspection.
    
    Args:
        func: The QoI function to call
        mcds: pcdl.TimeStep or None -> The mcds object for single snapshot
        list_mcds: list of pcdl.TimeStep or None -> The mcds time series object for multiple snapshots
    
    Returns:
        Result of the QoI function
    """
    # Require stored metadata for dispatch.
    param_name = getattr(func, '__param_name__', None)

    # Reject non-wrapped callables (no __param_name__ attribute).
    if param_name is None:
        raise ValueError(
            "QoI function is missing required '__param_name__' metadata. "
            "Wrap callables with _create_wrapper_for_qoi_function or recreate_qoi_functions first."
        )
    
    # Handle wrapped/string-defined QoI functions.
    if param_name in ['df_cell', 'df'] and mcds is not None: # Function expects cell dataframe
        return func(mcds.get_cell_df())
    elif param_name in ['df_subs'] and mcds is not None: # Function expects substrate dataframe
        return func(mcds.get_conc_df())
    elif param_name in ['mcds'] and mcds is not None: # Function expects the mcds object
        return func(mcds)
    elif param_name in ['mcds_ts'] and list_mcds is not None: # Function expects the mcds time series object
        if mcds == list_mcds[-1]: # Ensure we only compute once per time series (last mcds passed)
            return func(list_mcds)
        else:
            return None # Skip computation for other snapshots

    raise ValueError(
        f"Could not call QoI function with param_name='{param_name}', "
        f"mcds provided={mcds is not None}, list_mcds provided={list_mcds is not None}."
    )

def _create_wrapper_for_qoi_function(func: callable, param_name: str, qoi_name: str) -> callable:
    """
    Create a wrapper function for a QoI function that preserves parameter inspection capability.
    
    Args:
        func: The original QoI function to wrap
        param_name: The name of the parameter that the function expects (e.g., 'df_cell', 'df_subs', 'mcds', 'mcds_ts')
        qoi_name: The name of the QoI function (used for setting the wrapper's __name__)
    """
    # Create a wrapper function that calls the pre-compiled lambda
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    # Store the original parameter name as an attribute for inspection
    wrapper.__param_name__ = param_name
    wrapper.__name__ = qoi_name
    
    return wrapper

def _extract_lambda_source(func):
    source = textwrap.dedent(inspect.getsource(func))
    lambda_index = source.find("lambda")
    if lambda_index == -1:
        raise ValueError("No lambda expression found")

    candidate = source[lambda_index:].strip()

    # Try progressively shorter suffixes until the lambda expression parses.
    # This handles trailing commas and surrounding syntax from dict literals,
    # calls, or other enclosing expressions.
    for end in range(len(candidate), 0, -1):
        snippet = candidate[:end].rstrip()
        if not snippet:
            continue

        try:
            tree = ast.parse(f"({snippet})", mode="eval")
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Lambda):
                extracted = ast.get_source_segment(f"({snippet})", node)
                if extracted:
                    return extracted

    raise ValueError("No lambda expression found")

def _convert_qoi_function_to_string(qoi_func: callable, qoi_name:str):
    """
    Convert a callable QoI function to its source code string representation.
    Args:
        qoi_func: The QoI function to convert (must be a callable)
        qoi_name: The name of the QoI function (used for error messages)
    """
    # Check if the qoi_func is a callable function converted to string
    if callable(qoi_func):
        return _extract_lambda_source(qoi_func)
    else:
        raise ValueError(f"QoI function for '{qoi_name}' is not callable.")
        
def _create_named_function_from_string(qoi_name: str, func_str: str, qoi_def:dict={}) -> callable:
    """
    Dynamically creates a named function from a string.
    
    Args:
        qoi_name: The name of the function to be created.
        func_str: The string representation of the function.
        qoi_def (dict): first-class object, that can be used in qoi_functions
            lambda string, mapped to their name.
            e.g. for a function definition, if the function definition is:
            def my_func():
                print('hello world!')
                return 0
            then the qoi_def dict would look like this:
            {'my_func': my_func}

    Returns:
        The created function with preserved parameter inspection capability.
    """
    # Extract arg name from lambda (e.g., "lambda df_subs:" -> "df_subs")
    arg_match = re.search(r'lambda\s+(\w+):', func_str)
    if arg_match:
        arg_name = arg_match.group(1)
    else:
        arg_name = 'mcds'
    
    # Create a restricted namespace with necessary imports and no built-in functions.
    namespace = {'pd': pd, 'np': np, 'len': len, 'sum': sum, 'map': map, '__builtins__': {}}
    # Add locally defined QoI helper callables so lambda strings can reference them.
    namespace.update(qoi_def)
    namespace.update({
        name: obj
        for name, obj in globals().items()
        if callable(obj) and name.startswith('qoi_func')
    })

    # Evaluate the lambda function once at creation time
    try:
        compiled_func = eval(func_str, namespace)
    except Exception as e:
        raise ValueError(f"Error evaluating QoI function string '{func_str}': {e}")
    
    # Create a wrapper function that calls the pre-compiled lambda
    wrapper = _create_wrapper_for_qoi_function(func=compiled_func, param_name=arg_name, qoi_name=qoi_name)
    
    return wrapper

def recreate_qoi_functions(qoi_functions:dict, qoi_def:dict={}) -> dict:
    """
    Recreate QoI functions from their string representations.
    
    Args:
        qoi_functions: Dictionary of QoI functions (keys as names, values as strings).
            Pass a mapping like the following:
                qoi_functions = {
                    "mean_substrate": "lambda df_subs: df_subs['substrate'].mean()",
                    "live_cell_count": "lambda df_cell: len(df_cell[df_cell['dead'] == False])",
                    "max_volume": "lambda df: df['total_volume'].max()",
                    "mean_radial_distance": "lambda df: df[['position_x', 'position_y', 'position_z']].apply(lambda row: ((row['position_x']**2 + row['position_y']**2 + row['position_z']**2)**0.5), axis=1).mean()"
                }
        qoi_def (dict): first-class object, that can be used in qoi_functions
            lambda string, mapped to their name.
            e.g. for a function definition, if the function definition is:
                def my_func():
                    print('hello world!')
                    return 0
            then the qoi_def dict would look like this:
                {'my_func': my_func}

    Returns:
        Dictionary of recreated QoI functions (keys as names, values as callables)
    """
    recreated_qoi_funcs = {}
    for qoi_name, func_str in qoi_functions.items():
        try:
            if callable(func_str):
                signature = inspect.signature(func_str)
                param_name = list(dict(signature.parameters).keys())[0] 
                recreated_qoi_funcs[qoi_name] = _create_wrapper_for_qoi_function(func=func_str, param_name=param_name, qoi_name=qoi_name)
            else:
                recreated_qoi_funcs[qoi_name] = _create_named_function_from_string(qoi_name=qoi_name, func_str=func_str, qoi_def=qoi_def)
        except Exception as e:
            raise ValueError(f"Error recreating QoI function '{qoi_name}': {e}")
    return recreated_qoi_funcs
    
def summary_function(outputPath:str, summaryFile:Union[str, None], dic_params:dict, SampleID:int, ReplicateID:int, qoi_functions:dict, RemoveFolder:bool=True, drop_columns:Union[list, None]=None,) -> Union[pd.DataFrame, None]:
    """
    Generic summary function for creating custom QoIs (Quantities of Interest) based on df_cell elements.

    Args:
        outputPath: str -> Path to the PhysiCell output directory.
        summaryFile: Union[str, None] -> File to store the summary (optional).
        dic_params: dict -> Dictionary of simulation parameters.
        SampleID: int -> Unique identifier for the sample.
        ReplicateID: int -> Unique identifier for the replicate.
        qoi_functions: dict -> Dictionary of QoI functions with keys as QoI names and values as functions/lambdas.
        RemoveFolder: bool -> Whether to remove the output folder after processing.
        drop_columns: list -> List of columns to drop from the DataFrame.

    Returns:
        pd.DataFrame or None -> DataFrame with the computed QoIs or None if saved to a file.
    """
    try:
        # Check if any function needs microenvironment data
        needs_microenv = _check_functions_need_microenv(qoi_functions)
        # Load the time series
        mcds_ts = pcdl.TimeSeries(outputPath, microenv=needs_microenv, graph=False, settingxml=None, verbose=False)
        list_mcds=mcds_ts.get_mcds_list()
        #  All data is stored as a list of mcds
        if qoi_functions is None:
            # Optional: Remove replicate output folder
            if (RemoveFolder): rmtree(outputPath)
            # Entire list of mcds is returned if drop_columns is empty
            if not drop_columns:
                return list_mcds
            else:
                df_list = []
                for mcds in list_mcds:
                    df_cell = mcds.get_cell_df()
                    df_cell.drop(columns=drop_columns, inplace=True, errors='ignore')
                    df_list.append(df_cell)
                return df_list
        else:
            df_list = []
            for mcds in list_mcds:
                try:
                    qoi_data = {}
                    for name, func in qoi_functions.items():
                        func_result = safe_call_qoi_function(func, mcds=mcds, list_mcds=list_mcds)
                        if func_result is not None:
                            qoi_data[name] = func_result
                    if not qoi_data: continue  # Skip if no QoI data was computed
                    # print(f"Computed QoIs at time {mcds.get_time()}: {qoi_data}")
                except Exception as e:
                    raise RuntimeError(f"Error computing QoIs in summary_function: {e}")
                
                data = {
                    'time': mcds.get_time(),
                    'sampleID': SampleID,
                    'replicateID': ReplicateID,
                    **qoi_data
                }
                data_conc = {**data, **dic_params}
                df_list.append(data_conc)
            df = pd.DataFrame(df_list)
    except FileNotFoundError as e:
        raise RuntimeError(f"Error: Required file not found in {outputPath}. {e}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while processing QoIs: {e}")

    # Optional: Remove replicate output folder
    if (RemoveFolder): rmtree(outputPath)

    # Save to file or return DataFrame
    if summaryFile:
        df.to_csv(summaryFile, sep='\t', encoding='utf-8')
        return None
    else:
        return df

def qoi_func_radial_density_summary(df, center=[0,0,0]):
    """
    Extract summary statistics from radial distribution of cells
    Args:
        df: DataFrame with columns ['position_x', 'position_y', 'position_z']
        center: List or array-like of length 3 representing the center point for radial distance calculation (default is [0,0,0])
    """
    cell_centers = df[['position_x', 'position_y', 'position_z']].values
    dist_from_center = np.linalg.norm(cell_centers - np.array(center), axis=1)

    return {
        "center_of_mass": np.average(dist_from_center),
        "median": np.median(dist_from_center),
        "spread": np.std(dist_from_center),
        "iqr": np.percentile(dist_from_center, 75) - np.percentile(dist_from_center, 25),
        "skewness": scipy.stats.skew(dist_from_center),
        "kurtosis": scipy.stats.kurtosis(dist_from_center),
    }

def qoi_func_persistent_homology(df:pd.DataFrame, Plot=False) -> tuple:
    """
    Compute persistent homology vectorization using muspan.
    (source: https://docs.muspan.co.uk/latest/_collections/topology/Topology%203%20-%20persistence%20vectorisation.html)
    
    Args:
        df_cells: DataFrame -> DataFrame containing cell data with 'position_x', 'position_y', and 'cell_type' columns.
        Plot: bool -> Whether to plot the persistence diagram (optional).
    
    Returns:
        tuple -> (pd.Series with vectorized persistent homology features, figure or None)
    """
    import matplotlib.pyplot as plt
    try:
        import muspan
    except ImportError:
        raise ImportError("muspan library is required for computing persistent homology. Please install it via 'pip install muspan'.")

    # Extract cell positions
    points = df[['position_x', 'position_y']].to_numpy()
    labels = df['cell_type'].to_numpy()
    
    # Create a muspan domain and add points
    domain = muspan.domain('Position Data')
    domain.add_points(points, 'Cell Positions')
    domain.add_labels('Celltype', labels)

    # Query to select cells of types 'A', 'B', ... in each domain
    q_cell_types = muspan.query.query(domain, ('label', 'Celltype'), 'in', df['cell_type'].unique().tolist())

    # Compute Vietoris-Rips filtrations
    feature_persistence = muspan.topology.vietoris_rips_filtration(domain, population=q_cell_types, max_dimension=1)

    # Plot persistence diagram (optional)
    figure = None
    if Plot:
        fig, axes = plt.subplots(figsize=(15, 6), nrows=1, ncols=2)
        # Plot domain with cell types
        muspan.visualise.visualise(domain, 'Celltype', ax=axes[0], add_cbar=False, marker_size=2.5, objects_to_plot=q_cell_types)
        # Plot persistence diagram
        muspan.visualise.persistence_diagram(feature_persistence, ax=axes[1])
        figure = [fig, axes]
        

    # Vectorise the persistence homology diagram for the domain using statistical method
    vectorised_ph,name_of_features = muspan.topology.vectorise_persistence(feature_persistence, method='statistics')
    return pd.Series(vectorised_ph, index=name_of_features), figure

def qoi_func_relational_ph( df: pd.DataFrame, landmark_type: str, witness_type: str, max_dim: int = 1, mode: str = "distance", ax = None) -> tuple:
    """
    Relational Persistent Homology using Dowker/Witness idea.
    A→B (A as vertices, B as witnesses) describes how B are arranged around A geometry.

    Args:
        df : DataFrame with columns ['position_x','position_y','cell_type']
        landmark_type : cell type to use as landmarks (A)
        witness_type : cell type to use as witnesses (B)
        max_dim : PH dimension (0 or 1)
        mode: 'distance' or 'count'
        ax : matplotlib axis for plotting  (optional)

    Returns:
        tuple -> (pd.Series with vectorized persistent homology features (summary statistics), raw persistence diagram (list of (dimension, (birth, death)) pairs)
    """
    from scipy.spatial.distance import cdist
    from scipy.spatial import Delaunay
    import gudhi, muspan
    import matplotlib.pyplot as plt

    # Extract A = Landmarks, B = Witnesses
    A_points = df[df["cell_type"] == landmark_type][["position_x","position_y"]].to_numpy()
    B_points = df[df["cell_type"] == witness_type][["position_x","position_y"]].to_numpy()
    if len(A_points) == 0 or len(B_points) == 0:
        print("Warning: Both landmark_type and witness_type must be present.")
        # Return empty Series without pre-defined index - will be filled with NaN by the caller
        return pd.Series(dtype=float), None
    # Pairwise distances B×A
    D = cdist(B_points, A_points)

    # Build candidate simplex list from landmarks
    if len(A_points) < 3:
        print("Warning: At least 3 landmark points are required for relational PH.")
        # Return empty Series - will be filled with NaN by the caller
        return pd.Series(dtype=float), None
    # Use Delaunay to match Python repo behavior
    tri = Delaunay(A_points)
    # vertices = all points
    vertices = list(range(len(A_points)))
    # edges from Delaunay
    edges = set()
    faces = set()
    for simplex in tri.simplices:
        simplex = list(simplex)
        # all edges
        for i in range(3):
            for j in range(i+1, 3):
                edges.add(tuple(sorted([simplex[i], simplex[j]])))
        # triangle
        faces.add(tuple(sorted(simplex)))
    edges = list(edges)
    faces = list(faces)

    # Compute filtration values
    def dowker_distance(simplex):
        """Distance-based Dowker: min_w max_a d(a,w)."""
        return np.min(np.max(D[:, simplex], axis=1))
    
    def dowker_count(simplex):
        """Count-based Dowker: number of witnesses covering all simplex vertices."""
        return -np.sum(np.all(D[:, simplex] <= np.max(D[:, simplex], axis=0), axis=1))
    
    # Compute filtration values based on the distance
    fil_func = dowker_distance if mode == "distance" else dowker_count
    f_vertex = np.array([fil_func([i]) for i in vertices])
    f_edge   = np.array([fil_func(list(e)) for e in edges])
    if max_dim >= 2:
        f_face   = np.array([fil_func(list(f)) for f in faces])

    # Build GUDHI SimplexTree
    st = gudhi.SimplexTree()
    # vertices
    for i, fv in enumerate(f_vertex):
        st.insert([i], filtration=float(fv))
    # edges
    for (e, fv) in zip(edges, f_edge):
        st.insert(list(e), filtration=float(fv))
    # faces
    if max_dim >= 2:
        for (f, fv) in zip(faces, f_face):
            st.insert(list(f), filtration=float(fv))
    st.initialize_filtration()

    # Compute persistence
    diag = st.persistence()

    # Convert GUDHI persistence output to muspan-style dict {'dgms': [array_dim0, array_dim1, ...]}
    dgms = []
    for d in range(max_dim + 1):
        intervals = [pair for dim, pair in diag if dim == d]
        if len(intervals) == 0:
            dgms.append(np.zeros((0, 2)))
        else:
            dgms.append(np.array(intervals))
    feature_persistence = {'dgms': dgms}

    # Vectorize diagram using muspan's statistics. Some degenerate diagrams can
    # produce empty finite arrays in muspan/numpy percentile calls.
    try:
        vec, names = muspan.topology.vectorise_persistence(feature_persistence, method="statistics")
        vec = pd.Series(vec, index=names)
    except (IndexError, ValueError) as e:
        print(
            f"Warning: Could not vectorize relational PH for "
            f"{landmark_type}->{witness_type}: {e}"
        )
        # Return empty Series - will be filled with NaN by the caller
        vec = pd.Series(dtype=float)

    # Plot
    if ax is not None:
        axes = gudhi.plot_persistence_diagram(diag, axes=ax)

    return vec, diag
