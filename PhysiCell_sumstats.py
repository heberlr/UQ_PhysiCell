import numpy as np
import anndata as ad
import pandas as pd
import pcdl

'''
A library of summary statistics to be applied to PhysiCell simulation output files.

Each method takes arguments
1) outputPath: a PhysiCell output directory and 
2) configFile_name: a PhysiCell config file name (contained in the outputPath directory) 

Each method will return the appropriate vector-valued summary.
'''
def final_population(outputPath, configFile_name):
    mcds = pcdl.pyMCDS(
        xmlfile = "final.xml",
        settingxml = configFile_name,
        output_path = outputPath
    )
    return len(mcds.data['discrete_cells']['data']['ID'])
