import numpy as np
import anndata as ad
import pandas as pd
import pcdl
import os
import shutil
import xml.etree.ElementTree as ET

'''
A library of summary statistics to be applied to PhysiCell simulation output files.

Each method takes arguments
1) outputPath: a PhysiCell output directory and 
2) configFile_name: a PhysiCell config file name (contained in the outputPath directory) 

Each method will return the appropriate vector-valued summary.
'''


# Live cell count at the specified simulation time (in minutes)


def live_pop(outputPath, configFilePath, time='final', clear_output=True):
    pop_dic = {'population': []}
    xml_file = 'final.xml'
    if (time != 'final'):
        dt = float(ET.parse(configFilePath).getroot().findall(
            path=".//save/full_data/interval")[0].text
                   )  # save interval
        T_max = float(ET.parse(configFilePath).getroot().findall(
            path=".//overall/max_time")[0].text
                      )  # max simulation time
        N_output = int(np.floor(time / dt))  # output index
        if (time <= T_max):
            xml_file = "output%08d" % N_output + ".xml"  # intermediate PhysiCell timestep, e.g. output00000001
        else:
            print(
                f"WARNING: invalid simulation time {time} in summary statistic function with max simulation time {T_max}.")
    mcds = pcdl.TimeStep(
        xmlfile=xml_file,
        output_path=f"./{outputPath}".strip('/'),
        settingxml=None,
        microenv=False,
        graph=False,
        verbose=False
    )
    if clear_output:
        shutil.rmtree(outputPath)  # remove output folder
        os.remove(configFilePath)  # remove config file XML
    pop_dic['population'].append(len(mcds.get_cell_df()))
    return pop_dic.values()


def cycle_fraction(outputPath, configFilePath, time='final', clear_output=True):
    phase_ratio_dic = {'G0G1_phase': [], 'S_phase': [],'G2_phase': [],'M_phase': [], 'apoptotic': []}
    xml_file = 'final.xml'
    if (time != 'final'):
        dt = float(ET.parse(configFilePath).getroot().findall(
            path=".//save/full_data/interval")[0].text
                   )  # save interval
        T_max = float(ET.parse(configFilePath).getroot().findall(
            path=".//overall/max_time")[0].text
                      )  # max simulation time
        N_output = int(np.floor(time / dt))  # output index
        if (time <= T_max):
            xml_file = "output%08d" % N_output + ".xml"  # intermediate PhysiCell timestep, e.g. output00000001
        else:
            print(
                f"WARNING: invalid simulation time {time} in summary statistic function with max simulation time {T_max}.")
    mcds = pcdl.TimeStep(
        xmlfile=xml_file,
        output_path=f"./{outputPath}".strip('/'),
        settingxml=None,
        microenv=False,
        graph=False,
        verbose=False
    )
    if clear_output:
        shutil.rmtree(outputPath)  # remove output folder
        os.remove(configFilePath)  # remove config file XML
    cell_df = mcds.get_cell_df()
    pop = len(cell_df)
    phase_ratio_dic['G0G1_phase'].append(len(cell_df[cell_df['current_phase']=='G0G1_phase'])/pop)
    phase_ratio_dic['S_phase'].append(len(cell_df[cell_df['current_phase']=='S_phase'])/pop)
    phase_ratio_dic['M_phase'].append(len(cell_df[cell_df['current_phase']=='M_phase'])/pop)
    phase_ratio_dic['G2_phase'].append(len(cell_df[cell_df['current_phase']=='G2_phase'])/pop)
    phase_ratio_dic['apoptotic'].append(len(cell_df[cell_df['current_phase']=='apoptotic'])/pop)
    return phase_ratio_dic.values()

def live_mean_to_sd_ratio(outputPath, configFile_name):
    return


def Hill_half_max(outputPath, configFile_name):
    return
