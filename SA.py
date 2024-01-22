import numpy as np
from SALib.sample import sobol

def generate_Sobol_SA_Samples(fileSamples=None):
    default_values = {'viral_replication_rate': 0.15, 'min_virion_count': 1.0, 'burst_virion_count': 100, 'virion_threshold_for_interferon': 10, 'min_virion_detection_threshold': 1}
    problem = {
        'num_vars': len(default_values),
        'names': default_values.keys(),
        'bounds': np.array([[0,0,0,0,0],[0.3,2.0,200.0,20.0,2.0]]).T # 100% percent of variation from default values
    }
    # Sample the parameters
    param_values = sobol.sample(problem, 128)
    # Save in .npy file
    if(fileSamples): np.save(fileSamples,param_values)
    print(param_values.shape)


if __name__ == '__main__':
    # generate_Sobol_SA_Samples("Samples_00.npy")
    np.save("Samples_01.npy", np.array([[1,2,3,4,5],[6,7,8,9,10]]))