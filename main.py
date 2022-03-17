import argparse
import os
import numpy as np

from computational_methods import IntegralMethod
from covariances import Brownian
from karhunen import KarhunenLoeve

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', required=True,
                        help='Folder where file is stored.')
    return parser.parse_args()

def main(args):

    # Stochastic process configuration ------------------------------------------------------------
    cov_funct = Brownian()
    random_distributions = "Gaussian"
    support = [0, 1]
    file_prefix = "brownian"
    # ---------------------------------------------------------------------------------------------
    # Hyperparameters -----------------------------------------------------------------------------
    n_outcomes = 40
    eigen_method = IntegralMethod
    eigen_method_config = {'integral': {'discret_size': 1000, 'scheme': "unif", 'tol': 1e-12}}
    eigen = eigen_method.from_config(eigen_method_config['integral'], support)
    # ---------------------------------------------------------------------------------------------

    # Apply integral method to get eigen from covariance function
    output = eigen.compute_eigen(cov_funct)

    # Compute Karhunen-Loeve expansions
    karlov = KarhunenLoeve(output['time_line'], output['eigen_values'], output['eigen_functs'])

    # Create output folder if it doesnt exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    # Get covariance function from Mercer expansion and store data
    np.savetxt(os.path.join(args.output_folder, file_prefix + '_cov_data.txt'), karlov.get_covariance())

    # Get Stochastic process realization from Karhuen-Loeve expansion and store data
    np.savetxt(os.path.join(args.output_folder, file_prefix + '_one_kar_data.txt'),
               karlov.get_process_sample(random_distributions))

    # Get Gaussian Process realizations
    kar_data = karlov.get_process_samples(n_outcomes)

    # Get filename
    kar_filename = os.path.join(args.output_folder, file_prefix + '_kar_data.txt')
    kar_file = open(kar_filename, 'a')

    # Save realizations to plot
    for outcome_indx in range(n_outcomes):

        np.savetxt(kar_file, kar_data[outcome_indx])
        kar_file.write("\n\n")

    kar_file.close()

if __name__ == '__main__':
    ARGS = args()
    main(ARGS)