# Import modules
import argparse
import numpy as np

# Import our classes and functions
from eigen_numeric import HaarMethod
from covariances import Brownian
from karhunen import KarhunenLoeve
from save_output import DataStorage


# Argparse
def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', required=True,
                        help='Folder where file is stored.')
    return parser.parse_args()

def main(args):
    """Compute discrete covariance function and Karhunen-Loeve expansion
        from analytic covariance function."""

    # Stochastic process configuration ------------------------------------------------------------
    cov_funct = Brownian()
    random_distributions = "Gaussian"
    support = [0, 1]
    file_prefix = cov_funct.get_prefix()
    # ---------------------------------------------------------------------------------------------
    # Hyperparameters -----------------------------------------------------------------------------
    n_outcomes = 1
    discret_size = 2**4
    eigen = HaarMethod()
    # ---------------------------------------------------------------------------------------------
    # Output Configuration ------------------------------------------------------------------------
    saver = DataStorage(args.output_folder)
    # ---------------------------------------------------------------------------------------------

    # Apply integral method to get eigen from covariance function
    discret_eigen = eigen.compute_eigen_from_kernel(cov_funct, discret_size, support)

    # Compute Karhunen-Loeve expansions
    karlov = KarhunenLoeve(discret_eigen['time_line'],
                           discret_eigen['eigen_values'],
                           discret_eigen['eigen_functs'])
    
    # Get covariance function from Mercer expansion and store data
    saver.save_numpy(file_prefix + 'cov_data', karlov.get_covariance())

    # Get stochastic process realization from Karhuen-Loeve expansion and store data
    saver.save_numpy(file_prefix + 'one_kar_data', karlov.get_process_sample(random_distributions))

    # Get process realizations and store data
    randoms_list = [np.random.normal(0, 1, len(discret_eigen['eigen_values']))
                    for _ in range(n_outcomes)]
    kar_data = karlov.get_process_samples(randoms_list)
    saver.save_numpies(file_prefix + 'kar_data', kar_data)

    # TODO: Create analytical solution for Brownian proces.
    # Get Gaussian Process realizations from true eigen and store data
    # True eigen values
    eigen_values_gt = np.array([1./(((k+1/2.)**2)*np.pi**2)
                                for k in range(len(discret_eigen['eigen_values']))])
    # True eigen functions
    eigen_functs_gt = np.array([np.sqrt(2)*np.sin(np.pi*t*(k+1./2))
                                for t in discret_eigen['time_line']
                                for k in range(len(discret_eigen['eigen_values']))]).reshape(discret_size, -1)
    # Get right sign
    nozero_sign = True
    row_indx = 0
    while nozero_sign and row_indx < discret_size:
        col_sign = np.sign(discret_eigen['eigen_functs'][row_indx])
        nozero_sign = 0 in col_sign
        row_indx += 1
    if row_indx < discret_size:
        eigen_functs_gt = eigen_functs_gt*col_sign
    # Karhunen-Loeve instance
    kar_gt = KarhunenLoeve(discret_eigen['time_line'], eigen_values_gt, eigen_functs_gt)
    # Get random samples
    kar_gt_data = kar_gt.get_process_samples(randoms_list)
    # Store data
    saver.save_numpies(file_prefix + 'kar_gt_data', kar_gt_data)


if __name__ == '__main__':
    ARGS = args()
    main(ARGS)

"""
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "IM: unif",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main.py",
            "console": "integratedTerminal",
            "args": [
                "--output_folder",
                "${workspaceFolder}/output",
                "--scheme",
                "integral_unif",
            ]
        },
        {
            "name": "IM: trapez",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main.py",
            "console": "integratedTerminal",
            "args": [
                "--output_folder",
                "${workspaceFolder}/output",
                "--scheme",
                "integral_trapez",
            ]
        },
        {
            "name": "Haar Method",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/mainhaar.py",
            "console": "integratedTerminal",
            "args": [
                "--output_folder",
                "${workspaceFolder}/output"
            ]
        }
    ]
}
"""