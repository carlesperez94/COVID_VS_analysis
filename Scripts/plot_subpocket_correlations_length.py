# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
import os
import sys
from typing import Tuple, List, Optional, Union

# External imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Local imports
SCRIPT_PATH = os.path.dirname(__file__)
sys.path.append(os.path.abspath(SCRIPT_PATH + '/..'))
from PELEIterator import SimIt
from Helpers.Subpockets import read_subpocket_dataframe


# Script information
__author__ = "Marti Municoy"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy"
__email__ = "marti.municoy@bsc.es"


def parse_args() -> Tuple[List[str], str, str, float, str,
                          Optional[List[int]], Optional[List[int]]]:
    parser = ap.ArgumentParser()

    parser.add_argument("traj_paths", metavar="PATH", type=str,
                        nargs='*',
                        help="Path to PELE trajectory files")

    parser.add_argument("--subpockets", metavar="STR", type=str,
                        help="Subpockets csv file name",
                        default='subpockets.csv')

    parser.add_argument("--ic50", metavar="STR", type=str,
                        help="IC50 csv file name",
                        default='ic50.csv')

    parser.add_argument("-p", "--percentile", metavar="FLOAT", type=float,
                        help="Percentile applied to the set of "
                        + "intersections that are obtained for each subpocket",
                        default=95)

    parser.add_argument('-t', '--length_type', default='maximum_steps',
                        nargs='?', type=str,
                        choices=['maximum_steps', 'maximum_trajectories'],
                        help='Type of simulation length to evaluate: '
                        + 'maximum_steps or maximum_trajectories. '
                        + 'Default: maximum_steps')

    parser.add_argument('--maximum_trajectories', metavar="INT", type=int,
                        help="Maximum number of PELE trajectories to use in "
                        + "the plot", default=None, nargs='*')

    parser.add_argument('--maximum_steps', metavar="INT", type=int,
                        help="Maximum number of PELE steps to use in the plot",
                        default=None, nargs='*')

    parser.add_argument('-o', '--output', metavar='STR', type=str,
                        default='subpocket_correlations_length.png',
                        help='Output name for the resulting plot')

    args = parser.parse_args()

    return args.traj_paths, args.subpockets, args.ic50, args.percentile, \
        args.length_type, args.maximum_trajectories, args.maximum_steps, \
        args.output


def get_columns(all_sim_it: SimIt, csv_file_name: str) -> list:
    columns = []
    for PELE_sim_path in all_sim_it:
        if (not PELE_sim_path.joinpath(csv_file_name).is_file()):
            print(' - Skipping simulation because subpockets csv file '
                  + 'was missing')
            continue

        data = pd.read_csv(PELE_sim_path.joinpath(csv_file_name))
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

        for col in data.columns:
            if ('_intersection' in col and 'nonpolar' not in col):
                if (col not in columns):
                    columns.append(col)

    return columns


def extract_subpocket_results(all_sim_it: SimIt, csv_file_name: str,
                              percentile: float, columns: list,
                              length_type: str,
                              length_iterator: List[Union[float, int]]
                              ) -> pd.DataFrame:
    subpocket_results = pd.DataFrame()

    if length_type == 'maximum_steps':
        key = 'step'
        offset = 0
    elif length_type == 'maximum_trajectories':
        key = 'trajectory'
        offset = 1
    else:
        raise ValueError('Invalid length type value: {}'.format(length_type))

    for PELE_sim_path in all_sim_it:
        print('')
        print(' - Reading data from {}'.format(PELE_sim_path))

        if (not PELE_sim_path.joinpath(csv_file_name).is_file()):
            print(' - Skipping simulation because intersection csv file '
                  + 'is missing')
            continue

        print('   - Retrieving subpocket intersections')
        data = pd.read_csv(PELE_sim_path.joinpath(csv_file_name))

        for length_try in length_iterator:
            # Filtering data frame
            f_data = data[data[key] < length_try + offset]

            # Retrieving subpocket results
            metrics = [PELE_sim_path.name, length_try]
            for col in columns:
                values = f_data[col].values
                metrics.append(np.percentile(values, percentile))

            subpocket_results = pd.concat([subpocket_results,
                                           pd.DataFrame([metrics],
                                                        columns=['path',
                                                                 'length']
                                                        + columns)])

    return subpocket_results


def add_ic50s(subpocket_results: pd.DataFrame, ic50_csv: str
              ) -> pd.DataFrame:
    print(' - Retrieving IC50 values')

    ic50 = pd.read_csv(ic50_csv)
    subpocket_results = subpocket_results.merge(
        ic50, left_on='path', right_on='path')
    subpocket_results['pIC50'] = - np.log10(
        subpocket_results.loc[:, 'IC50'] / 1000000)

    return subpocket_results


def make_plot(all_sim_it: SimIt, csv_file_name: str, ic50_csv: str,
              columns: list, percentile: float, length_type: str,
              max_trajs: Optional[List[int]],
              max_steps: Optional[List[int]],
              output_name: str):
    fig, ax = plt.subplots()
    fig.suptitle('Subpocket-pIC50 correlations with {}'.format(percentile)
                 + '-percentile vs simulation length')

    ax.set_axisbelow(True)
    ax.grid(True, color='white')
    ax.set_facecolor('lightgray')

    if length_type == 'maximum_steps':
        length_iterator = max_steps
    elif length_type == 'maximum_trajectories':
        length_iterator = max_trajs
    else:
        raise ValueError('Invalid length type value: {}'.format(length_type))

    subpocket_results = extract_subpocket_results(all_sim_it,
                                                  csv_file_name,
                                                  percentile, columns,
                                                  length_type,
                                                  length_iterator)

    subpocket_results = add_ic50s(subpocket_results, ic50_csv)

    results = []
    for length_try in length_iterator:
        f_subpocket_results = \
            subpocket_results[subpocket_results['length'] == length_try]

        Y = f_subpocket_results.loc[:, columns].to_numpy()
        x = f_subpocket_results['pIC50'].to_numpy()

        current_results = ()
        for y in Y.transpose():
            lin_reg = LinearRegression().fit(y.reshape(-1, 1),
                                             x.reshape(-1, 1))
            x_pred = lin_reg.predict(y.reshape(-1, 1))

            current_results += (r2_score(x, x_pred), )

        results.append(current_results)

    for result, col in zip(zip(*results), columns):
        ax.plot(length_iterator, result, 'x-',
                label=col.strip('_intersection'))

    ax.set_xlabel('Simulation length ({})'.format(
        length_type.replace('_', ' ')))
    ax.set_ylabel('r2 score')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
              ncol=len(columns), fancybox=True, markerscale=2)

    plt.tight_layout(rect=(0, 0, 0.95, 0.94))
    plt.savefig(output_name)
    plt.close()


def main():
    # Parse args
    PELE_sim_paths, csv_file_name, ic50_csv, percentile, length_type, \
        max_trajs, max_steps, output_name = parse_args()

    all_sim_it = SimIt(PELE_sim_paths)

    print(' - Simulations that will be analyzed:')
    for sim_path in all_sim_it:
        print('   - {}'.format(sim_path.name))

    # TODO get rid of this compatibility issue
    columns, _, _ = read_subpocket_dataframe(all_sim_it, csv_file_name,
                                             'intersection')

    print(' - Subpockets found:')
    for col in columns:
        print('   - {}'.format(col.strip('_intersection')))

    if (len(columns) == 0):
        raise ValueError('No subpocket intersections were found in the '
                         + 'simulation paths that were supplied')

    make_plot(all_sim_it, csv_file_name, ic50_csv, columns, percentile,
              length_type, max_trajs, max_steps, output_name)


if __name__ == "__main__":
    main()
