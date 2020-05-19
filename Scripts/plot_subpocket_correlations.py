# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
import os
import sys
from typing import Tuple, List, Optional

# External imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpl_patches
from sklearn.linear_model import LinearRegression
from sklearn import metrics as skmetrics

# Local imports
SCRIPT_PATH = os.path.dirname(__file__)
sys.path.append(os.path.abspath(SCRIPT_PATH + '/..'))
from Helpers import SimIt
from Helpers.Subpockets import read_subpocket_dataframe


# Script information
__author__ = "Marti Municoy"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy"
__email__ = "marti.municoy@bsc.es"


def parse_args() -> Tuple[List[str], str, str, float, float, Optional[int],
                          str, str]:
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
                        + "occupancies that are obtained for each subpocket",
                        default=95)

    parser.add_argument('--trajectories_fraction', metavar="FLOAT", type=float,
                        help="Fraction of trajectories to use in the plot",
                        default=1)

    parser.add_argument('--maximum_steps', metavar="INT", type=int,
                        help="Maximum number of PELE steps to use in the plot",
                        default=None)

    parser.add_argument('-m', '--metric', default='occupancy',
                        nargs='?', type=str,
                        choices=['occupancy', 'nonpolar_occupancy',
                                 'aromatic_occupancy', 'positive_charge',
                                 'negative_charge'],
                        help='Metric whose correlation will be analyzed')

    parser.add_argument('-o', '--output', metavar='STR', type=str,
                        default='subpocket_correlations.png',
                        help='Output name for the resulting plot')

    args = parser.parse_args()

    return args.traj_paths, args.subpockets, args.ic50, args.percentile, \
        args.trajectories_fraction, args.maximum_steps, args.metric, \
        args.output


def apply_filtering(data: pd.DataFrame, traj_fraction: float,
                    max_steps: Optional[int]) -> pd.DataFrame:
    if max_steps is not None:
        print('   - Filtering by maximum step threshold ({})'.format(
            max_steps))
        new_data = data[data['model'] < max_steps]
        print('     - {} entries were filtered to {}'.format(len(data),
                                                             len(new_data)))
        data = new_data

    if traj_fraction < 1:
        print('   - Filtering by trajectories fraction threshold ({})'.format(
            traj_fraction))
        all_trajs = list(set(data['trajectory'].to_numpy()))
        selected_trajs = np.random.choice(all_trajs,
                                          int(traj_fraction * len(all_trajs)),
                                          replace=False)
        new_data = data[data['trajectory'].isin(selected_trajs)]
        print('     - {} entries were filtered to {}'.format(len(data),
                                                             len(new_data)))
        data = new_data

    return data


def make_plots(all_sim_it: SimIt, columns: List[str], csv_file_name: str,
               percentile: float, traj_fraction: float, max_steps: int,
               ic50_csv: str, output: str, metric: str, pretty_metric: str,
               units: str):
    fig, axs = plt.subplots(len(columns), 1, figsize=(20, 15))
    fig.suptitle('Subpocket-LIG {}'.format(pretty_metric))

    for i, col in enumerate(columns):
        axs[i].set_title(col.strip('_' + metric))
        axs[i].set_ylabel('{}-LIG {} ({})'.format(
            pretty_metric, col.replace('_' + metric, ''), units))

    subpocket_results = pd.DataFrame()
    for PELE_sim_path in all_sim_it:
        print('')
        print(' - Reading data from {}'.format(PELE_sim_path))

        if (not PELE_sim_path.joinpath(csv_file_name).is_file()):
            print(' - Skipping simulation because '
                  + '{} csv file is missing'.format(pretty_metric))
            continue

        print('   - Retrieving subpocket {}'.format(pretty_metric))
        data = pd.read_csv(PELE_sim_path.joinpath(csv_file_name))

        metrics = [PELE_sim_path.name, ]
        for col in columns:
            values = data[col].values
            metrics.append(np.percentile(values, percentile))

        subpocket_results = pd.concat([subpocket_results,
                                       pd.DataFrame([metrics],
                                                    columns=['path', ]
                                                    + columns)])

        data = apply_filtering(data, traj_fraction, max_steps)

    print(' - Retrieving IC50 values')
    ic50 = pd.read_csv(ic50_csv)
    subpocket_results = subpocket_results.merge(
        ic50, left_on='path', right_on='path')
    subpocket_results['pIC50'] = - np.log10(
        subpocket_results.loc[:, 'IC50'] / 1000000)

    fig, axs = plt.subplots(int(len(columns) / 2) + len(columns) % 2, 2,
                            figsize=(15, 5 * int(len(columns) / 2
                                                 + len(columns) % 2)))
    fig.suptitle('Subpocket occupancy vs -pIC50')

    X_all = subpocket_results.loc[:, columns].values
    y_all = subpocket_results['pIC50'].values

    for i, col in enumerate(columns):
        ax = axs[int(i / 2)][i % 2]
        ax.set_title(col.replace('_' + metric, ''))
        ax.set_ylabel('{}-percentile of {} in {} ({})'.format(
            percentile, pretty_metric, col.replace('_' + metric, ''), units))
        ax.set_xlabel('-pIC50')

        x_array = np.array([X[i] for X in X_all])
        ax.plot(y_all, x_array, ls='', c='r', marker='x')

        ax.set_axisbelow(True)
        ax.grid(True, color='white')
        ax.set_facecolor('lightgray')

        lin_reg = LinearRegression()
        lin_reg.fit(y_all.reshape(-1, 1), x_array.reshape(-1, 1))
        y_pred = lin_reg.predict(y_all.reshape(-1, 1))
        for x, xp, y, path in zip(x_array, y_pred, y_all,
                                  subpocket_results['path'].values):
            if (xp == min(y_pred)):
                min_y = y
            if (xp == max(y_pred)):
                max_y = y

            ax.annotate(path,
                        (y, x),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center')

        ax.plot((min_y, max_y), (min(y_pred), max(y_pred)), 'k--', linewidth=1)
        ax.autoscale(tight=False)

        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                                         lw=0, alpha=0)]

        score = "r2 = {:.3f}".format(skmetrics.r2_score(x_array, y_pred))
        labels = []
        labels.append(score)

        ax.legend(handles, labels, loc='best', fontsize='small',
                  fancybox=True, framealpha=0.7,
                  handlelength=0, handletextpad=0)

    # Empty unpaired axis
    if (i % 2 == 0):
        fig.delaxes(axs[int(i / 2)][1])

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(output)
    plt.close()


def main():
    # Parse args
    PELE_sim_paths, csv_file_name, ic50_csv, percentile, traj_fraction, \
        max_steps, metric, output = parse_args()

    all_sim_it = SimIt(PELE_sim_paths)

    print(' - Simulations that will be analyzed:')
    for sim_path in all_sim_it:
        print('   - {}'.format(sim_path.name))

    # TODO get rid of this compatibility issue
    metric = metric.replace('occupancy', 'intersection')

    columns, pretty_metric, units = read_subpocket_dataframe(all_sim_it,
                                                             csv_file_name,
                                                             metric)

    make_plots(all_sim_it, columns, csv_file_name, percentile, traj_fraction,
               max_steps, ic50_csv, output, metric, pretty_metric, units)


if __name__ == "__main__":
    main()
