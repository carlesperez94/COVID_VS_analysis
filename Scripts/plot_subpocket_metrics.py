# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
import os
import sys
from collections import defaultdict

# External imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

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


def parse_args():
    parser = ap.ArgumentParser()

    parser.add_argument("traj_paths", metavar="PATH", type=str,
                        nargs='*',
                        help="Path to PELE trajectory files")

    parser.add_argument("--subpockets", metavar="STR", type=str,
                        help="Subpockets csv file name",
                        default='subpockets.csv')

    parser.add_argument("--ic50", metavar="STR", type=str,
                        help="IC50 csv file name",
                        default=None)

    parser.add_argument('-m', '--metric', default='occupancy',
                        nargs='?', type=str,
                        choices=['occupancy', 'nonpolar_occupancy',
                                 'aromatic_occupancy', 'positive_charge',
                                 'negative_charge'],
                        help='Metric whose distribution will be plotted')

    parser.add_argument('-o', '--output', metavar='STR', type=str,
                        default=None,
                        help='Output name for the resulting plot')

    args = parser.parse_args()

    return args.traj_paths, args.subpockets, args.ic50, args.metric, \
        args.output


def main():
    # Parse args
    PELE_sim_paths, csv_file_name, ic50_csv, metric, output = parse_args()

    all_sim_it = SimIt(PELE_sim_paths)

    print(' - Simulations that will be analyzed:')
    for sim_path in all_sim_it:
        print('   - {}'.format(sim_path.name))

    # TODO get rid of this compatibility issue
    metric = metric.replace('occupancy', 'intersection')

    columns, pretty_metric, units = read_subpocket_dataframe(all_sim_it,
                                                             csv_file_name,
                                                             metric)

    fig, axs = plt.subplots(len(columns), 1, figsize=(20, 15))
    fig.suptitle('Subpocket-LIG {}'.format(pretty_metric))

    for i, col in enumerate(columns):
        axs[i].set_title(col.strip('_' + metric))
        axs[i].set_ylabel('{}-LIG {} ({})'.format(
            pretty_metric, col.strip('_' + metric), units))

    metrics = defaultdict(dict)
    for PELE_sim_path in all_sim_it:

        if (not PELE_sim_path.joinpath(csv_file_name).is_file()):
            print(' - Skipping simulation because '
                  + '{} csv file is missing'.format(pretty_metric))
            continue

        data = pd.read_csv(PELE_sim_path.joinpath(csv_file_name))

        for i, col in enumerate(columns):
            metrics[col][PELE_sim_path.name] = data[col].to_numpy()

    try:
        if (ic50_csv is None):
            raise ValueError
        ic50_data = pd.read_csv(ic50_csv)

        ic50_data['-pIC50'] = - np.log10(ic50_data.loc[:, 'IC50'] / 1000000)
        ic50_data.sort_values(by='-pIC50', inplace=True)

        for i, col in enumerate(columns):
            ordered_intersects = []
            ordered_labels = []
            for path, pic50 in zip(ic50_data['path'], ic50_data['-pIC50']):
                if (path in metrics[col]):
                    ordered_intersects.append(metrics[col][path])
                    ordered_labels.append(path)

            axs[i].boxplot(ordered_intersects,
                           labels=ordered_labels,
                           showfliers=False)

    except ValueError:
        print(' - Unordered plot')
        for i, col in enumerate(columns):
            axs[i].boxplot(list(metrics[col].values()),
                           labels=[i.split('_')[1] for i in
                                   list(metrics[col].keys())],
                           showfliers=False)

    for ax in axs:
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)

    if (output is None):
        output = 'subpocket_{}.png'.format(metric)

    plt.tight_layout(h_pad=5, rect=(0, 0.05, 1, 0.95))
    plt.savefig(output)
    plt.close()


if __name__ == "__main__":
    main()
