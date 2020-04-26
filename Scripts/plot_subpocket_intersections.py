# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
from pathlib import Path
import os
import sys
from collections import defaultdict

# External imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Pool

# Local imports
SCRIPT_PATH = os.path.dirname(__file__)
sys.path.append(os.path.abspath(SCRIPT_PATH + '/..'))
from Helpers.Utils import convert_string_to_numpy_array
from Helpers.PELEIterator import SimIt
from Helpers.ReportUtils import extract_metrics


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

    parser.add_argument("--csv", metavar="STR", type=str,
                        help="Intersections csv file name",
                        default='intersections.csv')

    args = parser.parse_args()

    return args.traj_paths, args.csv


def analyze_subpocket_intersections(report_csv, report_path, subpockets):
    intersections = defaultdict(list)
    metrics = extract_metrics((report_path, ), (3, ))[0]

    accepted_steps = []
    for m in metrics:
        accepted_steps.append(int(m[0]))

    for a_s in accepted_steps:
        row = report_csv[report_csv['model'] == a_s]
        for sp in subpockets:
            value = float(row['{}_intersection'.format(sp)])
            intersections[sp].append(value)

    return intersections


def main():
    # Parse args
    PELE_sim_paths, csv_file_name = parse_args()

    all_sim_it = SimIt(PELE_sim_paths)

    print(' - Simulations that will be analyzed:')
    for sim_path in all_sim_it:
        print('   - {}'.format(sim_path.name))

    subpockets = []
    for PELE_sim_path in all_sim_it:
        if (not PELE_sim_path.joinpath(csv_file_name).is_file()):
            continue

        data = pd.read_csv(PELE_sim_path.joinpath(csv_file_name))
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

        for s in data.columns:
            if (s not in subpockets):
                subpockets.append(s)

    if (len(subpockets) == 0):
        raise ValueError('No subpockets were found in the simulation paths ' +
                         'that were supplied')

    fig, axs = plt.subplots(len(subpockets), 1, figsize=(10, 8))
    fig.suptitle('Subpocket-LIG volume intersection')

    for i, subpocket in enumerate(subpockets):
        axs[i].set_title(subpocket)
        axs[i].set_ylabel('{}-LIG volume intersection ($\AA^3$)'.format(subpocket))

    intersects = defaultdict(dict)
    for PELE_sim_path in all_sim_it:

        if (not PELE_sim_path.joinpath(csv_file_name).is_file()):
            print(' - Skipping simulation because intersection csv file ' +
                  'is missing')
            continue

        data = pd.read_csv(PELE_sim_path.joinpath(csv_file_name))

        for i, subpocket in enumerate(subpockets):
            intersects[subpocket][PELE_sim_path.name] = \
                data[subpocket].to_numpy()

    for i, subpocket in enumerate(subpockets):
        print(intersects[subpocket])
        axs[i].boxplot(list(intersects[subpocket].values()),
                       labels=list(intersects[subpocket].keys()))

    plt.tight_layout(rect=(0, 0, 0.95, 0.95))
    plt.savefig('subpocket_intersections.png')
    plt.close()


if __name__ == "__main__":
    main()