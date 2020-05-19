# -*- coding: utf-8 -*-


# Standard imports
import argparse as ap
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import os
from typing import List, Tuple, Optional, Dict

# Local imports
from Helpers import SimIt
from Helpers import ChainConverterForMDTraj, Subpocket
from Helpers import ImpactTemplate
from Helpers.Subpockets import build_residues
from Helpers.Data import build_dataframe_from_results
from Helpers.ReportUtils import extract_metrics

# External imports
import mdtraj as md
import pandas as pd


# Script information
__author__ = "Marti Municoy"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy"
__email__ = "marti.municoy@bsc.es"


def parse_args() -> Tuple[str, str, str, str, str, List[str],
                          Optional[List[str]], Optional[List[float]], str,
                          Optional[int], str, str, Optional[str], bool,
                          Optional[str]]:
    parser = ap.ArgumentParser()
    parser.add_argument("traj_paths", metavar="PATH", type=str,
                        nargs='*',
                        help="Path to PELE trajectory files")

    parser.add_argument("-l", "--ligand_resname",
                        metavar="LIG", type=str, default='LIG',
                        help="Ligand residue name")

    parser.add_argument("-t", "--topology_path",
                        metavar="PATH", type=str,
                        default='output/topologies/topology_0.pdb',
                        help="Relative path to topology")

    parser.add_argument("--template_path",
                        metavar="PATH", type=str,
                        default='DataLocal/Templates/OPLS2005/HeteroAtoms/',
                        help="Relative path to ligand's template folder")

    parser.add_argument("-r", "--report_name",
                        metavar="NAME", type=str,
                        default='report',
                        help="PELE report name")

    parser.add_argument("-s", "--subpocket", nargs='*', action='append',
                        metavar="C:R", type=str, default=[],
                        help="Chain (C), residue (R) of the subset of "
                        + "the residues that define the subpocket")

    parser.add_argument("--subpocket_names", nargs='*',
                        metavar="NAME", type=str, default=None,
                        help="Name of each subpocket")

    parser.add_argument("--subpocket_radii", nargs='*',
                        metavar="RADIUS", type=float, default=None,
                        help="Fixed radius of each subpocket")

    parser.add_argument("-p", "--probe_atom_name",
                        metavar="NAME", type=str, default='CA',
                        help="Name of probe atom that will be used to "
                        + "define the subpocket")

    parser.add_argument("-n", "--processors_number",
                        metavar="N", type=int, default=None,
                        help="Number of processors")

    parser.add_argument("-o", "--output_name",
                        metavar="PATH", type=str, default="subpockets.csv")

    parser.add_argument("--PELE_output_path",
                        metavar="PATH", type=str, default='output',
                        help="Relative path to PELE output folder")

    parser.add_argument("--include_aromaticity",
                        metavar="STR", type=str, default=None,
                        help='Aromaticity analysis is included by retriving '
                        + 'aromaticity data from the supplied file')

    parser.add_argument('--include_rejected_steps',
                        dest='include_rejected_steps',
                        action='store_true')

    parser.add_argument("--alternative_output_path",
                        metavar="PATH", type=str, default=None,
                        help="Alternative path to save output results")

    parser.set_defaults(include_rejected_steps=False)

    args = parser.parse_args()

    return args.traj_paths, args.ligand_resname, args.topology_path, \
        args.template_path, args.report_name, args.subpocket, \
        args.subpocket_names, args.subpocket_radii, args.probe_atom_name, \
        args.processors_number, args.output_name, args.PELE_output_path, \
        args.include_aromaticity, args.include_rejected_steps, \
        args.alternative_output_path


def print_initial_info(all_sim_it: SimIt, subpockets_residues: List[str],
                       subpocket_names: Optional[List[str]]):
    print(' - The following PELE simulation paths will be analyzed:')
    for PELE_sim_path in all_sim_it:
        print('   - {}'.format(PELE_sim_path))

    print(' - The following subpockets will be analyzed:')
    for i, subpocket_residues in enumerate(subpockets_residues):
        if (subpocket_names is not None
                and len(subpocket_names) == len(subpockets_residues)):
            print('   - {}: {}'.format(subpocket_names[i], subpocket_residues))
        else:
            print('   - {}: {}'.format('S{}'.format(i + 1),
                                       subpocket_residues))


def file_check(topology_path: Path, template_path: Path,
               aromaticity_path: Path) -> bool:
    if (not topology_path.is_file()):
        print(' - Skipping simulation because topology file with '
              + 'connectivity was missing')
        return False

    if (not template_path.is_file()):
        print(' - Skipping simulation because template file with '
              + 'ligand parameters was missing')
        return False

    if (aromaticity_path is not None and not aromaticity_path.is_file()):
        print(' - Warning: aromaticity file is missing, aromatic '
              + 'occupancy will not be calculated')

    return True


def get_aromaticities(aromaticity_path: Optional[Path]
                      ) -> Optional[Dict[str, bool]]:
    if aromaticity_path is None:
        return None

    if not aromaticity_path.is_file():
        return None

    data = pd.read_csv(str(aromaticity_path))

    atom_names = [i.strip() for i in data.atom.to_list()]
    aromaticities = map(bool, data.aromatic.to_list())

    return dict(zip(atom_names, aromaticities))


def subpocket_analysis(sim_path: Path, subpockets: list, topology_path: Path,
                       lig_resname: str, trajectory: Path) -> List[Tuple]:
    results = []

    for i, snapshot in enumerate(md.load(str(trajectory),
                                         top=str(topology_path))):
        entry = (sim_path.name,
                 int(trajectory.parent.name),
                 int(''.join(filter(str.isdigit, trajectory.name))),
                 i)  # type: Tuple
        for subpocket in subpockets:
            centroid, _, intersection, nonpolar_intersection, \
                aromatic_intersection, net_charge, positive_charge, \
                negative_charge = subpocket.full_characterize(snapshot,
                                                              lig_resname)
            entry = (entry + (centroid, intersection, nonpolar_intersection,
                              aromatic_intersection, net_charge,
                              positive_charge, negative_charge))

        results.append(entry)

    return results


def get_simulation_subpockets(residues: list, radii: Optional[List[float]],
                              topology_path: Path) -> List[Subpocket]:
    chain_converter = ChainConverterForMDTraj(str(topology_path))

    if (radii is not None and len(radii) != len(residues)):
        print(' - Warning: length of subpocket_radii does not match '
              + 'with length of subpockets, fixed radii are ignored.')
        radii = None

    subpockets = []
    for i, subpocket_residues in enumerate(residues):
        residues = build_residues([(i.split(':')[0],
                                    int(i.split(':')[1]))
                                   for i in subpocket_residues],
                                  chain_converter)

        if (radii is None):
            radius = None
        else:
            radius = radii[i]

        subpockets.append(Subpocket(residues, radius))

    return subpockets


def handle_subpocket_naming(subpocket_names: Optional[List[str]],
                            subpockets: List[Subpocket]
                            ) -> List[str]:
    if (subpocket_names is not None
            and len(subpocket_names) != len(subpockets)):
        print(' - Warning: length of subpocket_names does not match '
              + 'with length of subpockets, custom names are ignored.')
        subpocket_names = None

    if (subpocket_names is None):
        subpocket_names = []
        for i, subpocket in enumerate(subpockets):
            subpocket_names.append('S{}'.format(i + 1))

    return subpocket_names


def add_steps(report_df: pd.DataFrame, report_path: Path,
              subpockets: List[Subpocket],
              include_rejected_steps: bool) -> pd.DataFrame:
    metrics = extract_metrics((report_path, ), (2, 3))[0]

    new_report_df = pd.DataFrame()

    total_steps = []
    accepted_steps = []
    for t_s, a_s in metrics:
        total_steps.append(int(t_s))
        accepted_steps.append(int(a_s))

    if include_rejected_steps:
        for t_s, a_s in zip(total_steps, accepted_steps):
            row = report_df[report_df['model'] == a_s]
            row.insert(4, 'step', t_s)
            new_report_df = pd.concat([new_report_df, row])
    else:
        for a_s in set(accepted_steps):
            row = report_df[report_df['model'] == a_s]
            row.insert(4, 'step', total_steps[accepted_steps.index(a_s)])
            new_report_df = pd.concat([new_report_df, row])

    return new_report_df


def main():
    # Parse args
    PELE_sim_paths, lig_resname, topology_relative_path, \
        template_relative_path, report_name, subpockets_residues, \
        subpocket_names, subpocket_radii, probe_atom_name, proc_number, \
        output_name, PELE_output_path, aromaticity_file, \
        include_rejected_steps, alternative_output_path = parse_args()

    all_sim_it = SimIt(PELE_sim_paths)

    print_initial_info(all_sim_it, subpockets_residues, subpocket_names)

    for PELE_sim_path in all_sim_it:
        print('')
        print(' - Analyzing {}'.format(PELE_sim_path))

        topology_path = PELE_sim_path.joinpath(topology_relative_path)
        template_path = PELE_sim_path.joinpath(template_relative_path,
                                               lig_resname.lower() + 'z')
        if aromaticity_file is not None:
            if alternative_output_path is not None:
                aromaticity_path = Path(alternative_output_path)
                aromaticity_path = aromaticity_path.joinpath(
                    PELE_sim_path.name)
            else:
                aromaticity_path = PELE_sim_path
            aromaticity_path = aromaticity_path.joinpath(aromaticity_file)
        else:
            aromaticity_path = None

        if not file_check(topology_path, template_path, aromaticity_path):
            continue

        ligand_template = ImpactTemplate(str(template_path))

        aromaticities = get_aromaticities(aromaticity_path)

        subpockets = get_simulation_subpockets(subpockets_residues,
                                               subpocket_radii, topology_path)

        for subpocket in subpockets:
            subpocket.set_ligand_atoms(topology_path, lig_resname)
            subpocket.set_ligand_template(ligand_template)
            subpocket.set_ligand_aromaticity(aromaticities)

        # Build PELE iterables
        sim_it = SimIt(PELE_sim_path)
        sim_it.build_traj_it(PELE_output_path, 'trajectory', 'xtc')
        sim_it.build_repo_it(PELE_output_path, report_name)

        trajectories = [traj for traj in sim_it.traj_it]
        reports = [repo for repo in sim_it.repo_it]

        # Handle subpocket naming
        subpocket_names = handle_subpocket_naming(subpocket_names, subpockets)

        # Subpocket analysis
        parallel_function = partial(subpocket_analysis, PELE_sim_path,
                                    subpockets, topology_path, lig_resname)

        print('   - Subpocket analysis')
        with Pool(proc_number) as pool:
            results = pool.map(parallel_function, trajectories)

        data = build_dataframe_from_results(
            results, ['simulation', 'epoch', 'trajectory', 'model']
            + [j for i in zip(
                ["{}_centroid".format(i) for i in subpocket_names],
                ["{}_intersection".format(i) for i in subpocket_names],
                ["{}_nonpolar_intersection".format(i)
                 for i in subpocket_names],
                ["{}_aromatic_intersection".format(i)
                 for i in subpocket_names],
                ["{}_net_charge".format(i) for i in subpocket_names],
                ["{}_positive_charge".format(i) for i in subpocket_names],
                ["{}_negative_charge".format(i) for i in subpocket_names]
            ) for j in i])

        if include_rejected_steps:
            print('   - Considering rejected steps')

        with Pool(proc_number) as pool:
            results = []
            for report in reports:
                epoch = int(report.parent.name)
                trajectory = int(''.join(filter(str.isdigit, report.name)))
                report_csv = data[(data['epoch'] == epoch)
                                  & (data['trajectory'] == trajectory)]
                r = pool.apply_async(add_steps,
                                     (report_csv, report, subpockets,
                                      include_rejected_steps))

                results.append(r)

            g_results = []
            for r in results:
                g_results.append(r.get())

        data = pd.concat(g_results)

        if (alternative_output_path is not None):
            output_path = Path(alternative_output_path)
            output_path = output_path.joinpath(PELE_sim_path.name)
            output_path = output_path.joinpath(output_name)
            try:
                os.makedirs(str(output_path.parent))
            except FileExistsError:
                pass
        else:
            output_path = PELE_sim_path.joinpath(output_name)

        data.to_csv(str(output_path), index=False)


if __name__ == "__main__":
    main()
