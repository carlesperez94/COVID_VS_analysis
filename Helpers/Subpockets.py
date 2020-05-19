# -*- coding: utf-8 -*-


# Standard imports
import os
import sys
import io
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

# External imports
import numpy as np
import mdtraj as md
from multiprocessing import current_process
import pandas as pd

# Local imports
# SCRIPT_PATH = os.path.dirname(__file__)
# sys.path.append(os.path.abspath(SCRIPT_PATH))
from .Utils import squared_distances
from .Template import ImpactTemplate
from .PELEIterator import SimIt


# Script information
__author__ = "Marti Municoy"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy"
__email__ = "marti.municoy@bsc.es"


# Constants
FOUR_THIRDS_PI = 4.1887902047863905


class Residue(object):
    def __init__(self, chain_id: int, res_id: int):
        if (not isinstance(chain_id, int)):
            raise TypeError('Invalid \'chain_id\', it must be an integer')
        self._chain_id = chain_id
        if (not isinstance(res_id, int)):
            raise TypeError('Invalid \'res_id\', it must be an integer')
        self._res_id = res_id

    @property
    def chain_id(self) -> int:
        return self._chain_id

    @property
    def res_id(self) -> int:
        return self._res_id

    def __repr__(self) -> str:
        return "{}:{}".format(self.chain_id, self.res_id)


class ChainConverterForMDTraj(object):
    def __init__(self, path_to_trajectory: str):
        self._path_to_trajectory = path_to_trajectory
        self._map = {}  # type: Dict[str, int]
        self._build_map()

    @property
    def path_to_trajectory(self) -> str:
        return self._path_to_trajectory

    @property
    def map(self) -> Dict[str, int]:
        return self._map

    def _build_map(self):
        with open(self.path_to_trajectory, 'r') as f:
            self._map = {}
            id_counter = 0
            for line in f:
                if (len(line) > 80):
                    chain = line[21]

                    if (chain not in self.map):
                        self.map[chain.upper()] = id_counter
                        id_counter += 1


class Point(np.ndarray):
    def __new__(cls, coords: Union[Tuple[float], List[float]]) -> np.array:
        if (len(coords) != 3):
            raise ValueError('Size of \'coords\' must be 3')

        try:
            for c in coords:
                if (not isinstance(float(c), float)):
                    raise ValueError
        except ValueError:
            raise ValueError('Wrong value found in \'coords\'')

        return np.asarray(coords).view(cls)

    @property
    def x(self) -> float:
        return self[0]

    @property
    def y(self) -> float:
        return self[1]

    @property
    def z(self) -> float:
        return self[2]

    def write_to_PDB(self, path: Optional[str] = None,
                     out_file: Optional[str] = None,
                     point_id: int = 0):
        if (path is None and out_file is None):
            raise ValueError('Either a \'path\' or a \'out_file\' need to '
                             + 'be supplied')

        if (out_file is not None):
            if (not isinstance(out_file, io.IOBase)):
                raise ValueError('Invalid \'out_file\'')
            out_file.write('ATOM    {:3d}  '.format(point_id)
                           + 'CEN BOX A {:3d} '.format(point_id)
                           + '{:>11.3f}{:>8.3f}{:>8.3f}'.format(*self)
                           + '  1.00  0.00\n')

        if (path is not None):
            if (not isinstance(path, str)):
                raise ValueError('Invalid \'path\'')
                path = Path(path)
                if (not path.parent.isdir()):
                    raise ValueError('Path {} does not exist'.format(path))

            with open(str(path), 'w') as f:
                f.write('ATOM    {:3d}  '.format(point_id)
                        + 'CEN BOX A {:3d} '.format(point_id)
                        + '{:>11.3f}{:>8.3f}{:>8.3f}'.format(*self)
                        + '  1.00  0.00\n')


class Subpocket(object):
    def __init__(self, list_of_residues: List[Residue],
                 radius: Optional[Union[str, float, int]] = None):
        self._list_of_residues = list_of_residues
        if (radius is None):
            self.fixed_radius = None
        else:
            try:
                self.fixed_radius = float(radius)
            except ValueError:
                raise ValueError('Wrong subpocket radius')
        self._ligand_atoms = []  # type: list
        self._ligand_template = None  # type: Optional[ImpactTemplate]
        self._ligand_aromaticity = None  # type: Optional[Dict[str, bool]]

    @property
    def list_of_residues(self) -> List[Residue]:
        return self._list_of_residues

    @property
    def ligand_atoms(self) -> list:
        return self._ligand_atoms

    @property
    def ligand_template(self) -> Optional[ImpactTemplate]:
        return self._ligand_template

    @property
    def ligand_aromaticity(self) -> Optional[Dict[str, bool]]:
        return self._ligand_aromaticity

    def set_ligand_atoms(self, topology: md.Trajectory, lig_resname: str):
        snapshot = md.load(str(topology))

        self._ligand_atoms = []
        atom_ids = snapshot.top.select('resname {}'.format(lig_resname))
        for atom_id in atom_ids:
            atom = snapshot.top.atom(atom_id)
            self._ligand_atoms.append(atom)

    def set_ligand_template(self, lig_template: ImpactTemplate):
        self._ligand_template = lig_template

    def set_ligand_aromaticity(self,
                               ligand_aromaticity: Dict[str, bool]):
        self._ligand_aromaticity = ligand_aromaticity

    def get_residue_coords(self, snapshot: md.Trajectory) -> np.array:
        coords = []
        for r in self.list_of_residues:
            index = snapshot.top.select('chainid {} '.format(r.chain_id)
                                        + 'and residue {} '.format(r.res_id)
                                        + 'and name CA')
            coords.append(snapshot.xyz[0][index][0] * 10)

        return np.stack(coords)

    def get_centroid(self, snapshot: md.Trajectory
                     ) -> Tuple[Point, np.array]:
        coords = self.get_residue_coords(snapshot)

        return Point(np.mean(coords, axis=0)), coords

    def get_default_radius(self, snapshot: md.Trajectory
                           ) -> Tuple[float, Point]:
        centroid, coords = self.get_centroid(snapshot)

        radius = np.sqrt(np.max(
            [squared_distances(c, centroid) for c in coords]))

        return radius, centroid

    def get_volume(self, snapshot: md.Trajectory) -> float:
        radius = self.fixed_radius  # type: Optional[float]

        if (self.fixed_radius is None):
            radius, _ = self.get_default_radius(snapshot)

        return FOUR_THIRDS_PI * np.power(radius, 3)

    def _calculate_intersections(self, atom_radii: Dict[str, float],
                                 atom_coords: Dict[str, np.array],
                                 centroid: Point, s_r: float
                                 ) -> Dict[str, float]:
        intersections = {}
        for atom, a_r in atom_radii.items():
            # Calculate bounds
            lower_bound = np.abs(s_r - a_r)
            upper_bound = s_r + a_r

            # Calculate distance d between centers
            coords = atom_coords[atom]
            d = np.linalg.norm(centroid - coords)

            # Minimum radius
            min_radius = np.min((a_r, s_r))

            # Calculate intersections
            if (d >= upper_bound):
                intersections[atom] = 0.
            elif (d <= lower_bound):
                intersections[atom] = FOUR_THIRDS_PI * np.power(min_radius, 3)
            else:
                intersections[atom] = np.pi
                intersections[atom] *= np.power(s_r + a_r - d, 2)
                dd = np.power(d, 2)
                da = 2 * d * a_r
                aa = -3 * np.power(a_r, 2)
                ds = 2 * d * s_r
                ss = -3 * np.power(s_r, 2)
                sa = 6 * s_r * a_r
                intersections[atom] *= dd + da + aa + ds + ss + sa
                intersections[atom] /= 12 * d

        return intersections

    def get_intersections(self, snapshot: md.Trajectory, ligand_resname: str,
                          centroid: Point = None, radius: float = None
                          ) -> Dict[str, float]:
        if (centroid is None or radius is None):
            if (self.fixed_radius is None):
                radius, centroid = self.get_default_radius(snapshot)
            else:
                radius = self.fixed_radius
                centroid, _ = self.get_centroid(snapshot)

        lig = snapshot.topology.select('resname {}'.format(ligand_resname))

        atom_radius = {}
        atom_coords = {}
        for atom_id in lig:
            atom = snapshot.top.atom(atom_id)
            name = atom.name
            atom_radius[name] = atom.element.radius * 10  # In angstroms
            atom_coords[name] = snapshot.xyz[0][atom_id] * 10  # In angstroms

        intersections = self._calculate_intersections(atom_radius, atom_coords,
                                                      centroid, radius)

        return intersections

    def get_sum_of_intersections(self, snapshot: md.Trajectory,
                                 ligand_resname: str, centroid: Point = None,
                                 radius: float = None) -> Dict[str, float]:
        intersections = self.get_intersections(snapshot, ligand_resname,
                                               centroid, radius)
        return np.sum(list(intersections.values()))

    def get_nonpolar_intersection(self, intersections: Dict[str, float]
                                  ) -> float:
        try:
            assert isinstance(self.ligand_template, ImpactTemplate)
        except AssertionError:
            raise AssertionError('Ligand template was not set')

        nonpolar_intersection = float(0)
        for atom in self.ligand_atoms:
            intersection = intersections[atom.name]
            element = atom.element.name
            charge = self.ligand_template.get_parameter_by_name(atom.name,
                                                                'charge')

            if ((element == 'carbon' or element == 'hydrogen')
                    and (charge <= 0.2)):
                nonpolar_intersection += intersection

        return nonpolar_intersection

    def get_aromatic_intersection(self, intersections: Dict[str, float]
                                  ) -> float:
        try:
            assert isinstance(self.ligand_aromaticity, dict)
        except AssertionError:
            return -1.0

        aromatic_intersection = float(0)
        for atom in self.ligand_atoms:
            intersection = intersections[atom.name]
            try:
                aromatic = self.ligand_aromaticity[atom.name]
            except KeyError:
                if current_process()._identity == (1,):  # type: ignore
                    print('     - Warning: invalid ligand aromaticity file, '
                          + 'aromatic occupancy will not be calculated')
                self._ligand_aromaticity = None
                return -1.0

            if aromatic:
                aromatic_intersection += intersection

        return aromatic_intersection

    def get_charge(self, intersections: Dict[str, float]
                   ) -> Tuple[float, float, float]:
        try:
            assert isinstance(self.ligand_template, ImpactTemplate)
        except AssertionError:
            raise AssertionError('Ligand template was not set')

        net_charge = float(0)
        positive_charge = float(0)
        negative_charge = float(0)
        for atom in self.ligand_atoms:
            intersection = intersections[atom.name]

            # Radius in angstroms
            radius = atom.element.radius * 10
            volume = FOUR_THIRDS_PI * np.power(radius, 3)
            charge = self.ligand_template.get_parameter_by_name(
                atom.name, 'charge')
            norm_charge = charge * intersection / volume

            net_charge += norm_charge

            if norm_charge > 0:
                positive_charge += norm_charge
            elif norm_charge < 0:
                negative_charge += norm_charge

        return net_charge, positive_charge, negative_charge

    def full_characterize(self, snapshot: md.Trajectory, ligand_resname: str
                          ) -> Tuple[Point, float, Dict[str, float], float,
                                     float, float, float, float]:
        if (self.fixed_radius is None):
            radius, centroid = self.get_default_radius(snapshot)
        else:
            radius = self.fixed_radius
            centroid, _ = self.get_centroid(snapshot)
        intersections = self.get_intersections(snapshot, ligand_resname,
                                               centroid, radius)

        np_intersection = self.get_nonpolar_intersection(intersections)

        aromatic_intersection = self.get_aromatic_intersection(intersections)

        net_charge, positive_charge, negative_charge = self.get_charge(
            intersections)

        return centroid, FOUR_THIRDS_PI * np.power(radius, 3), \
            np.sum(list(intersections.values())), np_intersection, \
            aromatic_intersection, net_charge, positive_charge, negative_charge


def build_residues(residues_list: List[Tuple[Union[str, int], int]],
                   chain_ids_map: ChainConverterForMDTraj = None
                   ) -> List[Residue]:
    """
    It builds an array of Residue objects

    Parameters
    ----------
    residues_list : list of (chain_id, residue_id)
        It is a list of residue identifiers that will be used to build
        Residue objectes

    Returns
    -------
    residue_objects : list of Residue objects
        It contains the Residue objectes that have been created with the
        given identifiers
    """

    residue_objectes = []

    for r in residues_list:
        # Check format
        try:
            c_id, r_id = r
        except ValueError:
            raise ValueError('Wrong format in \'residues_list\', only '
                             + '2-dimensional tuples are allowed: '
                             + '(chain_id, residue_id)')

        # Check types
        if (not isinstance(c_id, int) and not isinstance(c_id, str)):
            raise TypeError('Wrong \'chain_id\', it can only be a '
                            + 'string or an integer')
        elif (isinstance(c_id, str) and len(c_id) > 1):
            raise TypeError('Wrong \'chain_id\', only single-character '
                            + 'strings are acepted')
        if (not isinstance(r_id, int)):
            raise TypeError('Wrong \'residue_id\', it can only be '
                            + 'an integer')
        if (chain_ids_map is not None):
            if (not isinstance(chain_ids_map, ChainConverterForMDTraj)):
                raise TypeError('Wrong \'chain_ids_map\', it has to be '
                                + 'an instance of ChainCovertedForMDTraj '
                                + 'class')

        # Convert chain_id to the right id
        if (isinstance(c_id, str)):
            if (chain_ids_map is None):
                raise ValueError('When a character is supplied, a '
                                 + '\'chain_ids_map\' is also required')
            c_id = chain_ids_map.map[c_id.upper()]

        residue_objectes.append(Residue(c_id, r_id))

    return residue_objectes


def read_subpocket_dataframe(all_sim_it: SimIt, csv_file_name: str,
                             metric: str) -> Tuple[List[str], str, str]:
    columns = []
    for PELE_sim_path in all_sim_it:
        if (not PELE_sim_path.joinpath(csv_file_name).is_file()):
            print(' - Skipping simulation because subpockets csv file '
                  + 'was missing')
            continue

        data = pd.read_csv(PELE_sim_path.joinpath(csv_file_name))
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

        for col in data.columns:
            if (col.endswith(metric)):
                if (metric == 'intersection'
                        and col.endswith('nonpolar_intersection')):
                    continue
                if (metric == 'intersection'
                        and col.endswith('aromatic_intersection')):
                    continue
                if (col not in columns):
                    columns.append(col)

    pretty_metric = metric.replace('_', ' ')
    if (metric == 'intersection' or metric == 'nonpolar_intersection'
            or metric == 'aromatic_intersection'):
        if (len(pretty_metric.split()) > 1):
            pretty_metric = pretty_metric.split()[0] + ' volume ' + \
                pretty_metric.split()[1]
        else:
            pretty_metric = 'volume ' + pretty_metric
        units = '$\AA^3$'
        # TODO get rid of this compatibility issue
        pretty_metric = pretty_metric.replace('intersection', 'occupancy')
    else:
        pretty_metric += ' occupancy'
        units = 'a.u.'

    print('   - Subpockets found:')
    for col in columns:
        print('     - {}'.format(col.strip('_' + metric)))

    if (len(columns) == 0):
        raise ValueError('No subpocket {} '.format(pretty_metric)
                         + 'were found in the simulation paths that '
                         + 'were supplied')

    return columns, pretty_metric, units
