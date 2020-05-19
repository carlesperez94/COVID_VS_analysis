# -*- coding: utf-8 -*-


# Standard imports
from pathlib import Path
from typing import Tuple, List, Optional, Union, FrozenSet, Dict

# External imports
import pandas as pd


# Script information
__author__ = "Marti Municoy"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Marti Municoy"
__email__ = "marti.municoy@bsc.es"


class HBondLinker(object):
    def __init__(self, chain: str, residue: str,
                 atoms: Union[List[str], Tuple[str, ...], None] = None):
        self._chain = chain
        self._residue = residue
        self.set_atoms(atoms)

    @property
    def chain(self) -> str:
        return self._chain

    @property
    def residue(self) -> str:
        return self._residue

    @property
    def atoms(self) -> FrozenSet[str]:
        return self._atoms

    def __str__(self) -> str:
        return 'HBondLinker ' + self.__repr__()

    def __repr__(self) -> str:
        return '{}:{}:'.format(self.chain, self.residue) \
            + ','.join(self.atoms)

    def __eq__(self, other) -> bool:
        return (self.chain, self.residue, self.atoms) == \
            (other.chain, other.residue, other.atoms)

    def __hash__(self) -> int:
        return hash((self.chain, self.residue, self.atoms))

    def set_chain(self, chain: str):
        self._chain = chain

    def set_residue(self, residue: str):
        self._residue = residue

    def set_atoms(self,
                  atoms: Union[List[str], Tuple[str, ...], None] = None):
        if atoms is None:
            self._atoms = frozenset(('*', ))
        else:
            self._atoms = frozenset(atoms)

    def match_with(self, other: 'HBondLinker') -> bool:
        if (self.chain, self.residue) == (other.chain, other.residue):
            if '*' in self.atoms:
                return True
            if '*' in other.atoms:
                return True
            if len(self.atoms.intersection(other.atoms)) > 0:
                return True
        return False

    def match_with_any(self, others: List['HBondLinker']
                       ) -> Tuple[bool, Optional['HBondLinker']]:
        for other in others:
            if self.match_with(other):
                return True, other
        return False, None


def get_hbond_linkers(h_bonds: List[str]) -> List[HBondLinker]:
    hb_linker_list = []  # type: List[HBondLinker]
    for hb in h_bonds:
        # hb has the following format CHAIN:RESIDUE[:ATOM1,ATOM2]
        hb_data = hb.split(':')
        if len(hb_data) not in (2, 3):
            raise ValueError('Error: H bonds '
                             + '\'{}\' '.format(h_bonds)
                             + 'have a wrong format')
        # In case there are no atoms specified, take them all
        if len(hb_data) == 2:
            chain, residue = hb_data[:2]
            atoms = None
        # In case atoms are specified
        elif len(hb_data) == 3:
            chain, residue = hb_data[:2]
            atoms = tuple(hb_data[2].split(','))
        # Otherwise, wrong format
        else:
            raise ValueError('Error: H bonds '
                             + '\'{}\' '.format(h_bonds)
                             + 'have a wrong format')
        hb_linker = HBondLinker(chain, residue, atoms)

        hb_linker_list.append(hb_linker)

    return hb_linker_list


def check_hbonds_linkers(hbond_linkers: List[HBondLinker]):
    hb_linkers_found = set()  # type: Set[HBondLinker]
    for hb_linker in hbond_linkers:
        if hb_linker in hb_linkers_found:
            raise ValueError('{} defined twice'.format(hb_linker))

        for hb_linker_found in hb_linkers_found:
            if (hb_linker_found.chain == hb_linker.chain
                    and hb_linker_found.residue == hb_linker.residue
                    and ('*' in hb_linker_found.atoms
                         or '*' in hb_linker.atoms
                         or len(hb_linker_found.atoms.intersection(
                            hb_linker.atoms)) != 0)):
                raise ValueError('{} and {} share some atoms'.format(
                    hb_linker_found, hb_linker))

        hb_linkers_found.add(hb_linker)


def print_hbonds(hbond_linkers: List[HBondLinker]):
    for hbl in hbond_linkers:
        print('   - {}:{}:{}'.format(hbl.chain, hbl.residue,
                                     ','.join(hbl.atoms)))


def build_linkers(raw_linkers):
    hb_bunch = raw_linkers.strip('\'\"[]')
    if not hb_bunch:
        return []
    hbs = hb_bunch.split()
    hbs = [i.strip(',[]\'\"') for i in hbs]
    try:
        hb_linkers = get_hbond_linkers(hbs)
    except ValueError:
        print('   - Warning: line \'{}\' '.format(raw_linkers)
              + 'has wrong format. H bonds in this line will be ignored.')
        hb_linkers = []
    return hb_linkers


def extract_hbond_linkers(hbonds_path: Path) -> Tuple[pd.DataFrame, int, int]:
    output_info_path = Path(str(hbonds_path).replace(hbonds_path.suffix, '')
                            + '.info')

    if (output_info_path.is_file()):
        with open(str(output_info_path), 'r') as file:
            # Skip four header lines
            file.readline()
            n_donors = int(file.readline().split()[0])
            n_acceptors = int(file.readline().split()[0])
            file.readline()
    else:
        print('   - Warning: H bonds info output file not found')
        n_donors = -1
        n_acceptors = -1

    data = pd.read_csv(str(hbonds_path))

    data.hbonds = data.hbonds.apply(build_linkers)

    return data, n_donors, n_acceptors


def hbond_fulfillment(data: pd.DataFrame, golden_hbonds_1: List[HBondLinker],
                      golden_hbonds_2: List[HBondLinker],
                      minimum_g2_conditions: int):
    # Compatibility with older pandas versions
    try:
        hb_linkers = data.hbonds.to_list()
    except AttributeError:
        hb_linkers = data.hbonds.values

    total_fulfillments = 0
    fulfillments_by_g1_hbond = {}  # type: Dict[HBondLinker, int]
    fulfillments_by_g2_hbond = {}  # type: Dict[HBondLinker, int]
    mask = []
    for _hb_linkers in hb_linkers:
        g1_matchs = 0
        g2_matchs = 0
        for hb_linker in _hb_linkers:
            g1_match, other_g1_hb = hb_linker.match_with_any(golden_hbonds_1)
            if g1_match:
                g1_matchs += 1
                fulfillments_by_g1_hbond[other_g1_hb] = \
                    fulfillments_by_g1_hbond.get(other_g1_hb, 0) + 1

            g2_match, other_g2_hb = hb_linker.match_with_any(golden_hbonds_2)
            if g2_match:
                g2_matchs += 1
                fulfillments_by_g2_hbond[other_g2_hb] = \
                    fulfillments_by_g2_hbond.get(other_g2_hb, 0) + 1

        if ((g1_matchs == len(golden_hbonds_1))
                and (g2_matchs >= minimum_g2_conditions)):
            total_fulfillments += 1
            accepted = True
        else:
            accepted = False

        mask.append(accepted)

    return total_fulfillments, len(hb_linkers), fulfillments_by_g1_hbond, \
        fulfillments_by_g2_hbond, data[mask]
