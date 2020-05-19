import pandas as pd
import sys
import os
SCRIPT_PATH = os.path.dirname(__file__)
sys.path.append(os.path.abspath(SCRIPT_PATH + '/..'))
from Helpers.Hbonds import HBondLinker, build_linkers


def test_build_linkers():
    data = pd.read_csv('data/hbonds_2.csv')
    data.hbonds = data.hbonds.apply(build_linkers)

    for hb_linkers in data.hbonds:
        for hb_linker in hb_linkers:
            assert isinstance(hb_linker, HBondLinker)
