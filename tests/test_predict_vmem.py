"""Short test for vmem prediction."""
# Copyright 2018-2023 TeNPy Developers, GNU GPLv3


# force python to use the contents of this repo and no other installation:
import sys
import os
sys.path.insert(0,os.path.join(os.getcwd(), "."))


import tenpy.algorithms as algo
import tenpy.models as mods
import tenpy.networks.mps as mps
from tenpy.algorithms import dmrg
import numpy as np #TODO: remove
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def test_bosonic_model_TEBD():
    """Test with a bosonic chain."""
    L = 15
    model = mods.hubbard.BoseHubbardChain({"conserve":None, "U":1, "t":1, "bc_MPS": "finite", "L": L, "n_max": 4})
    # Test TEBD first
    psi = mps.MPS.from_product_state(model.lat.mps_sites(), [0]*L)
    engine = algo.tebd.TEBDEngine(psi, model, {"trunc_params": {"chi_max": 33}})
    # expected value:
    # bond | 0   | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 | 11 | 12 | 13 | 14 | 15 |
    # chi  | 5   | 10 | 15 | 20 | 25 | 30 | 33 | 33 | 33 | 33 | 30 | 25 | 20 | 15 | 10 | 5  |
    # add matrices:
    num_entries = 5*10 + 10*15 + 15*20 + 20*25 + 25*30 + 30*33 + 33*33 + 33*33 + 33*33 + 33*30 + 30*25 + 25*20 + 20*15 + 15*10 + 10*5
    num_entries *= 5 # physical leg
    # expected MPS vmem:
    MPS_vmem = (num_entries * 8) / 1024
    assert abs(engine.estimate_RAM(mini=0) - MPS_vmem) < 1e-10, "TEBD RAM did not match expectation (expected: %d, gotten:%d)" % (MPS_vmem, engine.estimate_RAM())


def test_bosonic_model_DMRG():
    """Test with a bosonic chain."""
    L = 15
    model = mods.hubbard.BoseHubbardChain({"U":1, "t":1, "bc_MPS": "finite", "L": L, "n_max": 4})
    psi = mps.MPS.from_product_state(model.lat.mps_sites(), [0]*L)
    engine = algo.dmrg.TwoSiteDMRGEngine(psi, model, {"trunc_params": {"chi_max": 33}})
    # expected value:
    # bond | 0   | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 | 11 | 12 | 13 | 14 | 15 |
    # chi  | 5   | 10 | 15 | 20 | 25 | 30 | 33 | 33 | 33 | 33 | 30 | 25 | 20 | 15 | 10 | 5  |
    # add matrices:
    num_entries = 5*10 + 10*15 + 15*20 + 20*25 + 25*30 + 30*33 + 33*33 + 33*33 + 33*33 + 33*30 + 30*25 + 25*20 + 20*15 + 15*10 + 10*5
    num_entries *= 5 # physical leg
    # expected MPS vmem:
    MPS_vmem = (num_entries * 8) / 1024
    # environment:
    MPS_env_vmem = sum(np.array([5, 10, 15, 20, 25, 30, 33, 33, 33, 33, 30, 25, 20, 15, 10])**2 * 5) * 8 / 1024
    MPO_vmem = (4**2*5**2 * (L-2) * 8 + 2*4*5**2* 2 * 8) / 1024
    assert engine.estimate_RAM(mini=0) == (MPS_vmem+MPS_env_vmem+MPO_vmem)/8, "DMRG RAM did not match expectation (expected: %d, gotten:%d)" % ((MPS_vmem+MPS_env_vmem+MPO_vmem)/8, engine.estimate_RAM())



test_bosonic_model_TEBD()
test_bosonic_model_DMRG()
