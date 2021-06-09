# coding: utf-8
import os
import sys
import pytest
import warnings
try:
    from teilab.utils._warnings import TeiLabImprementationWarning
except ModuleNotFoundError:
    here     = os.path.abspath(os.path.dirname(__file__))
    REPO_DIR = os.path.dirname(here)
    sys.path.append(REPO_DIR)
    print(f"You didn't install 'UiTei-Lab-Course', so add {REPO_DIR} to search path for modules.")
    from teilab.utils._warnings import TeiLabImprementationWarning

def pytest_addoption(parser):
    parser.addoption("--teilab-warnings", choices=["error", "ignore", "always", "default", "module", "once"], default="ignore")

def pytest_configure(config):
    action = config.getoption("teilab_warnings")
    warnings.simplefilter(action, category=TeiLabImprementationWarning)
    warnings.simplefilter(action, category=FutureWarning)
