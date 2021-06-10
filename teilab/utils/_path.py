#coding: utf-8
import os

__all__ = [
    "UTILS_DIR", "MODULE_DIR", "REPO_DIR", "DATA_DIR", "SAMPLE_LIST_PATH",
]

UTILS_DIR        = os.path.dirname(os.path.abspath(__file__)) #: path/to/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments/teilab/utils
MODULE_DIR       = os.path.dirname(UTILS_DIR)                 #: path/to/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments/teilab
REPO_DIR         = os.path.dirname(MODULE_DIR)                #: path/to/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments
DATA_DIR         = os.path.join(MODULE_DIR, "data")           #: path/to/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments/teilab/data
SAMPLE_LIST_PATH = os.path.join(DATA_DIR, "sample_list.txt")  #: path/to/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments/teilab/data/sample_list.txt