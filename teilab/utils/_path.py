#coding: utf-8
import os

__all__ = [
    "UTILS_DIR", "MODULE_DIR", "REPO_DIR", 
]

UTILS_DIR     = os.path.dirname(os.path.abspath(__file__)) #: path/to/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments/teilab/utils
MODULE_DIR    = os.path.dirname(UTILS_DIR)                 #: path/to/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments/teilab
REPO_DIR      = os.path.dirname(MODULE_DIR)                #: path/to/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments