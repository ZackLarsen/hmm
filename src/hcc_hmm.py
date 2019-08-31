
"""
Hidden Markov Model (HMM) Python Implementation
Author: Zack Larsen
Date: August 31, 2019
"""

import sys
import os
from collections import Counter, namedtuple

import numpy as np
import numba
from numba import jit
import importlib
from sklearn.metrics import cohen_kappa_score
import json
import pickle


# Windows:
homedir = 'C:\\Users\\'
# Mac OS X:
#homedir = '/Users/zacklarsen/Zack_Master/Projects/Work Projects/hmm'

sys.path.append(os.path.join(homedir, 'src/'))
from hmm import *









