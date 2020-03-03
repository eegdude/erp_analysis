# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
%load_ext autoreload
%autoreload 2

import pathlib
import pickle
import os

import numpy as np
import pandas as pd

from scipy import signal

import mne
from matplotlib import pyplot as plt

import constants
import dataset
import folders
import iter_topography_fork
import analysis_and_plotting_functions as aps

# %%
plt.rcParams['figure.figsize'] = [12,12]
# %%
# Load dataset into memory (if short of memory, use preload=False)
ds = dataset.DatasetReader(data_path=folders.database_path, preload=True)
#%%
for user in set(ds.markup['user']):
    subset = ds.markup.loc[ (ds.markup['user'] == user) &
                            (ds.markup['reg'] == 'all_neutral')
                            ]

    subset = aps.subset(ds, subset, drop_channels=['A1'])
    aps.plot_evoked_response(subset)


subset = ds.markup.loc[ ds.markup['reg'] == 'all_neutral'
                        ]
subset = aps.subset(ds, subset, drop_channels=['A1'], reference = 'average')
p = aps.plot_evoked_response(subset)


# %%
