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
plt.rcParams['figure.figsize'] = [16,9]
# %%
# Load dataset into memory (if short of memory, use preload=False)
ds = dataset.DatasetReader(data_path=folders.database_path, preload=True)
#%%
#delta plots for every user
for user in set(ds.markup['user']):
    userdict = {}
    for reg in set(ds.markup['reg']):
        subset = ds.markup.loc[ (ds.markup['user'] == user) &
                                (ds.markup['reg'] == reg)
                                ]
    
        userdict[reg] = aps.subset(ds, subset, drop_channels=['A2', 'Fp1'], reference = 'average')['delta']
    p = aps.plot_evoked_response(userdict, title=f'user {user}')
#%%
#delta plots for grand average
u1 = min(set(ds.markup['user']))
u2 = max(set(ds.markup['user']))
ga_dict = {}
for reg in set(ds.markup['reg']):
    subset = ds.markup.loc[(ds.markup['reg'] == reg)
                            ]

    ga_dict[reg] = aps.subset(ds, subset, drop_channels=['A2', 'Fp1'], reference = 'average')['delta']
p = aps.plot_evoked_response(ga_dict, title=f'grand average, users {u1}-{u2}')


# %%
