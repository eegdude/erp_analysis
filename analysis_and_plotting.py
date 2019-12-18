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
plt.rcParams['figure.figsize'] = [12,6]
# %%
# # Create dataset from raw data
# dataset.EpDatasetCreator(markup_path=folders.markup_path,
#                         database_path=folders.database_path_car_ica,
#                         data_folder=folders.data_folder,
#                         reference_mode='average', 
#                         ICA=True,
#                         fit_with_additional_lowpass=True
#                         )
# %%
# Load dataset into memory (if short of memory, use preload=False)
ds = dataset.DatasetReader(data_path=folders.database_path_car_ica, preload=True)

# %%
# blind vs sighted
def single_reg(markup, reg = 'brl_static6_all1', draw=True):
    subset_blind = markup.loc[ (markup['blind'] == 1) &
                                (markup['reg'] == reg)
                                ]
    payload_blind = aps.subset(ds, subset_blind)

    subset_sighted = markup.loc[ (markup['blind'] == 0) &
                                (markup['reg'] == reg)
                                ]
    payload_sighted = aps.subset(ds, subset_sighted)

    if draw:
        aps.plot_evoked_response(data=payload_blind, title='blind subjects')

        aps.plot_evoked_response(data=payload_sighted, title='sighted subjects')

        aps.plot_evoked_response(data={'blind': payload_blind['delta'],
                                    'sighted': payload_sighted['delta']},
                                        title='Target EPs')

        aps.plot_evoked_response(data = {'blind': payload_blind['nontarget'].crop(tmax=0.3),
                                        'sighted': payload_sighted['nontarget'].crop(tmax=0.3)
                                        },
                                        title='Nontarget EPs')


        p = payload_sighted['delta'].plot_topomap(times='peaks', scalings={'eeg':1}, show=False)
        p.suptitle('sighted delta')
        # p.show()

        p = payload_blind['delta'].plot_topomap(times='peaks', scalings={'eeg':1}, show=False)
        p.suptitle('blind delta')
        # p.show()


        p = payload_sighted['nontarget'].crop(tmax=0.3).plot_topomap(times='peaks', scalings={'eeg':1}, show=False)
        p.suptitle('sighted nontarget')
        # p.show()

        p = payload_blind['nontarget'].crop(tmax=0.3).plot_topomap(times='peaks', scalings={'eeg':1}, show=False)
        p.suptitle('blind nontarget')
        # p.show()

    return payload_blind, payload_sighted
#%%
single_reg(ds.markup, reg = 'brl_static6_all1')
single_reg(ds.markup, reg = 'brl_static6_all8')
#%%
def regs(demography = 'blind', draw=True):
    demographies = {'blind':1, 'sighted':0}
    regs = ['brl_static6_all8', 'brl_static6_all1']
    subset_r1 = ds.markup.loc[ (ds.markup['blind'] == demographies[demography]) &
                                (ds.markup['reg'] == regs[0])
                                ]
    payload_r1 = aps.subset(ds, subset_r1)

    subset_r2 = ds.markup.loc[ (ds.markup['blind'] == demographies[demography]) &
                                (ds.markup['reg'] == regs[1])
                                ]
    payload_r2 = aps.subset(ds, subset_r2)

    if draw:
        aps.plot_evoked_response(data={regs[0]: payload_r1['delta'],
                                    regs[1]: payload_r2['delta']},
                                        title=f'Delta EPs in {demography}')

        aps.plot_evoked_response(data={regs[0]: payload_r1['nontarget'].crop(tmax=0.3),
                                    regs[1]: payload_r2['nontarget'].crop(tmax=0.3)},
                                        title=f'nontarget EPs in {demography}')
    return payload_r1, payload_r2
#%%
regs(demography='blind')
regs(demography='sighted')
#%%
def hands(markup, demography='blind', reg='brl_static6_all8', draw=True):
    demographies = {'blind':1, 'sighted':0}

    subset_rh = markup.loc[ (markup['blind'] == demographies[demography]) &
                                (markup['reg'] == reg) &
                                (markup['finger'].isin([7,6,5,4]))
                                ]
    payload_rh = aps.subset(ds, subset_rh)

    subset_lh = markup.loc[ (markup['blind'] == demographies[demography]) &
                                (markup['reg'] == reg) &
                                (markup['finger'].isin([0,1,2,3]))
                                ]
    payload_lh = aps.subset(ds, subset_lh)
    if draw:
        aps.plot_evoked_response(data={ 'right': payload_rh['delta'],
                                        'left': payload_lh['delta']},
                                        title=f'Delta EPs in {demography} \n({reg})')

        aps.plot_evoked_response(data={'right': payload_rh['nontarget'].crop(tmax=0.3),
                                        'left': payload_lh['nontarget'].crop(tmax=0.3)},
                                        title=f'nontarget EPs in {demography} \n({reg})')

        p = payload_rh['delta'].plot_topomap(times='peaks', scalings={'eeg':1}, show=False)
        p.suptitle(f'delta right-hand EPs in {demography} ({reg})')
        # p.show()
        
        p = payload_lh['delta'].plot_topomap(times='peaks', scalings={'eeg':1}, show=False)
        p.suptitle(f'delta left-hand EPs in {demography} ({reg})')
        # p.show()

        p = payload_rh['nontarget'].crop(tmax=0.3).plot_topomap(times='peaks', scalings={'eeg':1}, show=False)
        p.suptitle(f'nontarget right-hand EPs in {demography} ({reg})')
        # p.show()
        
        p = payload_lh['nontarget'].crop(tmax=0.3).plot_topomap(times='peaks', scalings={'eeg':1}, show=False)
        p.suptitle(f'nontarget left-hand EPs in {demography} ({reg})')
        # p.show()

    return payload_lh, payload_rh
#%%
hands(markup=ds.markup, demography='blind', reg='brl_static6_all8')
hands(markup=ds.markup, demography='sighted', reg='brl_static6_all8')


#%%
# nontarget right vs left for sighted
right_hand = []
left_hand = []
condition_names = ['left_hand', 'right_hand']

for user in set(ds.markup.loc[ds.markup['blind'] == 0]['user']): 
    subset_right = ds.markup.loc[ (ds.markup['user'] == user) &
                                (ds.markup['reg'] == 'brl_static6_all8') &
                                (ds.markup['finger'].isin([7,6,5,4]))
                                ]

    subset_left = ds.markup.loc[ (ds.markup['user'] == user) &
                                (ds.markup['reg'] == 'brl_static6_all8') &
                                (ds.markup['finger'].isin([0,1,2,3]))
                                ]

    right_hand.append(aps.subset(ds, subset_right)['nontarget'].crop(tmax=0.3).drop_channels(['ecg', 'A1', 'A2']))
    left_hand.append(aps.subset(ds, subset_left)['nontarget'].crop(tmax=0.3).drop_channels(['ecg', 'A1', 'A2']))


X = [np.array([a.data.T for a in left_hand]), 
    np.array([a.data.T for a in right_hand])]
info = left_hand[0].info
times = left_hand[0].times * 1e3

aps.cluster_and_plot(X, info, times,  condition_names=condition_names,
                    threshold=None, n_permutations=1000, tail=0, n_jobs=4)
# %%
# nontarget right vs left for blind

right_hand = []
left_hand = []
condition_names = ['left_hand', 'right_hand']

for user in set(ds.markup.loc[ds.markup['blind'] == 1]['user']):
    left_hand_payload, right_hand_payload = hands(  markup=ds.markup.loc[ds.markup['user'] == user],
                                                    demography='blind' if ds.markup.loc[ds.markup['user'] == user]['blind'].iloc[0] else 'sighted', #redundant                                                    
                                                    reg='brl_static6_all8',
                                                    draw=False)
    right_hand.append(right_hand_payload['nontarget'].crop(tmax=0.3).drop_channels(['ecg', 'A1', 'A2']))
    left_hand.append(left_hand_payload['nontarget'].crop(tmax=0.3).drop_channels(['ecg', 'A1', 'A2']))



X = [np.array([a.data.T for a in left_hand]), 
    np.array([a.data.T for a in right_hand])]
info = left_hand[0].crop(tmax=0.3).info
times = left_hand[0].crop(tmax=0.3).times * 1e3

aps.cluster_and_plot(X, info, times,  condition_names=condition_names,
                    threshold=None, n_permutations=1000, tail=0)

#%%
# nontarget right vs left for all
right_hand = []
left_hand = []
condition_names = ['left_hand', 'right_hand']

for user in set(ds.markup['user']):
    
    subset_right = ds.markup.loc[ (ds.markup['user'] == user) &
                                (ds.markup['reg'] == 'brl_static6_all8') &
                                (ds.markup['finger'].isin([7,6,5,4]))
                                ]

    subset_left = ds.markup.loc[ (ds.markup['user'] == user) &
                                (ds.markup['reg'] == 'brl_static6_all8') &
                                (ds.markup['finger'].isin([0,1,2,3]))
                                ]

    right_hand.append(aps.subset(ds, subset_right)['nontarget'].crop(tmax=0.3).drop_channels(['ecg', 'A1', 'A2']))
    left_hand.append(aps.subset(ds, subset_left)['nontarget'].crop(tmax=0.3).drop_channels(['ecg', 'A1', 'A2']))


X = [np.array([a.data.T for a in left_hand]), 
    np.array([a.data.T for a in right_hand])]
info = left_hand[0].info
times = left_hand[0].times * 1e3

aps.cluster_and_plot(X, info, times,  condition_names=condition_names,
                    threshold=None, n_permutations=1000, tail=0, n_jobs=4)

# %%
# target blind vs sighted
blind = []
sighted = []
condition_names = ['blind', 'sighted']


for user in set(ds.markup.loc[ds.markup['blind'] == 1]['user']):
    subset_blind = ds.markup.loc[ (ds.markup['user'] == user) &
                                (ds.markup['reg'] == 'brl_static6_all8')
                                ]
    blind.append(aps.subset(ds, subset_blind)['delta'].drop_channels(['ecg', 'A1', 'A2']))

for user in set(ds.markup.loc[ds.markup['blind'] == 0]['user']):
    subset_sighted = ds.markup.loc[ (ds.markup['user'] == user) &
                                (ds.markup['reg'] == 'brl_static6_all8')
                                ]
    sighted.append(aps.subset(ds, subset_sighted)['delta'].drop_channels(['ecg', 'A1', 'A2']))



X = [np.array([a.data.T for a in blind]), 
    np.array([a.data.T for a in sighted])]
info = blind[0].info
times = blind[0].times * 1e3

aps.cluster_and_plot(X, info, times, condition_names=condition_names,
                    threshold=None, n_permutations=1000, tail=1, n_jobs=4)


# %%
# target blind vs sighted - letter averaged
blind = []
sighted = []
condition_names = ['blind', 'sighted']


for user in set(ds.markup.loc[ds.markup['blind'] == 1]['user']):
    subset_blind = ds.markup.loc[ (ds.markup['user'] == user) &
                                (ds.markup['reg'] == 'brl_static6_all8') &
                                ]
    blind.append(aps.subset(ds, subset_blind)['delta'].drop_channels(['ecg', 'A1', 'A2']))

for user in set(ds.markup.loc[ds.markup['blind'] == 0]['user']):
    subset_sighted = ds.markup.loc[ (ds.markup['user'] == user) &
                                (ds.markup['reg'] == 'brl_static6_all8') &
                                ]
    sighted.append(aps.subset(ds, subset_sighted)['delta'].drop_channels(['ecg', 'A1', 'A2']))


X = [np.array([a.data.T for a in blind]), 
    np.array([a.data.T for a in sighted])]
info = blind[0].info
times = blind[0].times * 1e3

aps.cluster_and_plot(X, info, times, condition_names=condition_names,
                    threshold=None, n_permutations=1000, tail=1, n_jobs=4)

# %%
# target vs nontarget
target = []
nontarget = []
condition_names = ['target', 'nontarget']

for user in set(ds.markup['user']):
    subset_target= ds.markup.loc[ (ds.markup['user'] == user) &
                                (ds.markup['reg'] == 'brl_static6_all8')
                                ]
    ep=aps.subset(ds, subset_blind)

    target.append(ep['target'].drop_channels(['ecg', 'A1', 'A2']))
    nontarget.append(ep['nontarget'].drop_channels(['ecg', 'A1', 'A2']))


X = [np.array([a.data.T for a in target]), 
     np.array([a.data.T for a in nontarget])]
info = nontarget[0].info
times = nontarget[0].times * 1e3

aps.cluster_and_plot(X, info, times,  condition_names=condition_names,
                    threshold=None, n_permutations=1000, tail=1, n_jobs=4)



# %%
