# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
%load_ext autoreload
%autoreload 2

import pathlib
import pickle
import os

import numpy as np
from scipy import signal

import mne
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import constants
import dataset
import folders
import iter_topography_fork
import analysis_and_plotting_functions

# %%
plt.rcParams['figure.figsize'] = [20,10]
# %%
dataset.EpDatasetCreator(markup_path=folders.markup_path,
                        database_path=folders.database_path_ica,
                        data_folder=folders.data_folder,
                        reference_mode='average', 
                        ICA=True,
                        fit_with_additional_lowpass=True
                        )
# %%
ds = dataset.DatasetReader(data_path=folders.database_path_car_ica, preload=True)
#
# %%
def subset(ds, submarkup, reg='brl_static6_all8'):
    
    subset_t = submarkup.loc[ (submarkup['reg'] == reg) &
                            (submarkup['is_target'] == 1) 
                        ]
    subset_nt = submarkup.loc[ (submarkup['reg'] == reg) &
                            (submarkup['is_target'] == 0) 
                        ]

    evoked_t = ds.create_mne_evoked_from_subset(subset_t).apply_baseline((0,0))
    evoked_nt = ds.create_mne_evoked_from_subset(subset_nt).apply_baseline((0,0))
    evoked_delta = mne.EvokedArray(info = ds.info,
                                        data = evoked_t._data - evoked_nt._data,
                                        tmin = constants.epochs_tmin
                                        )
    payload = {
                'target': evoked_t,
                'nontarget': evoked_nt,
                'delta': evoked_delta
                }
    return payload
# %%
# blind vs sighted

subset_blind = ds.markup.loc[ ds.markup['blind'] == 1]
payload_blind = subset(ds, subset_blind)
payload_blind['delta'].plot_topomap(times='peaks', scalings={'eeg':1})
analysis_and_plotting_functions.plot_evoked_response(data = payload_blind)

subset_sighted = ds.markup.loc[ ds.markup['blind'] == 0]
payload_sighted = subset(ds, subset_sighted)
payload_sighted['delta'].plot_topomap(times='peaks', scalings={'eeg':1})
analysis_and_plotting_functions.plot_evoked_response(data = payload_sighted)


analysis_and_plotting_functions.plot_evoked_response(data = {'blind': payload_blind['delta'],
                                                             'sighted': payload_sighted['delta']
                                                            })



analysis_and_plotting_functions.plot_evoked_response(data = {'blind': payload_blind['nontarget'].crop(tmax=0.3),
                                                             'sighted': payload_sighted['nontarget'].crop(tmax=0.3)
                                                            })

payload_blind['nontarget'].crop(tmax=0.3).plot_topomap(times='peaks', scalings={'eeg':1})
payload_sighted['nontarget'].crop(tmax=0.3).plot_topomap(times='peaks', scalings={'eeg':1})


# %%
# late blind vs early blind
subset_blind = ds.markup.loc[ ds.markup['blindness_age'] == 0]
payload_blind = subset(ds, subset_blind)

subset_sighted = ds.markup.loc[ ds.markup['blindness_age'] >0]
payload_sighted = subset(ds, subset_sighted)


analysis_and_plotting_functions.plot_evoked_response(data = {'early_blind': payload_blind['delta'],
                                                             'late_blind': payload_sighted['delta']
                                                            })



analysis_and_plotting_functions.plot_evoked_response(data = {'early_blind': payload_blind['nontarget'].crop(tmax=0.3),
                                                             'late_blind': payload_sighted['nontarget'].crop(tmax=0.3)
                                                            })

payload_blind['nontarget'].crop(tmax=0.3).plot_topomap(times='peaks', scalings={'eeg':1})
payload_sighted['nontarget'].crop(tmax=0.3).plot_topomap(times='peaks', scalings={'eeg':1})
print ()




# %%

# late blind vs early blind
subset_blind = ds.markup.loc[ (ds.markup['blind'] == 1) &
                                (ds.markup['finger'] == 1]

payload_sighted = subset(ds, subset_sighted)


analysis_and_plotting_functions.plot_evoked_response(data = {'early_blind': payload_blind['delta'],
                                                             'late_blind': payload_sighted['delta']
                                                            })



analysis_and_plotting_functions.plot_evoked_response(data = {'early_blind': payload_blind['nontarget'].crop(tmax=0.3),
                                                             'late_blind': payload_sighted['nontarget'].crop(tmax=0.3)
                                                            })

payload_blind['nontarget'].crop(tmax=0.3).plot_topomap(times='peaks', scalings={'eeg':1})
payload_sighted['nontarget'].crop(tmax=0.3).plot_topomap(times='peaks', scalings={'eeg':1})
print ()

#%%
blind_markup = ds.markup.loc[ds.markup['blind'] == 1]
blind_evoked = []
for user in set(blind_markup['user']):
    us = blind_markup.loc[blind_markup['user'] == user]
    evoked = subset(ds, us)['target']
    print(evoked)
    blind_evoked.append(evoked)

sighted_markup = ds.markup.loc[ds.markup['blind'] == 1]
sighted_evoked = []
for user in set(sighted_markup['user']):
    us = sighted_markup.loc[sighted_markup['user'] == user]
    evoked = subset(ds, us)['nontarget']
    print(evoked)
    sighted_evoked.append(evoked)
# mne.stats.permutation_cluster_test()
#%%
info = sighted_evoked[0].info
#%%
connectivity, ch_names = mne.channels.find_ch_connectivity(sighted_evoked[0].info, ch_type='eeg')
print(type(connectivity)) #it's a sparse matrix!
X = [np.array([a.data[[0]+list(range(2,48)),:].T for a in sighted_evoked]), np.array([a.data[[0]+list(range(2,48)),:].T for a in blind_evoked])]
cluster_stats = mne.stats.spatio_temporal_cluster_test( X=X,
                                                        threshold=10, connectivity=connectivity,
                                                        n_permutations=1000, tail=1
                                                        )
T_obs, clusters, p_values, _ = cluster_stats
good_cluster_inds = np.where(p_values < 0.05)[0]
print (good_cluster_inds)
# %%
condition_names = ['target', 'nontarget']


times = sighted_evoked[0].times * 1e3

# grand average as numpy arrray
grand_ave = np.array(X).mean(axis=1)

# get sensor positions via layout
pos = mne.find_layout(info).pos

# loop over significant clusters
for i_clu, clu_idx in enumerate(good_cluster_inds):
    # unpack cluster information, get unique indices
    time_inds, space_inds = np.squeeze(clusters[clu_idx])
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)

    # get topography for F stat
    f_map = T_obs[time_inds, ...].mean(axis=0)

    # get signals at significant sensors
    signals = grand_ave[..., ch_inds].mean(axis=-1)
    sig_times = times[time_inds]

    # create spatial mask
    mask = np.zeros((f_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True

    # initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))
    title = 'Cluster #{0}'.format(i_clu + 1)
    fig.suptitle(title, fontsize=14)

    # plot average test statistic and mark significant sensors
    image, _ = mne.viz.plot_topomap(f_map, pos, mask=mask, axes=ax_topo,
                            cmap='Reds', vmin=np.min, vmax=np.max)

    # advanced matplotlib for showing image with figure and colorbar
    # in one plot
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel('Averaged F-map ({:0.1f} - {:0.1f} ms)'.format(
        *sig_times[[0, -1]]
    ))

    # add new axis for time courses and plot time courses
    ax_signals = divider.append_axes('right', size='300%', pad=1.2)
    for signal, name in zip(signals, condition_names):
        ax_signals.plot(times, signal, label=name)
    
    # add information
    ax_signals.axvline(0, color='k', linestyle=':', label='stimulus onset')
    ax_signals.set_xlim([times[0], times[-1]])
    ax_signals.set_xlabel('time [ms]')
    ax_signals.set_ylabel('evoked magnetic fields [fT]')

    # plot significant time range
    ymin, ymax = ax_signals.get_ylim()
    ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                             color='orange', alpha=0.3)
    ax_signals.legend(loc='lower right')
    ax_signals.set_ylim(ymin, ymax)

    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=.05)
    plt.show()
# %%