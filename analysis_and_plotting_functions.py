import pathlib
import pickle
import os
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats

import mne
from matplotlib import pyplot as plt

import constants
import iter_topography_fork

def _select_color(key):
    """If key on constants.plot_colors, return correct color, otherwise 
        return None
    
    Arguments:
        key {str} -- name of variable to search in constants.plot_colors
    
    Returns:
        [str] -- Color name as defined in matplotlib colormap or hex representation of color.
    """

    if key in constants.plot_colors:
        return constants.plot_colors[key]
    else:
        return None

def plot_evoked_response(   evoked_dict: dict, 
                            p3peaks: dict={}, n1peaks: dict={},
                            p300_n1_aim_fill: bool=True, peakdot: bool=True,
                            vlines: list=None,
                            fname: pathlib.Path=None,
                            alpha_dict:dict={},
                            title=None):
    """Plot multiple lines on one mne.viz.plot_topo-like figure
    
    Arguments:
        evoked_dict {dict} -- dict with waveforms to plot ({key: mne.Evoked}) 
    
    Keyword Arguments:
        p3peaks {dict} -- p300 ampltudes and latencies (default: {{}})
        n1peaks {dict} -- n1 ampltudes and latencies (default: {{}})
        p300_n1_aim_fill {bool} -- [description] (default: {True})
        peakdot {bool} -- if True plot P3 and N1 peak (default: {True})
        vlines {list} -- vertical lines coordinates (default: {None})
        fname {pathlib.Path} -- file name to save picture (default: {None})
        alpha_dict {dict} -- control alpha for all lines in evoked_dict.
            If key is omitted, alpha equals 1 (default: {{}})
        title {[type]} -- figure title (default: {None})
    """
    
    non_evoked_data = {key: value for (key, value) in evoked_dict.items() if key in constants.non_evoked_keys}
    data = {key: value for (key, value) in evoked_dict.items() if key not in constants.non_evoked_keys}
    alpha_dict = {a:1 if a not in alpha_dict else alpha_dict[a] for a in data.keys()}
    info = list(data.values())[0].info
    channels = [a for a, b in zip(info['ch_names'], info['chs']) if b['kind'] == 2] # select eeg channels => in 'kind' mne.utils._bunch.NamedInt
    channel_inds = [info['ch_names'].index(a) for a in channels] # list of channels indices in data array

    fig = plt.figure()
    tpplt = [a for a in iter_topography_fork._iter_topography(info, layout=None, on_pick=None, fig=fig, layout_scale=0.945,
                                                    fig_facecolor='white', axis_facecolor='white', axis_spinecolor='white',
                                                    hide_xticklabels=True, hide_yticklabels=False, y_scale=3)]
    ylim_top = np.max([np.max(data[i].data[ch]) for ch in channel_inds for i in data.keys()])*1.2
    ylim_bottom = np.min([np.min(data[i].data[ch]) for ch in channel_inds for i in data.keys()])*1.2

    for n, ch in enumerate(channel_inds):
        ax = tpplt[n][0]
        idx = tpplt[n][1]
        ax.set_ylim(top=ylim_top, bottom=ylim_bottom)

        [ax.axvline(vl, color = 'black', linewidth=0.5, linestyle='--') for vl in [0, 0.1, 0.3]]
        ax.axhline(0, color = 'black', linewidth=0.5)

        for i in data.keys():
            ax.plot(data[i].times, data[i].data[ch], color=_select_color(i), label=i, alpha=alpha_dict[i])

        ax.set_title(data[list(data.keys())[0]].ch_names[ch])

        if p3peaks:
            p3p = p3peaks[channels[idx]]
            ax.plot(p3p[0], p3p[1], 'o', color='black', zorder=228)

        if n1peaks:
            n1p = n1peaks[channels[idx]]
            ax.plot(n1p[0], n1p[1], 'o', color='black', zorder=228)

        if non_evoked_data:
            if 'quantiles' in list(non_evoked_data.keys()):
                for i in data.keys():
                    if i in list(non_evoked_data['quantiles'].keys()):
                        ax.fill_between(data[i].times,
                            non_evoked_data['quantiles'][i]['upper'].data[ch],
                            non_evoked_data['quantiles'][i]['lower'].data[ch],
                            color=_select_color(i),
                            alpha=0.2)

        if p300_n1_aim_fill:
            if p3peaks:
                fsection = [b for a, b in zip(data['aim'].times, data['aim'].data[ch]) if a >= p3p[0] - 0.05 and a < p3p[0] + 0.05]
                section = [a for a in data['aim'].times if a >= p3p[0] - 0.05 and a < p3p[0] + 0.05]
                ax.fill_between(section, fsection, color='y', alpha=0.6)
            if n1peaks:
                if data[data.keys()[0]].ch_names[ch] in ['oz', 'o1', 'o2']:
                    ax.plot(n1p[0], n1p[1], 'o', color='black', zorder=228)
                    fsection = [b for a, b in zip(data['aim'].times, data['aim'].data[ch]) if a >= n1p[0] - 0.015 and a < n1p[0] + 0.015]
                    section = [a for a in data['aim'].times if a >= n1p[0] - 0.015 and a < n1p[0] + 0.015]
                    ax.fill_between(section, fsection, color = 'green', alpha=0.6)
    if vlines: # markup for stuff about EPs
        for vl in vlines:
            # print(vl)
            tpplt[vl[0]][0].axvline(vl[1]/500.0, alpha=0.2)

    # legend = tpplt[0][0].legend(loc = 'upper left', bbox_to_anchor=[-10, -10] prop={'size': 10})
    lhl = tpplt[n][0].get_legend_handles_labels()
    fig.suptitle(title, x=0, y= 0, ha='left', va='top')
    fig.legend(lhl[0], lhl[1], loc = 'upper left')

    
    if fname:
        plt.savefig(fname, dpi = 200)
    else:
        plt.show()
    plt.close()
    return fig

def _get_evoked_and_quantiles(ds, subset:pd.DataFrame,
                              upper:float=0.75, lower:float=0.25, method:str='mean'):
    """Return evoked potentials, quantiles and standard deviation from a subset of the dataset.
        This function has to load all subset into memory at once.
    
    Arguments:
        ds {DatasetReader} -- dataset object
        subset {pd.DataFrame} -- subset of ds.markup, meeting any arbitrary condition
    
    Keyword Arguments:
        upper {float} -- upper quantile (default: {0.75})
        lower {float} -- lower quantile (default: {0.25})
        method {str | callable} -- averaging method (default: {'mean'})
    
    Returns:
        [dict] -- Dictionary with evoked data and quantiles
    """

    epochs = ds.create_mne_epochs_from_subset(subset).apply_baseline(constants.evoked_baseline)
    evoked = epochs.average(method=method, picks=range(epochs.info['nchan'])) # bug in mne.epochs.average??
    q_lower = np.quantile(epochs.get_data(), lower, axis=0)
    q_upper = np.quantile(epochs.get_data(), upper, axis=0)
    std = np.std(epochs.get_data(), axis=0)

    payload = {'evoked': evoked,
            'lower': mne.EvokedArray(data=q_lower, info=ds.info, tmin=constants.epochs_tmin),
            'upper': mne.EvokedArray(data=q_upper, info=ds.info, tmin=constants.epochs_tmin),
            'std': mne.EvokedArray(data=std, info=ds.info, tmin=constants.epochs_tmin),
            }
    return payload

def subset(ds, submarkup:pd.DataFrame, drop_channels:list=['ecg', 'A1', 'A2'], quantiles:list=None, method='mean') -> dict:

    """Create Mne Evoked arrays for target, nontarget and delta EP

    Arguments:
        ds {dataset.DatasetReader} -- dataset
        submarkup {pd.DataFrame} -- subset of ds.markup, meeting any arbitrary condition
        drop_channels {list} -- channels to remove from evoked array. (default: {['ecg', 'A1', 'A2']})
                                use empty list to use all channels
        quantiles{list} -- Upper and lower quantiles (default: {None}). If not None, uses memory-greedy
            algorithm for getting average waveforms
        method{str|callable} -- method of combining data
    Returns:
        dict -- Payload-style dict with target, nontarget and difference EPs for given subset
    
    TODO: std calculation is memory-greedy, probably should rewrite it with Welford's algorithm
    """

    subset_t = submarkup.loc[submarkup['is_target'] == 1]
    subset_nt = submarkup.loc[submarkup['is_target'] == 0]

    if not quantiles:
        target = ds.create_mne_evoked_from_subset(subset_t).apply_baseline(constants.evoked_baseline)
        nontarget = ds.create_mne_evoked_from_subset(subset_nt).apply_baseline(constants.evoked_baseline)
        delta = mne.EvokedArray(info=ds.info,
                                data=target._data - nontarget._data,
                                tmin=constants.epochs_tmin,
                                nave=target.nave
                                )
        payload = {
                    'target': target.drop_channels(drop_channels),
                    'nontarget': nontarget.drop_channels(drop_channels),
                    'delta': delta.drop_channels(drop_channels)
                    }

    elif quantiles:
        target = _get_evoked_and_quantiles(ds, subset_t, lower=quantiles[0], upper=quantiles[1], method=method)
        nontarget = _get_evoked_and_quantiles(ds, subset_nt, lower=quantiles[0], upper=quantiles[1], method=method)
        nontarget2 = _get_evoked_and_quantiles(ds, subset_nt, lower=quantiles[0], upper=quantiles[1], method=method)
        delta = mne.EvokedArray(info=ds.info,
                                data=target["evoked"]._data - nontarget["evoked"]._data,
                                tmin=constants.epochs_tmin,
                                nave=target["evoked"].nave
                                )
        payload = {
                    "target": target["evoked"].drop_channels(drop_channels),
                    "nontarget": nontarget["evoked"].drop_channels(drop_channels),
                    "delta": delta.drop_channels(drop_channels),
                    "quantiles": {
                            "target": {
                                    "lower": target["lower"].drop_channels(drop_channels),
                                    "upper": target["upper"].drop_channels(drop_channels),
                                    "std": target["std"].drop_channels(drop_channels),
                                    },
                            "nontarget": {
                                        "lower": nontarget["lower"].drop_channels(drop_channels),
                                        "upper": nontarget["upper"].drop_channels(drop_channels),
                                        "std": nontarget["std"].drop_channels(drop_channels),
                                        }
                                }
                    }
    return payload

def cluster_and_plot(X:list,
                    info:mne.Info, 
                    times:Iterable, 
                    condition_names:list,
                    threshold:float=None,
                    n_permutations:int=10000,
                    tail:int=1,
                    step_down_p:float=0,
                    n_jobs:int=1,
                    cutoff_pval:float=0.05,
                    plot_range:list=None,
                    spatial_exclude=None,
                    stat_fun:Callable=None):
    """Run cluster-based permutation test, plot clusters and print p-values of 
       significant ones.
    
    Arguments:
        X {list} -- list of numpy arrays to compare, [chanels x time x samples]
        info {mne.Info} -- Info object, corresponding to evoked data
        times {Iterable} -- X axis for cluster plot
        condition_names {list} -- Conditions to compare. Must be the same length as X
    
    Keyword Arguments:
        See mne.stats.spatio_temporal_cluster_test docs for details

        threshold {float} -- (default: {10})
        n_permutations {int} -- (default: {10000})
        tail {int} -- (default: {1})
        step_down_p {float} -- (default: {0})
        n_jobs {int} -- (default: {1})
        cutoff_pval {float} -- (default: {0.05})
        plot_range {list} -- (default: {None})
        spatial_exclude {[type]} -- (default: {None})
        stat_fun {Callable} -- (default: {None})
    
    Returns:
        [list] -- outputs of mne.stats.spatio_temporal_cluster_test
    """
    
    print (stat_fun)
    connectivity, ch_names = mne.channels.find_ch_connectivity(info, ch_type='eeg')
    cluster_stats = mne.stats.spatio_temporal_cluster_test( X=X,
                                                            threshold=threshold,
                                                            connectivity=connectivity,
                                                            n_permutations=n_permutations,
                                                            tail=tail,
                                                            n_jobs=n_jobs,
                                                            step_down_p=step_down_p,
                                                            stat_fun=stat_fun,
                                                            check_disjoint=True,
                                                            spatial_exclude=spatial_exclude,
                                                            seed=282,
                                                            t_power=1
                                                            )
    T_obs, clusters, p_values, _ = cluster_stats
    good_cluster_inds = np.where(p_values < cutoff_pval)[0]
    if len(p_values[good_cluster_inds]):
        print (p_values[good_cluster_inds])
    else:
        print ("No significant clusters found")
    # grand average as numpy arrray
    grand_ave = np.array([np.array(a).mean(axis=0) for a in X])
    print (np.array(X).shape, grand_ave.shape)
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
        print (f'Mean F-score for cluster {(np.mean(f_map))}')
        # get signals at significant sensors
        signals = grand_ave[..., ch_inds].mean(axis=-1)
        if plot_range:
            q_upper = np.quantile(grand_ave[..., ch_inds], plot_range[0], axis=-1)
            q_lower = np.quantile(grand_ave[..., ch_inds], plot_range[1], axis=-1)
        else:
            q_upper = np.zeros(signals.shape)
            q_lower = np.zeros(signals.shape)

        sig_times = times[time_inds]

        # create spatial mask
        mask = np.zeros((f_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True
        cluster_channels = [info['ch_names'][n] for n, a in enumerate([a[0] for a in mask]) if a]
        print (f'Cluster channels {cluster_channels}')
        # initialize figure
        fig, axs = plt.subplots(1, 2, figsize=(12,4))
        fig.suptitle('Cluster #{0}'.format(i_clu + 1), fontsize=16, x=0.4, y=0.9)

        # plot average test statistic and mark significant sensors
        image, _ = mne.viz.plot_topomap(f_map, pos, mask=mask, axes=axs[0],
                            vmin=np.min, vmax=np.max, show=False, 
                            names=info['ch_names'][0:1] + info['ch_names'][2:], show_names=False,
                            extrapolate='head', contours=10, outlines='skirt', mask_params={'markersize':4, 'markerfacecolor':'white'})
        fig.colorbar(image, ax=axs[0], shrink=0.6)

        axs[0].set_xlabel('Averaged F-map ({:0.1f} - {:0.1f} ms)'.format(*sig_times[[0, -1]]))
        # add new axis for time courses and plot time courses
        # ax_signals = divider.append_axes('right', size='300%', pad=1.2)
        for signal, name in zip(signals, condition_names):
            if name in list(constants.plot_colors.keys()):
                color = constants.plot_colors[name]
            else:
                color = None
            axs[1].plot(times, signal, label=name, color=color)
        
        # add information
        axs[1].axvline(0, color='k', linestyle=':', label='stimulus onset')
        axs[1].set_xlim([times[0], times[-1]])
        axs[1].set_xlabel('time [ms]')
        axs[1].set_ylabel('uV')

        # plot significant time range
        ymin, ymax = axs[1].get_ylim()
        axs[1].fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                                color='orange', alpha=0.3)
        axs[1].legend(loc='lower right')
        axs[1].set_ylim(ymin, ymax)

        # clean up viz
        mne.viz.tight_layout(fig=fig)
        fig.subplots_adjust(bottom=.05)
        plt.show()
    return cluster_stats

def assumptions_bonferroni_X(X):
    """
    WORK IN PROGRESS
    """

    bf_correction = X[0].shape[1]*X[0].shape[2]
    bf_correction = 1


    for channel in range(X[0].shape[1]):
        for sample in range(X[0].shape[-1]):

            c = X[0][:, channel, sample]
            d = X[1][:, channel, sample]

            ltest= (stats.levene(c, d))
            shapiro1= (stats.shapiro(d))
            shapiro2= (stats.shapiro(c))

            # if (ltest[1]<0.05/bf_correction):#  and \
            if (shapiro1[1]<0.05/bf_correction):# and \
            #     (shapiro2[1]<0.05/bf_correction):

                yield (channel, sample)

def clusterable_mwtest(x, y):
    """
    WORK IN PROGRESS
    """
    return np.array([stats.mannwhitneyu(x[:,a], y[:,a])[0] for a in range(x.shape[1])])

def clusterable_kwtest(x, y):
    """
    WORK IN PROGRESS
    """
    return np.array([stats.kruskal(x[:,a], y[:,a])[0] for a in range(x.shape[1])])
