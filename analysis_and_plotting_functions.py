import pathlib
import pickle
import os

import numpy as np
import pandas as pd
from scipy import signal

import mne
from matplotlib import pyplot as plt

import constants
import iter_topography_fork

def select_color(key):
    if key in constants.plot_colors:
        return constants.plot_colors[key]
    else:
        return None

def plot_evoked_response(data: dict, 
                            p3peaks: dict={}, n1peaks: dict={},
                            p300_n1_aim_fill: bool=True, peakdot: bool=True,
                            fname: pathlib.Path=None,
                            title=None
                            ):
    """
        plot topographic EP maps

        Args:
            data (dict): dict with waveforms to plot ({key: mne.Evoked}) 
                        need array with all channels
            p3peaks (dict): p300 ampltudes and latencies
            n1peaks (dict): n1 ampltudes and latencies
            fname (str): file name to save picture
            peakdot (bool): if True plot P3 and N1 peak
            p300_n1_aim_fill (bool): if True, fill area around peaks
    """
    info = list(data.values())[0].info
    channels = [a for a, b in zip(info['ch_names'], info['chs']) if b['kind'] == 2] # select eeg channels => in 'kind' mne.utils._bunch.NamedInt
    channel_inds = [info['ch_names'].index(a) for a in channels] # list of channels indices in data array
    


    fig = plt.figure()
    fig.suptitle(title, x=0.1, y=0.9, fontsize=20)
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
            ax.plot(data[i].times, data[i].data[ch], color=select_color(i), label=i)

        ax.set_title(data[list(data.keys())[0]].ch_names[ch])

        if p3peaks:
            p3p = p3peaks[channels[idx]]
            ax.plot(p3p[0], p3p[1], 'o', color='black', zorder=228)

        if n1peaks:
            n1p = n1peaks[channels[idx]]
            ax.plot(n1p[0], n1p[1], 'o', color='black', zorder=228)


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

    legend = tpplt[0][0].legend(loc = constants.legend_loc, prop={'size': 10})
    if fname:
        plt.savefig(fname, dpi = 200)
    else:
        plt.show()
    plt.close()
    return fig

def plot_vectors_with_peaks(vector: np.ndarray,
                            p3_data: int=None, n1_data: int=None, n4_data: int=None):
    """UNTESTED
    
    Arguments:
        vector {np.ndarray} -- [description]
    
    Keyword Arguments:
        p3_data {int} -- [description] (default: {None})
        n1_data {int} -- [description] (default: {None})
        n4_data {int} -- [description] (default: {None})
    """                            
    zero = int(np.abs(constants.epochs_tmin)*constants.fs)
    xaxis = list(range(-1*zero, vector.shape[0]*constants.ms_factor - zero, constants.ms_factor))
    plt.plot(xaxis, vector)
    for peak in p3_data:
        plt.suptitle(peak['channel'])
        
        plt.plot(peak['latency']*constants.ms_factor - zero, 
                    vector[peak['latency']],
                    'o', linewidth=4, color='red')
#             fsection = [b for a, b in zip(['aim'].times, data['aim'].data[ch]*1.0e6) if a >= n1p[0]-0.015 and a < n1p[0]+0.015]
#             section = []
        plt.fill_between(x = np.arange(peak['left_base'], peak['right_base']) * constants.ms_factor - zero,
                        y1 = vector[peak['left_base']: peak['right_base']], color = 'green', alpha = 0.6)
        
        # zero_crossings = np.where(np.diff(np.signbit(vector)))[0]
        # print (zero_crossings)
        # left_base = np.max([a for a in zero_crossings - peak['latency'] if a <0]) + peak['latency']
        # right_base_array = [a for a in zero_crossings - peak['latency'] if a >=0]
        # if len(right_base_array)<1:
        #     right_base = np.min(right_base_array) + peak['latency']

#             right_base = zero_crossings[np.argmin([a for a in zero_crossings - peak['latency'] if a<0])]
#             print (left_base, right_base)
#             print ([a for a in zero_crossings - peak['latency'] if a >0])

        selection = vector[peak['left_base']:peak['right_base']]
        selection -= selection[0]
        linear_trend = np.arange(vector[peak['left_base']], vector[peak['right_base']], 1/(peak['right_base'] - peak['left_base']))
        print (selection, linear_trend)
        plt.plot (xaxis, np.r_[np.zeros(peak['left_base']), selection, np.zeros(len(vector)-peak['right_base'])])

        plt.text(peak['latency']*constants.ms_factor - zero, 
                vector[peak['latency']],
                f"P300 latency {peak['latency']*constants.ms_factor - zero} ms")

        plt.axhline (0, color = 'black')
        plt.axhline(np.median(vector[peak['left_base']: peak['right_base']]))
        plt.axvline(zero, color = 'black')
        plt.axvline(zero + 300, linestyle = '--')
    plt.show()
    
def get_peaks_from_evoked(evoked: mne.EvokedArray):
    """
        UNTESTED
        Detect P300 and N1 peak ampltudes ans latencies

        Args:
            evoke (mne.EvokedArray): average waveforms

        Returns:
            dict: p3peaks and n1peaks (for plotting) and peaks_dict (for statiscical analysis)
    """
    for ch_ind in (16,17,18,24,26,27):
        vector = evoked._data[ch_ind]
        peaks = signal.find_peaks(vector, prominence=0.1, distance = 500) 
        p3p_l = peaks[0]

        p3p_ind = [n for n, p in enumerate(p3p_l) if p*constants.ms_factor > 280 and 
                                p* constants.ms_factor < 600]
        if len(p3p_ind) > 1:
            print (f'more than one P300 peaks detectedd {peaks[0][p3p_ind]}')
        p3_data = [{'latency': peaks[0][ind],
                    'right_base': peaks[1]['right_bases'][ind],
                    'left_base': peaks[1]['left_bases'][ind],
                    'prominences': peaks[1]['prominences'][ind],
                    'channel':constants.ch_names[ch_ind]} for ind in p3p_ind]
        print (p3_data)
        plot_vectors_with_peaks(vector, p3_data = p3_data)


#         peaks_dict = {'File':os.path.basename(self.eeg_filename)}
#         for p in p3peaks.keys():
#             peaks_dict
#             peaks_dict['p3a_{}'.format(p) ] = p3peaks[p][1]
#             peaks_dict['p3i_{}'.format(p) ] = p3peaks[p][0]
#             if p in ["po7", "po8","o1","oz","o2"]:
#                 peaks_dict['n1a_{}'.format(p) ] = n1peaks[p][1]
#                 peaks_dict['n1i_{}'.format(p) ] = n1peaks[p][0]
    
#         return p3peaks, n1peaks, peaks_dict

def subset(ds, submarkup:pd.DataFrame):
    """Create Mne Evoked arrays for target, nontarget and delta EP
    
    Arguments:
        ds {dataset.DatasetReader} -- dataset
        submarkup {pd.DataFrame} -- subset of ds.markup, meeting any arbitrary condition
    
    Returns:
        dict -- Payload-style dict with target, nontarget and difference EPs for given subset
    """    
    
    subset_t = submarkup.loc[submarkup['is_target'] == 1]
    subset_nt = submarkup.loc[submarkup['is_target'] == 0]

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