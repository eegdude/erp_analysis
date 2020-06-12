# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 12:56:31 2018
@author: skjerns
Gist to save a mne.io.Raw object to an EDF file using pyEDFlib
(https://github.com/holgern/pyedflib)
Disclaimer:
    - Saving your data this way will result in slight 
      loss of precision (magnitude +-1e-09).
    - It is assumed that the data is presented in Volt (V), 
      it will be internally converted to microvolt
    - BDF or EDF+ is selected based on the filename extension
    - Annotations are lost in the process.
      Let me know if you need them, should be easy to add.
"""

import pyedflib # pip install pyedflib
from pyedflib import highlevel # new high-level interface
from pyedflib import FILETYPE_BDF, FILETYPE_BDFPLUS, FILETYPE_EDF, FILETYPE_EDFPLUS
from datetime import datetime, timezone, timedelta
import mne
import os
import traceback

def _stamp_to_dt(utc_stamp):
    """Convert timestamp to datetime object in Windows-friendly way."""
    # The min on windows is 86400
    stamp = [int(s) for s in utc_stamp]
    if len(stamp) == 1:  # In case there is no microseconds information
        stamp.append(0)
    return (datetime.fromtimestamp(0, tz=timezone.utc) +
            timedelta(0, stamp[0], stamp[1]))  # day, sec, μs


def write_npy_edf(array, info, fname, overwrite=False):
    """
    Saves the raw content of an MNE.io.Raw and its subclasses to
    a file using the EDF+/BDF filetype
    pyEDFlib is used to save the raw contents of the RawArray to disk
    Parameters
    ----------
    array : np.ndarray
        numpy array (nchan;nsamples)
    info : mne.Info
        An object containing 
    fname : string
        File name of the new dataset. This has to be a new filename
        unless data have been preloaded. Filenames should end with .edf
    overwrite : bool
        If True, the destination file (if it exists) will be overwritten.
        If False (default), an error will be raised if the file exists.
    """
    if not overwrite and os.path.exists(fname):
        raise OSError('File already exists. No overwrite.')
        
    # static settings
    has_annotations = False
    if os.path.splitext(fname)[-1] == '.edf':
        file_type = FILETYPE_EDFPLUS if has_annotations else FILETYPE_EDF
        dmin, dmax = -32768, 32767 
    else:
        file_type = FILETYPE_BDFPLUS if has_annotations else FILETYPE_BDF
        dmin, dmax = -8388608, 8388607
    
    print('saving to {}, filetype {}'.format(fname, file_type))
    sfreq = info['sfreq']
    date = _stamp_to_dt(info['meas_date'])
    date = date.strftime('%d %b %Y %H:%M:%S')
    first_sample = 0
    last_sample  = max(array.shape)

    # convert data
    channels = array
    
    # convert to microvolts to scale up precision
    channels *= 1e6

    # set conversion parameters
    n_channels = len(channels)
    print (n_channels)
    # create channel from this
    try:
        f = pyedflib.EdfWriter(fname,
                               n_channels=n_channels, 
                               file_type=file_type)
        
        channel_info = []
        
        ch_idx = range(n_channels)
        for i in ch_idx:
            ch_dict = {'label': info['ch_names'][i], 
                        'dimension': 'uV', 
                        'sample_rate': info['sfreq'],
                        'physical_min': channels.min(), 
                        'physical_max': channels.max(), 
                        'digital_min':  dmin, 
                        'digital_max':  dmax, 
                        'transducer': '', 
                        'prefilter': ''}
            channel_info.append(ch_dict)
        f.setTechnician('ak')

        f.setSignalHeaders(channel_info)
        f.setStartdatetime(date)
        print(channels.shape)
        f.writeSamples(channels)
    except Exception:
        print (traceback.format_exc())
        return False
    finally:
        f.close()
    return True