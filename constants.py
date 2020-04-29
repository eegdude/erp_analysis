import mne
import pathlib

mne.set_log_level(verbose=0) # global mne log level, is set on import

# technical markers, present in envnt data
StartCycle = 777
EndCycle = 888
EndLearn = 888999
StartPlay = 999888
EndAll = 999
technical_markers = [StartCycle, EndCycle, EndLearn, EndAll, StartPlay]

# ECG analysis parameeters
Rpeak_event = 200
rejected_Rpeak_event = 404
Rpeaks_filename ='Rpeaks.npy'

# Folder parameters 
parents = list(pathlib.Path(__file__).parents)
pics_folder = pathlib.Path(parents[0] / 'pics')

BCI_type_gropued = True
cols = 	[[0, 6, 12, 18, 24, 30], [1, 7, 13, 19, 25, 31], [2, 8, 14, 20, 26, 32], [3, 9, 15, 21, 27, 33], [4, 10, 16, 22, 28, 34], [5, 11, 17, 23, 29, 35]]
rows = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29], [30, 31, 32, 33, 34, 35]]
groups = cols + rows

ch_names = ['F3', 'Fz', 'F4', 'FC5',
			'FC1', 'FC2', 'FC6', 'C3', 
			'Cz', 'C4', 'CP5', 'CP1', 'CP2', 'CP6',
			'P7', 'P3', 'Pz', 'P4', 'P8', 'PO7', 'POz', 'PO8', 
			'O1', 'O2', 'Fp1', 'A1']# +['stim']	
ch_types =	['eeg']*26
eog_channels = 'Fp1' # need to be string like "ch1,ch2" due to
						 # mne stuff
						 #	-> sumbit bug?
						 # None if using dedicated EOG channels
montage = 'standard_1005'

# Sampling frequency
fs = 500
ms_factor = 1000 // fs

events_offset = 0

# Epochs timing settings
epochs_tmin = -0.1
epochs_tmax = 0.8

# baseline settings
epochs_baseline = None
evoked_baseline = (-0.05, 0)
test_baseline = (0,0)

# Preprocessing settings: filtering

reference = 'average'
butter_filt = [0.1, 35]
butter_order = 4
notch = 50

# Preprocessing settings: ICA
n_ica_components = None # if None, uses maximum number of components

# plotting settings
uV_scaler = 1e6
legend_loc = (-5, 0)
plot_colors = {'target': 'red',
               'nontarget':'blue',
               'delta': 'black',

                'r_n':'red',
                'a_h':'green',
                'a_n':'blue',
                'r_h':'indigo',
                'a_pixel':'yellow',
			}
non_evoked_keys = ['quantiles']