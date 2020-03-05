import mne
import pathlib

mne.set_log_level(verbose=0)

StartCycle = 777
EndCycle = 888
EndLearn = 888999
StartPlay = 999888
EndAll = 999
technical_markers = [StartCycle, EndCycle, EndLearn, EndAll, StartPlay]

parents = list(pathlib.Path(__file__).parents)
pics_folder = pathlib.Path(parents[0] / 'pics')

BCI_type_gropued = True
rows = 	[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11],  # rows 6x7
        [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23],
        [24, 25, 26, 27, 28, 29], [30, 31, 32, 33, 34, 35],
        [36, 37, 38, 39, 40, 41]]

cols = [[0, 6, 12, 18, 24, 30, 36], [1, 7, 13, 19, 25, 31, 37],
        [2, 8, 14, 20, 26, 32, 38], [3, 9, 15, 21, 27, 33, 39],
        [4, 10, 16, 22, 28, 34, 40], [5, 11, 17, 23, 29, 35, 41]]
groups = cols + rows

ch_names = ["Fp1","ecg","A2","F3","Fz","F4",
            "C5","PO7","C3","Cz","C4","C6","CP1",
            "CP2","P3","Pz","P4","PO8","O1","Oz",
            "O2",'P5','P6','P7','P8','TP7','TP8']

ch_types =	['eeg', 'ecg'] + ['eeg']*25
eog_channels = 'Fp1' # need to be string like "ch1,ch2" due to
                         # mne stuff
                         #	-> sumbit bug?
                         # None if using dedicated EOG channels
montage = 'standard_1005'
fs = 500
ms_factor = 1000 // fs

events_offset = 0

epochs_tmin = -0.1
epochs_tmax = 0.8

epochs_baseline = None
evoked_baseline = (-0.05, 0)
test_baseline = (0,0)

butter_filt = [0.1, 35]
butter_order = 4
notch = 50

n_ica_components = None
mne_verbose_level = 1

uV_scaler = 1e6
legend_loc = (-5, 0)
plot_colors = {'target': 'red',
               'nontarget':'blue',
               'delta': 'black',

               'faces': 'red',
               'facesnoise':'green',
               'letters':'black',
               'noise':'blue',
               }