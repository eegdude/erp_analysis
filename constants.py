import mne
import pathlib

mne.set_log_level(verbose=0)

StartCycle = 777
EndCycle = 888
EndLearn = 888999
StartPlay = 999888
EndAll = 999
Rpeak_event = 200
rejected_Rpeak_event = 404
Rpeaks_filename ='Rpeaks.npy'
technical_markers = [StartCycle, EndCycle, EndLearn, EndAll, StartPlay]

parents = list(pathlib.Path(__file__).parents)
pics_folder = pathlib.Path(parents[0] / 'pics')
BCI_type_gropued = False

ch_names = ['Fp1', 'ecg', 'Fp2', 'F3', 'Fz', 'F4', 'FC5', 'FC3',
            'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'T7', 'C5', 'C3',
            'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'CP5', 'A1',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'A2', 'CP6', 'P7',
            'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7',
            'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']# +['stim']	
ch_types =	['eeg','ecg', 'eeg'] + ['eeg'] * 20 + ['eeg'] + \
            ['eeg'] * 5 + ['eeg'] + \
            ['eeg'] * 18 #+ ['stim']

eog_channels = 'Fp1,Fp2' # need to be string like "ch1,ch2" due to
                         # mne stuff
                         #	-> sumbit bug?
                         # None if using dedicated EOG channels
montage = 'standard_1005'
fs = 500
ms_factor = 1000 // fs

events_offset = 0.1

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

               'large': 'green',
               'small':'yellow',
               'brl_rsvp_2_2':'red',

                'left': "#aeb4a9",
                'right':"#e0c1b3",

                'sighted': 'orangered',
                'blind':'royalblue'}

non_evoked_keys = ['quantiles']