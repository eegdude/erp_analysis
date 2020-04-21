import pickle
import pathlib
import shutil
import csv
import ast
import os
import sys
import copy

import numpy as np
import pandas as pd

import mne
import heartpy as hp

import constants, folders
# mne.set_log_level(constants.mne_verbose_level)
class EegPreprocessing():
    """Process raw EEG and create epochs from single file
    """
    def __init__(self, reference_mode: str='original', ICA=False, 
                        fit_with_additional_lowpass=False):
        """
        Keyword Arguments:
            reference_mode {str, list} -- whether to re-reference data
                can be 'original' or empty list to preserve original reference,
                or any supported MNE reference argument. usually it's better to
                re-reference evoked data, unless it's needed for some continious
                eeg preprocessing. See `mne.set_eeg_reference` for details.
                (default: {'original'})
            ICA {bool} -- Whether to remove eye movements articact with ICA. 
                          ICA params are in constants.py (default: {False})
            fit_with_additional_lowpass {bool} -- whether to fit ICA with lowpass 
                filter at 1 Hz, and then to apply it to EEG filtered the way it's
                defined in constants.butter_filt (default: {False})
        """
        self.reference_mode = reference_mode
        self.ICA = ICA
        self.fit_with_additional_lowpass = fit_with_additional_lowpass

    def process_raw_eeg(self, raw: mne.io.RawArray) -> mne.io.RawArray:
        """Continious EEG processing pipeline.
            Re-refenence, optionally remove oculomotor artifacrs, 
            and bandpass filter data.
        
        Arguments:
            raw {mne.io.RawArray} -- raw EEG data
        
        Returns:
            mne.io.RawArray -- filtered and processed EEG data
        """
        raw = self.re_reference(raw)
        if self.ICA:
            raw = self.reject_eyes(raw, self.fit_with_additional_lowpass)
        else:
            raw = self.filter_eeg(raw)

        return raw
    
    def create_epochs(self, raw: mne.io.RawArray, events: np.ndarray) -> mne.epochs.Epochs:
        """Cut raw data by events
        
        Arguments:
            raw {mne.io.RawArray} -- raw EEG data
            events {np.ndarray} -- events in mne format [sample, any int, event_id]
        
        Returns:
            mne.epochs.Epochs -- [description]
        """
        epochs = mne.Epochs(raw, events,
                            tmin=constants.epochs_tmin,
                            tmax=constants.epochs_tmax,
                            baseline=constants.epochs_baseline,
                            preload=True,
                            verbose=0)   # generates a lot of uninformative output
        return epochs

    def reject_eyes(self, raw: mne.io.RawArray, fit_with_additional_lowpass:bool=False) -> mne.io.RawArray:
        """Use ICA for eye movement artifacts correction.
        
        Arguments:
            raw {mne.io.RawArray} -- Raw EEG with eye movement artifacts
            fit_with_additional_lowpass {bool} -- whether to fit ICA with lowpass 
                filter at 1 Hz, and then to apply it to EEG filtered the way it's
                defined in constants.butter_filt (default: {False})
        
        Returns:
            mne.io.RawArray -- Raw EEF with removed eye components
        """
        if fit_with_additional_lowpass:
            print ('using experimental lowpass filter before ica')
            raw_for_ica = copy.deepcopy(raw)
            raw_for_ica = self.filter_eeg(raw_for_ica, butter=[1, constants.butter_filt[1]], notch=None)
        else:
            raw_for_ica = raw
        
        raw = self.filter_eeg(raw)

        ica = mne.preprocessing.ICA(n_components=constants.n_ica_components, 
                                    random_state=42, verbose='ERROR')
        ica.fit(raw_for_ica, reject={})
        eog_inds, eog_scores = ica.find_bads_eog(raw_for_ica, ch_name=constants.eog_channels)
        print(f'Found {len(eog_inds)} eye-dependent components out of {constants.n_ica_components}: {eog_inds}')
        ica.exclude.extend(eog_inds)
        ica.apply(raw)
        return raw

    def re_reference(self, raw: mne.io.RawArray) -> mne.io.RawArray:
        """Optionally apply new EEG reference, in according to self.reference_mode.
        
        Arguments:
            raw {mne.io.RawArray} -- Continious EEG
        
        Returns:
            mne.io.RawArray -- Re-referenced EEG
        """
        if self.reference_mode == 'original':
            print('preserving original refrence')
            raw = raw.set_eeg_reference(ref_channels=[], projection=False)
        else:
            print(f'applying {self.reference_mode} refrence')
            raw = raw.set_eeg_reference(ref_channels=self.reference_mode, projection=False)
        return raw
    
    def filter_eeg(self, raw: mne.io.RawArray,
                         notch: int=constants.notch,
                         butter: list=constants.butter_filt
                         ) -> mne.io.RawArray:
        """Frequency filter for continious EEG
        
        Arguments:
            raw {mne.io.RawArray} -- Continious EEG
        
        Keyword Arguments:
            notch {[int, bool]} -- notch filter frequency 
                                   if False, this filter is not applied
                                   (default: {constants.notch})
            butter {[list, bool]} -- cutoff frequencies [l_freq, h_freq] 
                                    if False, this filter is not applied
                                    (default: {constants.butter_filt})
        
        Returns:
            mne.io.RawArray -- Filtered EEG data
        """
        picks = list(np.where(np.array(constants.ch_types) != 'misc')[0])
        if notch:   
            raw.notch_filter(notch, picks=picks, verbose='ERROR')
        if butter:	
            raw.filter(l_freq=min(butter), h_freq=max(butter), picks=picks, verbose='ERROR')
        return raw

class EpDatasetCreator():
    def __init__(self,
                markup_path: pathlib.Path,
                database_path: pathlib.Path,
                data_folder: pathlib.Path,
                ignore_users: list=[],
                reference_mode: str='original',
                ICA: bool=False,
                fit_with_additional_lowpass: bool=False,
                ecg_analysis:str=None):
        """Create preprocessed EEG dataset and store it on disc
        
        Arguments:
            markup_path {pathlib.Path} -- path to csv file with per-record markup
            database_path {pathlib.Path} -- folder where to put database
                                            All previous data in this folder
                                            will be deleted!
            data_folder {pathlib.Path} -- [description]
        
        Keyword Arguments:
            ignore_users {list} -- id of users to not include in db (default: {[]})
            reference_mode {str or list} -- EEG reference to be used in 
                EEGPreprocessing class constructor (default: {'original'})
            ICA {bool} -- Whether to remove eye movements articact with ICA. 
                ICA params are in constants.py (default: {False})
            fit_with_additional_lowpass {bool} -- whether to fit ICA with lowpass 
                filter at 1 Hz, and then to apply it to EEG filtered the way it's
                defined in constants.butter_filt (default: {False})
            ecg_analysis {str} -- optionally detect R peaks and find closest ERPs 
                for the analysis of cardiosynchronous activity.
                    manual - manual reviewing of raw eeg data with R-peaks
                    processed - read preprocessed R-peak data
                (default: {None})
        """
      
        self.ignore_users = ignore_users
        self.ecg_analysis = ecg_analysis

        self.epoch_counter_global = 0

        self.info_written_to_db = None
        self.global_markup = []

        self.database = database_path.resolve()
        self.create_pickled_database(self.database)

        self.preprocessing = EegPreprocessing(reference_mode=reference_mode, ICA=ICA,
            fit_with_additional_lowpass=fit_with_additional_lowpass)
        
        self.markup = self.read_csv_markup(markup_path)
        self.load_eeg_from_markup(data_folder)

    def _event_array_labeler(self, array: np.ndarray, target: int):
        """Receive events and current target, find out which events are target 
           and which are nontarget. Works both with row-column BCI and 
           classic oddball
            TODO: better naming
        Arguments:
            array {np.ndarray} -- event data
            target {int} -- current target
        
        Returns:
            [np.ndarray] -- array of ones and zeroes
        """

        if not constants.BCI_type_gropued:
            array[:,-1] = [1 if a == target else 0 for a in array[:,-1]]
            return array
        else:
            arr = []
            for event in array[:,-1]:
                active_stims = constants.groups[event%1000]
                if target in active_stims:
                    arr.append(True)
                else:
                    arr.append(False)
            array[:,-1] = arr
            return array
    
    def labeler(self, events: list, targets = None):
        """Receive events and current target, find out which events are target 
           and which are nontarget. Wraps around _event_array_labeler
        Arguments:
            array {np.ndarray} -- event data in mne format
            target {int} -- current target
        
        Returns:
            [np.ndarray] -- event data in mne format with information on 
                targets and non-targets
        """
        labeled_events = [self._event_array_labeler(events[a], targets[a])
                        for a in range(len(targets))]
        return labeled_events
    
    def transform_eeg_and_events_for_mne(self,
                                        eeg: np.ndarray,
                                        chunked_events: list, 
                                        ch_names: list, 
                                        ch_types: list, 
                                        fs: int):
        """Recieve numpy array with EEG, metadata  and events, return 
            mne.RawArray and mne-like events
        
        Arguments:
            eeg {np.ndarray} -- EEG array
            chunked_events {list} -- list of lists of events. Each item in list 
                corresponds to single input cycle
            ch_names {list} -- channel names - need to match constants.montage
            ch_types {list} -- channel types (see mne.Info docs)
            fs {int} -- sampling frequency
        
        Returns:
            mne.RawArray -- Mne array of data
            chunked_events -- BCI events for every input cycle
        """        
        self.info = mne.create_info(ch_names=ch_names,
                                    ch_types=ch_types,
                                     montage=constants.montage,
                                     sfreq=fs,
                                     verbose=0)
        raw = mne.io.RawArray(eeg[1:,:], self.info)

        # montage = mne.channels.make_standard_montage(kind = constants.montage)
        # raw.set_montage(montage, verbose=0)

        chunked_events = [np.c_[[np.where(eeg[0,:] >= a )[0][0] for a in chunk[:,0]],
                        chunk[:,1], chunk[:,1]] for	chunk in chunked_events] # convert chunks to mne-like format
        chunked_events = [chunk.astype(int) for chunk in chunked_events]
        return raw, chunked_events

    def open_single_folder_eeg(	self, files_dict: dict, targets=None,
                                ignore_events_id: list=[], events_offset=None):
        eeg, chunked_events = self.read_eeg_and_evt_files(	files_dict = files_dict,
                                                            ignore_events_id = ignore_events_id,
                                                            events_offset = events_offset)

        raw, chunked_events = self.transform_eeg_and_events_for_mne(eeg,
                                                        chunked_events,
                                                        fs = constants.fs,
                                                        ch_names = constants.ch_names,
                                                        ch_types = constants.ch_types
                                                                )
        self.record_length = max(raw._data.shape)/raw.info['sfreq']
        chunked_events = self.labeler(chunked_events, targets = targets)
        chunked_events = np.array(chunked_events)
        return raw, chunked_events

    def read_eeg_and_evt_files(	self,
                                files_dict:dict,
                                ignore_events_id:list=None,
                                events_offset:float=None):
        """Read files produced by BCI experiment
        
        Arguments:
            files_dict {[dict]} -- [description]
        
        Keyword Arguments:
            ignore_events_id {[list]} -- optional empty events with no stimuli
                 (default: {None})
            events_offset {[float]} -- what is the difference between event time 
                and actual stimuli presentation time - need to measure for every 
                setup (default: {None})
        Returns:
            np.array -- eeg array
            list -- list of lists of events. Each item in list 
                corresponds to single input cycle
        """
        if files_dict['eeg'].suffix == '.fif':
            return read_fif_files(files_dict)
        elif files_dict['eeg'].suffix == '.npy':
            eeg = np.load(files_dict['eeg']).T
        
        evt = np.load(files_dict['evt'])
        if events_offset:
            evt [:,0] -= events_offset

        fname = str(files_dict['evt'].name + '.csv',)
        # np.savetxt(X = evt.astype('int'), fname=fname, fmt='%i', delimiter=',')
        splitter_list = np.where(evt[:,1] == constants.StartCycle)[0]
        chunked_events = np.split(evt, splitter_list)
        if not chunked_events[0].shape[0]:      # !!!
            chunked_events = chunked_events[1:]	#first chunk is empty
        bool_mask = [[ True if int(a) not in constants.technical_markers + ignore_events_id
                        else False for a in b[:,1]]
                        for b in chunked_events]
        
        chunked_events = [chunked_events[n][b] for n,b in enumerate(bool_mask)]

        return eeg, chunked_events

    def get_files(self, folder: pathlib.Path,
                        mode: str = 'play', 
                        extension: str = 'npy') -> dict:
        print (folder)
        eeg_file = list(folder.glob(f'*data*{mode}*{extension}'))
        assert len(eeg_file)==1, \
            f'unable to find single eeg file of {mode} mode in folder {folder}'
        
        evt_file = list(folder.glob(f'*events*{mode}*{extension}'))
        assert len(eeg_file)==1, \
            f'unable to find single events file of {mode} mode in folder {folder}'
        
        fd_file = list(folder.glob(f'*photocell*{mode}*{extension}'))
        assert len(eeg_file)==1, \
            f'unable to find single photocell file of {mode} mode in folder {folder}'
        
        ans_file = list(folder.glob(f'*answers*txt'))
        if not len(ans_file) == 1:
            print(f'unable to find single answers file in folder {folder}')
            ans_file = [None]

        return dict(eeg = eeg_file[0], evt = evt_file[0], 
                    fd = fd_file[0], ans = ans_file[0])

    def read_csv_markup(self, markup_path: pathlib.Path) -> list:
        """Read per-record experimental markup
        
        Arguments:
            markup_path {pathlib.Path} -- [description]

        Returns:
            list -- per-record metadata
        """        
        with open(markup_path, encoding='utf-8') as m:
            reader = csv.DictReader(m)
            markup = [a for a in reader if a['user'] not in self.ignore_users]
        return markup

    def create_pickled_database(self, database_path: pathlib.Path = None) -> None:
        """Clear everything on database_path and create folder if nor exists yet
        
        Keyword Arguments:
            database_path {pathlib.Path} -- Where to put perprocessed EEG database
                (default: {None})
        """
        try:
            shutil.rmtree(database_path)
            print (f'Cleared data folder at {database_path}')
        except (NotADirectoryError, FileNotFoundError):
            pass
        pathlib.Path.mkdir(database_path)
        print (f'Created data folder at {database_path}')

    def write_epoch_to_db(self, epoch: np.ndarray) -> None:
        """save numpy array with perprocessed EEG data to database 
        
        Arguments:
            epoch {np.ndarray} -- Epoch array
        """
        with open (self.database / f'{self.epoch_counter_global}.pickle', 'wb') as fh:
            pickle.dump(epoch, fh)
    
    def write_info_to_db(self) -> None:
        """Save MNE info, corresponding to preprocessed epochs, to database
        """
        if self.info_written_to_db:
            pass
        else:
            print ('saving MNE info')
            with open (self.database / f'info.pickle', 'wb') as fh:
                pickle.dump(self.info, fh)
        self.info_written_to_db = True

    def save_epoch_markup(self) -> None:
        """Write global per-epoch dataset markup
        """
        with open (self.database / f'epochs_markup.csv', 'w', newline='', encoding='utf-8') as fh:
            fieldnames = list(self.global_markup[0].keys())
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for line in self.global_markup:
                writer.writerow(line)
    
    def read_Rpeak_events(self, folder):
        """Read previously saved R-peak events from folder
        
        Arguments:
            folder {[pathlib.Path]} -- path to raw EEG folder
        
        Returns:
            [np.ndarray] -- R-peak events
        """
        fp = folder / constants.Rpeaks_filename
        if os.path.exists(fp):
            Rpeak_events = np.load(fp)
        else:
            Rpeak_events = None
        return Rpeak_events

    def save_Rpeak_events(self, events, folder:pathlib.Path):
        """Save R-peak events to folder
        
        Arguments:
            events {[iterable]} -- R-peaks data
            folder {[pathlib.Path]} -- path to raw data folder
        """
        if len(events) != 0:
            print(f'saving R peaks array to {folder}')
            np.save(arr = events, file = folder / constants.Rpeaks_filename)

    def create_Rpeak_events(self, raw, record, hr_events:np.ndarray=[],
        ecg_channel:int=None):
        '''
        refactor
        '''
        if not ecg_channel:
            ecg_channel = constants.ch_names.index('ecg')
        if np.size(hr_events) == 0:
            peak_direction = int(record['ecg_r_peak_direction'])
        
            if peak_direction in [1, -1]:
                raw._data[ecg_channel,:] *= peak_direction

            raw = raw.filter(l_freq=5, h_freq=35, verbose='ERROR', picks=[1])
            filtered = raw._data[ecg_channel,:]
            filtered = hp.scale_data(filtered)
            raw = mne.io.RawArray(np.c_[raw._data[ecg_channel,:],
                                        filtered,
                                        raw._data[ecg_channel,:] - np.average(raw._data, axis=0)].T,
                                info = mne.create_info(['ecg', 'filt','car'],
                                sfreq = raw.info['sfreq'],
                                ch_types=['ecg', 'misc', 'ecg'])
                                )
            try:
                wd, m = hp.process(filtered, 500,
                    bpmmin=40, bpmmax=100, reject_segmentwise=True)
                pl = wd['peaklist']
                r = wd['removed_beats']
                pl = np.array([a for a in pl if a not in r])
                
                Rpeaks = np.c_[pl,
                    np.ones(pl.shape)*constants.Rpeak_event,
                    np.ones(pl.shape)*constants.Rpeak_event]
                rejected_Rpeaks = np.c_[r,
                    np.ones(r.shape)*constants.rejected_Rpeak_event,
                    np.ones(r.shape)*constants.rejected_Rpeak_event]
                hr_events = np.r_[Rpeaks, rejected_Rpeaks]
            except:
                hr_events = np.array([])
        
        raw.plot(events=hr_events if np.size(hr_events)>0 else None,
            event_color={constants.Rpeak_event:'green', constants.rejected_Rpeak_event:'red'},
            block=True, scalings={'ecg':1e-4, 'misc':1e2}, start=0, duration=5)
        for n_ann, ann in enumerate(raw.annotations):
            ann_samp = [int(ann['onset']*raw.info['sfreq']),
                        int((ann['onset']+ann['duration'])*raw.info['sfreq'])]
            
            Rpeaks_inside_annotation = []
            for n, ev in enumerate(hr_events):
                if (ev[0] >= min(ann_samp)) and (ev[0] <= max(ann_samp)):
                    Rpeaks_inside_annotation.append([ev, n])
            
            if Rpeaks_inside_annotation:
                for ev, n in Rpeaks_inside_annotation:
                    if ev[1] == constants.rejected_Rpeak_event:
                        ev[1], ev[2] = constants.Rpeak_event, constants.Rpeak_event
                    elif ev[1] == constants.Rpeak_event:
                        ev[1], ev[2] = constants.rejected_Rpeak_event, constants.rejected_Rpeak_event
                    hr_events[n] = ev
            else:
                event = ann_samp[0] + np.argmax(raw._data[0,ann_samp[0]:ann_samp[1]])
                if np.size(hr_events)>0:
                    hr_events = np.r_[hr_events, [[event, constants.Rpeak_event, constants.Rpeak_event]]]
                else:
                    hr_events = np.array([[event, constants.Rpeak_event, constants.Rpeak_event]])

        raw.annotations.crop(0,0) # clear annotations object
        
        if np.size(hr_events) > 0:
            record_length = max(raw._data.shape)/raw.info['sfreq']
            hr_events = hr_events[np.where(hr_events[:,-1] == constants.Rpeak_event)]
            print(f'detected {max(np.shape(hr_events)):.0f} R-peaks in {record_length:.0f} seconds of data')

        if self.ecg_analysis == 'manual':
            fine = input('fine? y/n/discard    ')
            if fine == 'y':
                return hr_events

            elif fine == 'n':
                return self.create_Rpeak_events(raw, record, hr_events)
            
            elif fine == 'discard':
                return np.array([])
            
            else:
                return np.array([])

    def ecg_analysis_routine(self, raw, folder, record, events):
        if self.ecg_analysis == 'manual':
            hr_events = self.read_Rpeak_events(folder)
            if hr_events is None:
                hr_events = self.create_Rpeak_events(raw, record, events)
                self.save_Rpeak_events(hr_events, folder)
                return 'contiune'
            else:
                print(f'read {max(np.shape(hr_events)):.0f} R-peaks in {self.record_length:.0f} seconds of data')

        elif self.ecg_analysis == 'processed':
            hr_events = self.read_Rpeak_events(folder)
        elif self.ecg_analysis == 'automatic':
            raise NotImplementedError
        elif not self.ecg_analysis:
            hr_events = None
            pass
        if np.size(hr_events) == 0:
            hr_events = None
        
        return hr_events
                
    def load_eeg_from_markup(self, data_folder: pathlib.Path) -> None:
        for record in self.markup:
            self.epoch_counter_record = 0

            targets = ast.literal_eval(record['targets'])
            fingers = ast.literal_eval(record['fingers'])

            epochs_targets = []
            epochs_ids = []
            epochs_fingers = []
            sessions_id = []

            folder = data_folder / record['user'] / record['folder']
            files_dict = self.get_files(folder)
            raw, chunked_events = self.open_single_folder_eeg(files_dict,
                ignore_events_id=ast.literal_eval(record['ignore_events_id']),
                targets=targets,
                events_offset=constants.events_offset)
            events = np.vstack(chunked_events)
            raw = self.preprocessing.process_raw_eeg(raw)

            hr_events = self.ecg_analysis_routine(raw, folder, record, events)

            epochs = self.preprocessing.create_epochs(raw, events)
            assert len(chunked_events) == len(targets), \
                'number of events is not equal to number of targets'
            session_id=0
            for chunk, target in zip(chunked_events, targets):
                epochs_chunk_id = 0
                for event in chunk:
                    epochs_targets.append(target)
                    epochs_ids.append(epochs_chunk_id)
                    epochs_fingers.append(fingers[event[1]])
                    sessions_id.append(session_id)
                    epochs_chunk_id += 1
                session_id += 1

            assert  len(epochs) == len(events) and \
                    len(epochs_targets) == len(epochs_fingers) and \
                    len(epochs) == len(epochs_fingers) and \
                    len(epochs_ids) == len(epochs), \
                    'something is f-d up, fix it asap'

            # if hr_events is not None:
            #     plot_events = np.r_[hr_events, events]
            # else:
            #     plot_events = events
            # raw.plot(events=plot_events,
            #     event_color={constants.Rpeak_event:'green', constants.rejected_Rpeak_event:'red', -1:'#fae5ac'},
            #     block=True, scalings={'ecg':1e-4, 'misc':1e2}, start=0, duration=5)

            for epoch, event, target, finger, session_id in zip(epochs, events, epochs_targets, epochs_fingers, sessions_id):
                if hr_events is not None:
                    r_deltas = hr_events[:,0] - event[0]
                    ms_before_r = np.min(r_deltas[r_deltas>=0])*constants.ms_factor
                    ms_after_r = abs(np.max(r_deltas[r_deltas<0])*constants.ms_factor)
                else:
                    ms_after_r = None
                    ms_before_r = None
                epoch_markup_line = {
                                    'id': self.epoch_counter_global,
                                    'finger': fingers[event[-2]],
                                    'target': target,
                                    'event': event[-2],
                                    'is_target': event[-1],
                                    'epoch_id': epochs_ids[self.epoch_counter_record],
                                    'session_id': session_id,
                                    'ms_after_r':ms_after_r,
                                    'ms_before_r':ms_before_r
                                    }
                epoch_markup_line.update(record)
                self.global_markup.append(epoch_markup_line)
                self.write_info_to_db()
                self.write_epoch_to_db(epoch)
                self.epoch_counter_record += 1
                self.epoch_counter_global += 1
            self.save_epoch_markup()

class DatasetReader():
    def __init__(self, data_path: str, preload=False) -> None:
        """Create DatasetReader object and load dataset.
        
        Arguments:
            data_path {str} -- path to processed EEG database
        
        Keyword Arguments:
            preload {bool} -- if True, load dataset into memory.
                if False, each epoch is read from disc every time. (default: {False})
        """
        self.data_path = pathlib.Path(data_path).resolve()
        with open(self.data_path / 'info.pickle', 'rb') as p:
            self.info = pickle.load(p)

        self.markup = pd.read_csv(self.data_path / 'epochs_markup.csv',
                            dtype = {'id':int,
                                    'finger':int,
                                    'event':int,
                                    'target':int,
                                    'is_target':int,
                                    'epochs_chunk_id':int,
                                    'folder':str,
                                    'ecg_r_peak_up':int,
                                    'reading_finger':object,
                                    'blind':int,
                                    'user':str,
                                    'ms_after_r':float,
                                    'ms_before_r':float
                                    })
        self.db_size = self.markup['id'].shape[0]
        self.percentage_read = 0

        if preload:
            self.load_epoch = self.load_from_memory
            self.global_in_memory_database = {id:self.load_pickle(id) for id in self.markup['id']}
        else:
            self.load_epoch = self.load_pickle

    def load_from_memory(self, id: int) -> np.ndarray:
        """return epoch from memory
        
        Arguments:
            id {int} -- epoch ID from ds.markup
        
        Returns:
            np.ndarray -- Epoch array
        """
        return self.global_in_memory_database[id]

    def load_pickle(self, id: int , verbose=True) -> np.ndarray:
        """load pickled epoch to memory
        
        Arguments:
            id {int} -- ID of epochs from ds.markup
        
        Keyword Arguments:
            verbose {bool} -- if True, print progress to console. (default: {True})
        
        Returns:
            np.ndarray -- pickled epoch
        TODO:
            1. Currently, the percentage counter works correctly only when loading whole dataset.
            2. Migrate to better database solution
        """
        with open(self.data_path / f'{id}.pickle', 'rb') as p:
            epoch = pickle.load(p)
            epoch *= constants.uV_scaler
        if verbose:
            if id/self.db_size*100 - self.percentage_read >= 1:
                self.percentage_read = int(id/self.db_size*100)
                print (f"\r{self.percentage_read + 1} percent complete", end='')
        return epoch

    def create_binary_events_from_subset(self, subset: pd.DataFrame):
        """recieve the subset of ds.markup, return mne-like events array with
           ones and zeroes for creation of Epochs object. No time points are present
           in the array

        Arguments:
            subset {[pd.DataFrame]} -- subset of ds.markup
        
        Returns:
            np.ndarray -- mne events array
        """
        return np.c_[list(range(len(subset['is_target']))), subset['is_target'], subset['is_target']]

    def create_mne_epochs_from_subset(self, subset: pd.DataFrame, reference=None) -> mne.EpochsArray: 
        """Receive set of epochs, return mne.Epochs object
        Arguments:
            subset {pd.DataFrame} -- subset of self.markup
        
        Keyword Arguments:
            reference {list} -- EEG reference (default: {None})
        
        Returns:
            mne.EpochArray -- Epochs object
        TODO:
            add rejection parameters
        """
        epochs_subset = [self.load_epoch(id) for id in subset['id']]
        epochs_subset = mne.EpochsArray(data=epochs_subset,
                                        info=self.info,
                                        tmin=constants.epochs_tmin,
                                        events=self.create_binary_events_from_subset(subset))
        if reference:
            epochs_subset = epochs_subset.set_eeg_reference(reference)
        return epochs_subset
    
    def create_mne_evoked_from_subset(self, subset: pd.DataFrame,
                                            reject_max_delta:float=1000,
                                            reference:list=None) -> mne.EvokedArray: 
        """Receive subset of ds.markup, return Evoked data without loading whole set 
           into memory
        
        Arguments:
            subset {pd.DataFrame} -- subset of self.markup
        
        Keyword Arguments:
            reject_max_delta {float} -- reject epochs with peak-to-peak amplitude
                larger then this number (default: {1000})
            reference {list} -- EEG reference (default: {None})
        
        Returns:
            mne.EvokedArray -- Evoked potential
        """
        data = self.load_epoch(subset['id'].reset_index(drop=True)[0])
        cc = 1
        for id in subset['id'].reset_index(drop=True)[1:]:
            ep = self.load_epoch(id)
            if reject_max_delta is not None:
                if np.mean(np.max(ep, axis=1) - np.min(ep, axis=1)) < reject_max_delta:
                    data += ep
                    cc += 1
        data/=cc
        evoked = mne.EvokedArray(info=self.info,
                                data=data,
                                tmin=constants.epochs_tmin,
                                nave=cc)
        if reference:
            evoked = evoked.set_eeg_reference(reference)
        return evoked

def reject_outliers(data:np.ndarray, m=1.5):
    """remove outliers from data array
    
    Arguments:
        data {np.ndarray} -- 1d-data array
    
    Keyword Arguments:
        m {float} -- how many standard deviation is considered as outlier
            (default: {1.5})
    
    Returns:
        np.ndarray -- array without outliers
    """
    return data[abs(data - np.mean(data)) < m * np.std(data)]

if __name__ == "__main__":
    # Create dataset from raw data\
    
    epd = EpDatasetCreator( markup_path=folders.markup_path,
                            database_path=folders.database_path,
                            data_folder=folders.raw_data_folder,
                            reference_mode='average',
                            ICA=True,
                            fit_with_additional_lowpass=True,
                            ecg_analysis='processed',
                            )