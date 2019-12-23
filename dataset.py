import constants, folders
import pickle
import pathlib
import shutil
import csv
import ast
import sys
import copy
import numpy as np
import pandas as pd
import mne

# mne.set_log_level(constants.mne_verbose_level)
class EegPreprocessing():
    """Process raw EEG and create epochs from single file
    """
    def __init__(self, reference_mode: str='original', ICA=False, 
                        fit_with_additional_lowpass=False):
        """
        Keyword Arguments:
            reference_mode {str, list} -- whether to re-reference data
                                    can be 'original' or empty list to preserve 
                                    original reference, or any supported MNE 
                                    reference argument. usually it's better to 
                                    re-reference evoked data, unless it's needed
                                    for some continious eeg preprocessing
                                    (default: {'original'})
            ICA {bool} -- Whether to remove eye movements articact with ICA. 
                          ICA params are in constants.py (default: {False})
        """
        self.reference_mode = reference_mode
        self.ICA = ICA
        self.fit_with_additional_lowpass = fit_with_additional_lowpass

    def process_raw_eeg(self, raw: mne.io.RawArray) -> mne.io.RawArray:
        """Continious EEG processing pipeline
        
        Arguments:
            raw {mne.io.RawArray} -- raw EEG data
        
        Returns:
            mne.io.RawArray -- filtered and processed EEG data
        """
        raw = self.re_reference(raw)
        # raw = self.filter_eeg(raw)
        if self.ICA:
            raw = self.reject_eyes(raw, self.fit_with_additional_lowpass)
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
                            verbose=0   # generates a lot of uninformative output
                            )
        return epochs

    def reject_eyes(self, raw: mne.io.RawArray, fit_with_additional_lowpass:bool=False) -> mne.io.RawArray:
        """Use ICA for eye movement artifacts correction.
        
        Arguments:
            raw {mne.io.RawArray} -- Raw EEG with eye movement artifacts
            fit_with_additional_lowpass {bool} -- experimental
        
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
            raw = raw.set_eeg_reference(ref_channels=[], projection=False)
        else:
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
                database_path: pathlib.Path=pathlib.Path('BrailleBCIDB'),
                data_folder: pathlib.Path=folders.data_folder,
                ignore_users: list=[],
                reference_mode: str='original',
                ICA: bool=False,
                fit_with_additional_lowpass: bool=False):
        """Read raw data from BCI and create database with per-epoch markup
        
        Arguments:
            markup_path {pathlib.Path} -- path to csv file with per-record markup
        
        Keyword Arguments:
            database_path {pathlib.Path} -- folder where to put database
                                            All previous data in this folder
                                            will be deleted!
                                            (default: {pathlib.Path('BrailleBCIDB')})
            ignore_users {list} -- id of users to not include in db (default: {[]})
            reference_mode {str or list} -- EEG reference to be used in 
                                            EEGPreprocessing class constructor
            ICA {bool} -- Whether to remove eye movements articact with ICA. 
                          ICA params are in constants.py (default: {False})
        """

        self.ignore_users = ignore_users

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
        array[:,-1] = [1 if a == target else 0 for a in array[:,-1]]
        return array
    
    def labeler(self, events: list, targets = None):
        labeled_events = [self._event_array_labeler(events[a], targets[a])
                        for a in range(len(targets))]
        return labeled_events
    
    def transform_eeg_and_events_for_mne(self,
                                        eeg: np.ndarray,
                                        chunked_events: list, 
                                        ch_names: list, 
                                        ch_types: list, 
                                        fs: int):
        self.info = mne.create_info(ch_names=ch_names,
                                    ch_types=ch_types,
                                     montage=constants.montage,
                                     sfreq=fs,
                                     verbose=0)
        raw = mne.io.RawArray(eeg[1:,:], self.info)

        # montage = mne.channels.make_standard_montage(kind = constants.montage)
        # raw.set_montage(montage, verbose=0)

        chunked_events = [np.c_[[np.where(eeg[0,:] >= a )[0][0] for a in chunk[:,0]],
                        chunk[:,1], chunk[:,1]] for	chunk in chunked_events] # convert chunks to mne format
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
        chunked_events = self.labeler(chunked_events, targets = targets)
        chunked_events = np.array(chunked_events)
        return raw, chunked_events

    def read_eeg_and_evt_files(	self,
                                files_dict,
                                ignore_events_id=None,
                                events_offset=None):

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

    def read_csv_markup(self, markup_path: str) -> dict:
        with open(markup_path, encoding='utf-8') as m:
            reader = csv.DictReader(m)
            markup = [a for a in reader if a['user'] not in self.ignore_users]
        return markup

    def create_pickled_database(self, database_path: pathlib.Path = None) -> None:
        try:
            shutil.rmtree(database_path)
            print (f'Cleared data folder at {database_path}')
        except (NotADirectoryError, FileNotFoundError):
            pass
        pathlib.Path.mkdir(database_path)
        print (f'Created data folder at {database_path}')

    def write_epoch_to_db(self, epoch: np.ndarray) -> None:
        with open (self.database / f'{self.epoch_counter_global}.pickle', 'wb') as fh:
            pickle.dump(epoch, fh)
    
    def write_info_to_db(self) -> None:
        if self.info_written_to_db:
            pass
        else:
            print ('saving MNE info')
            with open (self.database / f'info.pickle', 'wb') as fh:
                pickle.dump(self.info, fh)
        self.info_written_to_db = True

    def save_epoch_markup(self) -> None:
        with open (self.database / f'epochs_markup.csv', 'w', newline='', encoding='utf-8') as fh:
            fieldnames = list(self.global_markup[0].keys())
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for line in self.global_markup:
                writer.writerow(line)

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
            ecg_events = None                           # later detect ecg events and find time delta with events for every epoch
            raw = self.preprocessing.process_raw_eeg(raw)
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

            for epoch, event, target, finger, session_id in zip(epochs, events, epochs_targets, epochs_fingers, sessions_id):
                epoch_markup_line = {
                                    'id': self.epoch_counter_global,
                                    'finger': fingers[event[-2]],
                                    'target': target,
                                    'event': event[-2],
                                    'is_target': event[-1],
                                    'epoch_id': epochs_ids[self.epoch_counter_record],
                                    'session_id': session_id
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
                                    })

        if preload:
            self.load_epoch = self.load_from_memory
            self.global_in_memory_database = [self.load_pickle(id) for id in self.markup['id']]
        else:
            self.load_epoch = self.load_pickle

    def load_from_memory(self, id: int) -> np.ndarray:
        return self.global_in_memory_database[id]

    def load_pickle(self, id: int) -> np.ndarray:
        with open(self.data_path / f'{id}.pickle', 'rb') as p:
            epoch = pickle.load(p)
            epoch *= constants.uV_scaler
        return epoch

    def create_binary_events_from_subset(self, subset):
        return np.c_[list(range(len(subset['is_target']))), subset['is_target'], subset['is_target']]

    def create_mne_epochs_from_subset(self, subset: pd.DataFrame) -> mne.EpochsArray: 
        epochs_subset = [self.load_epoch(id) for id in subset['id']]
        epochs_subset = mne.EpochsArray(data=epochs_subset,
                                        info=self.info,
                                        tmin=constants.epochs_tmin,
                                        events=self.create_binary_events_from_subset(subset)                                        )
        return epochs_subset
    
    def create_mne_evoked_from_subset(self, subset: pd.DataFrame,
                                            tmin: float=constants.epochs_tmin) -> mne.EpochsArray: 
        data = self.load_epoch(subset['id'].reset_index(drop=True)[0])
        cc = 1
        for id in subset['id'].reset_index(drop=True)[1:]:
            data += self.load_epoch(id)
            cc += 1
        data /= cc
        return mne.EvokedArray(info=self.info,
                                data=data,
                                tmin=tmin,
                                nave=cc)

if __name__ == "__main__":
    # Create dataset from raw data
    EpDatasetCreator(markup_path=folders.markup_path,
                            database_path=folders.database_path,
                            data_folder=folders.data_folder,
                            reference_mode='average', 
                            ICA=True,
                            fit_with_additional_lowpass=True
                            )