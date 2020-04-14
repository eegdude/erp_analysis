
# EEG analysis for Braille BCI experiment

## Brief experiment description
10 blind and 10 sighted subjects were using tactile P300 BCI, implemented with Alva 640 comfort Braille display.

4 fingers on each hand were placed on separate Braille cells, serving as stimuli. Each stimulus was activated 10 times in random order. The task for the participant was to count activations.

For blind subjects, Braille proficiency was assessed with the reading test before the experiment.
The experiment included BCI sessions with large (8 dots) and small (1 dot) stimuli.

The present analysis is focused on the comparison of BCI performance for large and small stimuli in blind and sighted individuals.

For more details, see our preprint at ############

Raw dataset is available at https| //osf.io/yckhm/

## About this repo
This branch contains several Jupyter notebooks with complete analysis, used in the preprint.
*  demographics.ipynb <br>
    Age, sex and medical condition-related visualizations
*  np_draw.ipynb <br>
    Visualizations of evoked data in different conditions.
*  np_stats.ipynb <br>
    Statistics on evoked potentials - cluster-based permutation tests and more
*  classification.ipynb <br>
    
*  nb_heart.ipynb <br>
    Analysis of cardiosynchronous activity

Notebooks use functions from these .py files| 
*  dataset.py <br>
    Preprocessing and loading EEG and evoked data
*  analysis_and_plotting_functions.py <br>
    Statistical testing and plotting evoked potentials
*  constants.py <br>
    important signal processing parameters

Other branches contain data analysis routines for different experiments.

## Analysis workflow
The data analysis pipeline is optimized for easy computation of different statistics and machine learning metrics for evoked data.
The analysis includes two steps|  
1. Preprocessing data

    The continuous EEG is read from the disk, filtered, and re-referenced. Oculomotor artifacts are optionally removed using ICA. The resulting clean EEG is being cut into epochs, then the epochs are saved to disk as a single file.
    Global per-epochs markup is created with all metadata that is available before analysis, as well as computed during preprocessing.

    While this may not the best practice from the point of software architecture, but is probably fine if preprocessed data is not being read too often and serialization of the preprocessed database is not needed. It doesn't influence analysis results anyway.
2. Data analysis.

    The database is being loaded in the memory as a whole.
    This solution enables the user to explore, plot and calculate statistics on different subsets of evoked data without reading all dataset from disk every time and without repeating preprocessing steps.
    However, if there is not enough memory to fit the whole dataset, there is an option to read needed epochs one by one.

The dataset object includes a markup variable, which is a Pandas data frame, containing all available metadata for every epoch.
It can be searched and filtered as any Pandas data frame.

For example, 
```python
ds.markup.loc[ (ds.markup['blind'] == 1) &
               (ds.markup['reg'] == 'large') &
               (ds.markup['finger'].isin([7,6,5,4]))
                                ]
```
will return metadata for epochs, recorded from right hands of blind subjects, performing BCI task with large stimulus. More on this in [dataset description](#Dataset-description) section.

This metadata can then be used to select these epochs, average them, plot or feed into the classifier.


## Dataset description
### Per-record markup

|column name |description
|---|---|
|  user|  user id
|  folder|  folder with a record inside the user folder
|  reg|  BCI mode. 'large' for large stimuli, 'small' for small |  stimuli 
|  targets|  list of target stimuli for all learning and stimuli selection cycles
|  fingers|  which finger corresponds to specific stimuli code.<br> right hand  0 - pinky, 1 - ring, 2 - middle, 3 - index<br>left hand  4 - index, 5 - middle, 6 - ring, 7 - pinky
|  ecg_r_peak_direction|  1 if R-peak is up, -1 if R-peak is down, 0 or 2 if no ECG
|  leading_hand|  'r' if right, 'l' if left
|  ignore_events_id|  list of events to not include into analysis
|  reading_finger|  list of fingers used for Braille reading
|  blind|  1 if blind, 0 if sighted
|  order|  index of the record in the experiment. Failed sessions that have not made it into the dataset are counted.
|  blindness_age|  age of complete vision loss
|  reading_time|  time of reading Braille text of standard informed consent
|  braille_display_user|  1 if the subject uses Braille display in daily life
|  daily_braille_time|  self-reported daily Braille usage, hours
|  age|  user's age
|  music|  1 if the subject has musical experience (except vocal), 0 if not
|  remaining_vision|  1 if the user has residual vision, 0 if not
|  sex|  male or female
|  blind_years|  years passed from total vision loss
|  congenitaly_blind|  1 if congenital blindness, 0 if acquired

### Per-epoch markup
The preprocessed dataset has some added columns, which are calculated during the preprocessing stage.
|column name |description
|---|---|
|  id|  unique epoch id
|  event|  epoch event (activated stimuli id)
|  target|  id of current target stimuli
|  is_target|  1 if epoch is target, 0 if not
|  session_id|  stimuli selection cycle index inside experiment
|  epoch_id|  epoch index inside stimuli selection cycle (from 0 to 79)
|  ms_after_r|  milliseconds after R-peak
|  ms_before_r|  milliseconds before next R-peak

### Raw folder structure
```
\
└── markup.csv                               # metadata for all records
└── user 
    └── record
        └── _data_$MODE__play_$TIME.npy      # Raw EEG data
        └── _events_$MODE__play_$TIME.npy    # BCI events
        └── _photocell_$MODE__play_$TIME.npy # sync events
        └── Rpeaks.npy                       # preprocessed R-peaks, if availible
```
users are numbered from 6 to 26

$MODE can be 'large' for large stimuli and 'small' for small stimuli


## Reproduction manual
To successfully run the analysis,
1. Install all dependencies as specified in requirements.txt. The pipeline is tested with Python 3.7.4
2. Edit `folders.py`, adding the path to the downloaded raw data, the path where to put processed EEG database, and the path to markup.csv file, that describes the raw data.
3. Create the database with `dataset.py`. You may edit the analysis parameters like ICA there.
4. Some other filtering and plotting parameters can be adjusted in `constants.py`.
5. The analysis is implemented in Jupyter notebooks.
6. Enjoy)

## Work in progress
*  Create the version of the dataset with EEG data converted to BDF format with BIDS metadata.
*  Switch to a more efficient database solution
*  Migrate to mne 0.20
*  Roll out new data analysis features