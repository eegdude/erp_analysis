
# EEG analysis for matrix speller BCI experiment with facial stimuli

## Brief experiment description
The subjects are using 6x6 matrix speller with facial stimuli.
// insert description here

## Dataset description
The dataset follows BIDS format. Various stimuli parameters are descriped in events.tsv files.
|column name |description
|---|---|
|  onset |  time of event onset since the begining of the file
|  duration|  event duration (zero, retained for compatibility)
|  trial_type|  1 for target, 0 for non-target
|  value| event value. Stimuli is coded as number%1000, code >1000 corresponds to rare stimuli.
|  sample| sample of event start
|  target|  current target stimuli
|  is_rare|  is current stimuli rare
|  epoch_id|  0, not used
|  session_id|  current session number
|  valence| valence of current stimulus 
|  user| current user ID
|  order|  order of current record in the experiment
|  reg|  session, fix naming in next iteration
|  general_valence|  valence of frequent stimuli in the record
|  rare_valence|  valence of rare stimuli in the record

### sourcedata data folder structure
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
users are numbered from 1 to 16

$MODE corresponds to $session in BIDS structure

### Preprocessed data folder structure
```
\
└── markup.csv                               # metadata for all epochs, created at preprocessing stage
└── 0.pickle                                 # pickle files, corresponding to single epochs

...

└── $N.pickle
```
## Reproduction manual
To successfully run the analysis:
1. Install all dependencies as specified in requirements.txt. The pipeline is tested with Python 3.7.4
2. Edit `folders.py`, adding the path to the downloaded raw data, the path where to put processed EEG database, and the path to markup.csv file, that describes the raw data.
3. Create the database with `dataset.py`. You may edit the analysis parameters like ICA there.
4. Some other filtering and plotting parameters can be adjusted in `constants.py`.
5. The analysis is implemented in Jupyter notebooks.
6. Enjoy)

## Work in progress
*  Create the version of the dataset with EEG data converted to BIDS format.
*  Switch to a more efficient database solution
*  Migrate to mne 0.20
*  Roll out new data analysis features
*  Refactor code for more flexibility
*  Create better BIDS-convertor
*  Make BIDS the single entry point for analysis
