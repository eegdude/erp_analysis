import numpy as np
import pandas as pd
from scipy import stats, signal
import math
import pickle
import itertools
from sklearn import base, model_selection, metrics, discriminant_analysis, pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import constants


'''Create preprocessing functions
'''
class downsampler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.downsample_div = 10
    def fit(self, data, y=None):
        return self
    def transform(self, data):
        return signal.decimate(data, self.downsample_div, axis=-1)

class channel_selector(BaseEstimator, TransformerMixin):
    def __init__(self, classifier_channels = [16,17,18,19,20,25,27,29,34,36,42]):
        self.classifier_channels = classifier_channels
    def fit(self, data, y=None):
        return self
    def transform(self, data):
        if self.classifier_channels:
            return data[:, self.classifier_channels, :]
        else:
            return data

class epoch_cutter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, data, y=None):
        return self
    def transform(self, data):
        return data[:, :, int(abs(constants.epochs_tmin*constants.fs)):]

class reshaper(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, data, y=None):
        return self
    def transform(self, data):
        s = data.shape
        data= data.reshape((s[0], s[1]*s[2]))
        return data     

class printer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, data, y=None):
        return self
    def transform(self, data):
        print (f'data shape after preproc: {data.shape}')
        return data

def score_func(y, y_pred, **kwargs):
    target_p = [a for a, b in zip(y_pred, y) if b]
    nontarget_p = [a for a, b in zip(y_pred, y) if not b]
    nontarget_p = np.array_split(nontarget_p, len(nontarget_p)/len(target_p))
    nontarget_p = np.mean(nontarget_p, axis=1)
    target_p = np.mean(target_p)
    if len(np.where(nontarget_p>target_p)[0]) == 0:
        return 1
    else:
        return 0


'''Create classification pipline. Use LDA for feature vectors and PIPE for preprocessed dataset
'''
preproc_pipe = pipeline.make_pipeline(
#     epoch_cutter(),
    channel_selector(),
    downsampler(),
    reshaper(),
    # printer()
    )
LDA = discriminant_analysis.LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage='auto')
PIPE = pipeline.make_pipeline(preproc_pipe, LDA)


'''Crearte meta-pipelines to get metrics for whole dataset
'''
def classifier_metrics_aggergated(ds, subset=None,
    n_repeats=10, n_splits=1000, train_size=480, random_state=282, preprocessed=True):
    n_stimuli = len(set(subset['event']))
    total_scores = []
    users = sorted(set(subset['user']))
    for user in users:
        print (f'user {user}')
        user_subset = subset[subset['user'] == user]
        
        if preprocessed:
            X = ds.feature_vectors_db[user_subset['id'],:]
            y = user_subset['is_target']
            p = LDA.fit(X, y=y)
        else:
            X = ds.create_mne_epochs_from_subset(user_subset).crop(tmin=0)
            y = X.events[:,-1]
            X = X._data
            p = PIPE.fit(X, y=y)
            
        skf = model_selection.StratifiedShuffleSplit(test_size=n_stimuli*n_repeats, train_size=train_size, n_splits=n_splits, random_state=random_state)
        scores = model_selection.cross_val_score(p,
                                X,
                                y,
                                cv=skf,
                                scoring=metrics.make_scorer(score_func, needs_proba=True), verbose=1, n_jobs=-1)
        total_scores.append(np.mean(scores))
    return total_scores

def classifier_any_groups(ds, y, subset=None):
    n_stimuli = len(set(subset['event']))
    X = ds.create_mne_epochs_from_subset(subset).crop(tmin=0)
    print(X._data.shape)
    PIPE.fit(X._data, y=y)
    skf = model_selection.RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=282)
    scores = model_selection.cross_val_score(PIPE,
                            X._data,
                            y,
                            cv=skf,
                            scoring='balanced_accuracy')
    return scores, np.mean(scores)


'''Additional statistical and plotting functions
'''
def pickler(dump, name, filepath:str='classification_results.pickle'):
    """Dump classification results on disc
    """    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        data = {}
    data[name] = dump
    with open(filepath, 'wb') as f:
        data = pickle.dump(data, f)
        
def bits_per_trial(p, n:int=8):
    """Calculate bits per selection trial, as defined by Shannon.
    Arguments:
    n -- int -- number of stimuli
    """
    bpm =  math.log(n, 2) 
    if p !=0:
        bpm += p*math.log(p,2) 
    if p !=1:
        bpm+=(1-p)*math.log(((1-p)/(n-1)), 2)
    return bpm

def create_dataframe(totals, rng):
    tb = pd.DataFrame({'repeats':np.ravel([[a]*len(totals[0]) for a in rng]), 'accuracy':np.ravel(totals)})
    tb['bits_per_trial'] = list(map(bits_per_trial, tb['accuracy']))
    tb['itr'] = tb['bits_per_trial']*(60/(0.3*8*tb['repeats']))
    return tb

def boxplot(*args, ylabel='', xticks=['small', 'large'], color=None):
    sns.set(context='notebook', style='white')    
    sns.boxplot(data=args, color='white')
    sns.swarmplot(data=args, color=color)
    sns.utils.axlabel(xlabel=None, ylabel=ylabel, fontsize=16)
    plt.xticks(plt.xticks()[0], xticks)
    plt.axhline(1/8, linestyle='--')

def boxplot_from_dict(data, ylabel='', xticks=['small', 'large'], hue='blind', 
                    palette=[constants.plot_colors['sighted'], constants.plot_colors['blind']],
                    figsize=(4,5)):
    sns.set(context='notebook', style='white')    

    fig, axs = plt.subplots(1, len(data), figsize=figsize, sharey=True)
    for ax, title in zip(axs, data.keys()):
        b = sns.boxplot(x=data[title]['groups'], y=data[title]['accuracy'], color='white', ax=ax)
        s = sns.swarmplot(x=data[title]['groups'], y=data[title]['accuracy'], hue=data[title]['hue'], palette=palette, ax=ax)
        ax.get_legend().set_visible(False)
        ax.axhline(1/8, linestyle='--')
        ax.set_title(title)
    L = axs[-1].legend()
    L.set_bbox_to_anchor((1.2, 1))
    L.get_texts()[0].set_text(data[title]['group_names'][0])
    L.get_texts()[1].set_text(data[title]['group_names'][1])
    axs[0].set(xlabel=None, ylabel=ylabel)

def remove_outliers(df):
    df = pd.DataFrame(df)
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
    
def mwtest(*args):
    st =  stats.mannwhitneyu(args[0], args[1])
    print(st, ' *' if st[1]<=0.05 else '')
    return st

def wilcox(*args):
    st =  stats.wilcoxon(args[0], args[1])
    print(st, ' *' if st[1]<=0.05 else '')
    return st

def med(*args):
    st =  [np.median(a) for a in args]
    print(st)
    return st

def ttest_ind(*args):
    st =  stats.ttest_ind(args[0], args[1])
    print(st)
    return st

def accuracy_stats(a, b, independent=True):
    med(remove_outliers(a), remove_outliers(b))
    if independent:
        mwtest(remove_outliers(a), remove_outliers(b))
    else:
        wilcox(a, b)

def r2(x, y):
    st = stats.kendalltau(x, y)
    print (st, ' *' if st[1]<=0.05 else '')
    return st[0]**2