#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 15:00:15 2025

@author: freya
HFA_source analysis, averaged from 0.1~0.3s, baseline corrected from -0.1~0s
"""

import mne
import numpy as np
from pathlib import Path
from mne.time_frequency import csd_tfr
from mne.beamformer import apply_dics_tfr_epochs, make_dics

# In[1] epoch data
''' dataloading and selection'''
data_dir = '/Volumes/Freya/PhD_data/attention_MEG/python_epoched_data/' # epoch data
fwd_dir = '/Volumes/Freya/PhD_data/attention_MEG/python_fwd_data' # path of raw data

folder = Path(data_dir) 
epoch_list = list(folder.glob('*targ_epo.fif'))
name_list = [f.name[0:2] for f in epoch_list] # subjID list 

mean_hfa = np.zeros((8196,1))

for sub_ID in range(len(name_list)):
    sub_ID = 0
    epoch_data = mne.read_epochs(epoch_list[sub_ID])
    epoch_data = epoch_data.crop(tmin=-1.0, tmax=1.5)
    # epochs_hfa = epoch_data.filter(l_freq=80., h_freq=150., method='fir')
    # epochs_hfa.apply_hilbert(envelope=True)
    
    freqs = np.linspace(80, 150, 1)
    epochs_tfr = epoch_data.compute_tfr(
        "morlet", freqs, n_cycles=5, return_itc=False, output="complex", average=False
    )
    
    # load fwd 
    folder = Path(fwd_dir) 
    filename = '*'+name_list[sub_ID]+'-fwd.fif'
    fwd_path = next(folder.rglob(filename), None)
    fwd = mne.read_forward_solution(fwd_path)
    
    ''' covariance matrix'''
    csd = csd_tfr(epochs_tfr,tmin=-0.1, tmax=1.0)
    baseline_csd = csd_tfr(epochs_tfr, tmin=-0.5, tmax=-0.1)
    
    # compute scalar DICS beamfomer
    filters = make_dics(
        epoch_data.info,
        fwd,
        csd,
        noise_csd=baseline_csd,
        pick_ori="max-power",
        reduce_rank=True,
        real_filter=True,
    )
    
    # project the TFR for each epoch to source space
    epochs_stcs = apply_dics_tfr_epochs(epochs_tfr, filters, return_generator=True)
    
    data = np.zeros((fwd["nsource"], epochs_tfr.times.size))
    for stc in epochs_stcs:
        data += (stc.data * np.conj(stc.data)).real
    
    stc.data = data / len(epoch_data)
    stc.apply_baseline((-0.1, 0))
    
    stcs_avg = stc.copy().crop(0.1, 0.3).mean()
    
    mean_hfa += stcs_avg.data



stcs_avg.data = mean_hfa/len(name_list)
subjects_dir = '/Users/freya/mne_data/MNE-fsaverage-data' # for MRI data

stcs_avg.plot(
    subject='fsaverage',
    hemi='split',
    views=['lat', 'med'],
    subjects_dir=subjects_dir,
    time_label='HFA amplitude (80–150 Hz)'
)

''' anatomy result'''
labels_lh = mne.read_labels_from_annot(
    subject='fsaverage',      
    parc='aparc.a2009s',             
    hemi='lh'
)
labels_rh = mne.read_labels_from_annot(
    subject='fsaverage',
    parc='aparc.a2009s',
    hemi='rh'
)
labels = labels_lh + labels_rh

src = fwd['src']

label_ts = mne.extract_label_time_course(
    stcs_avg, labels, src=src, mode='mean_flip'
)

mean_amplitude = np.mean(np.abs(label_ts), axis=1)
sorted_idx = np.argsort(mean_amplitude)[::-1]

for i in sorted_idx[:5]:
    print(f"{labels[i].name}: {mean_amplitude[i]:.3f}")
