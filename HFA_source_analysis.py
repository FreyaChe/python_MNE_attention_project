#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 22:08:52 2025

@author: freya
"""

import mne
from tqdm import tqdm
import numpy as np
from pathlib import Path
from mne.beamformer import make_lcmv, apply_lcmv_epochs
import matplotlib.pyplot as plt
from mne.stats import permutation_t_test

# In[1] epoch data
mne.set_log_level('WARNING')  # only show warning and error

''' dataloading and selection'''
data_dir = '/Volumes/Freya/PhD_data/attention_MEG/python_epoched_data/' # epoch data
fwd_dir = '/Volumes/Freya/PhD_data/attention_MEG/python_fwd_data' # path of fwd data

folder = Path(data_dir) 
epoch_list = list(folder.glob('*targ_epo.fif'))
name_list = [f.name[0:2] for f in epoch_list] # subjID list 

mean_hfa = np.zeros((8196,len(name_list)))

for sub_ID in range(31):
    epoch_data = mne.read_epochs(epoch_list[sub_ID])
    epoch_data = epoch_data.crop(tmin=-1.0, tmax=1.5) # crop data to -1~1.5
    
    # HFA amplitude
    l_freq, h_freq = 80, 150
    epochs_hfa = epoch_data.filter(l_freq, h_freq, fir_design='firwin')
    epochs_hfa.apply_hilbert(envelope=True)
    epochs_hfa.apply_baseline(baseline=(-0.1, 0))
    
    # covariance
    data_cov = mne.compute_covariance(epochs_hfa, tmin=0.0, tmax=1.0)
    noise_cov = mne.compute_covariance(epochs_hfa, tmin=-0.5, tmax=0.0)
    
    # load fwd
    folder = Path(fwd_dir)
    filename = '*'+name_list[sub_ID]+'-fwd.fif'
    fwd_path = next(folder.rglob(filename), None)
    fwd = mne.read_forward_solution(fwd_path)
    
    # source analysis
    filters = make_lcmv(epochs_hfa.info, fwd, data_cov, reg=0.05,
                        noise_cov=noise_cov, weight_norm='nai')
    
    stcs = apply_lcmv_epochs(epochs_hfa, filters, return_generator=True, verbose=False)
    
    data = np.zeros((fwd["nsource"], epochs_hfa.times.size))
    for stc in tqdm(stcs, total=len(epochs_hfa), desc="Processing STCs"):
        data += (stc.data * np.conj(stc.data)).real
        
    stc.data = data / len(epochs_hfa)
    stc_avg = stc.copy().crop(0.1, 0.3).mean()  # average accross 0.1~0.3s
    mean_hfa[:, sub_ID] = stc_avg.data[:, 0]  # matrix of results

stc.data = mean_hfa # average result
stc.save('/Users/freya/Documents/GitHub/python_MNE_attention_project/Averaged_HFA')


# In[2] group result
stc = mne.read_source_estimate('/Users/freya/Documents/GitHub/python_MNE_attention_project/Averaged_HFA')
stc_avg = stc.copy()
# stc_avg.data = stc.copy().data.mean(axis = 1)[:,None]

# normarlize
norm_data = np.zeros((8196,len(name_list)))
mean_hfa = stc.data.copy()
for f in range(31):
    norm_data[:,f] = (mean_hfa[:,f] - np.min(mean_hfa[:,f])) / (np.max(mean_hfa[:,f]) - np.min(mean_hfa[:,f]))
    
stc_avg.data = norm_data.mean(axis = 1)[:,None]

# In[plot figure]
# visualization
subjects_dir = '/Users/freya/mne_data/MNE-fsaverage-data' # for MRI data

# lateral plot
brain = stc_avg.plot(
    subject='fsaverage',
    hemi='split',
    subjects_dir=subjects_dir,
    views='lat',
    size=(800, 420),
    background="w",
    time_viewer=False,
    show_traces=False,
    colorbar=False,
)
# brain.add_annotation("aparc.a2009s")
screenshot_lat = brain.screenshot()
brain.close()

# med plot
brain = stc_avg.plot(
    subject='fsaverage',
    hemi='split',
    subjects_dir=subjects_dir,
    views='med',
    size=(800, 395),
    background="w",
    time_viewer=False,
    show_traces=False,
    colorbar=False,
)
# brain.add_annotation("aparc.a2009s")
screenshot_med = brain.screenshot()
brain.close()


fig, axes = plt.subplots(nrows=1, ncols=2)
fig.subplots_adjust(wspace=0.02, hspace=0) 
lat_idx = 0
med_idx = 1

axes[lat_idx].imshow(screenshot_lat)
axes[lat_idx].axis("off")

axes[med_idx].imshow(screenshot_med)
axes[med_idx].axis("off")

fig.savefig('/Users/freya/Documents/GitHub/python_MNE_attention_project/HFA_source_plot.png', dpi=600, bbox_inches='tight')
plt.close(fig)

# In[anatomy for HFA ROI]
stc = mne.read_source_estimate('/Users/freya/Documents/GitHub/python_MNE_attention_project/Averaged_HFA')

''' anatomy result'''
labels = mne.read_labels_from_annot(
    subject='fsaverage',      
    parc='aparc.a2009s',        #HCPMMP1_combined,aparc.a2009s       
    hemi='both'
)

'''permutation for significant higher amplitude location'''
data_for_test = norm_data.T - 0.5
T_obs, p_values, H0 = permutation_t_test(
    data_for_test,n_permutations=10000,tail = 1)
significant_points = p_values < 0.05

stc_sig = stc_avg.copy()
stc_sig.data[~significant_points, :] = 0

''' find highest ROI'''
mean_amp = []
stc.data = stc_sig.data
for label in labels:
    stc_label = stc.in_label(label)
    mean_amp.append(stc_label.data.mean())
    
sorted_idx = np.argsort(mean_amp)[::-1]

selected_labels = []  
for i in sorted_idx[:2]:
    selected_labels.append(labels[i])
    print(f"{labels[i].name}: {mean_amp[i]:.3f}")

''' plot result'''
brain = stc_sig.plot(
    subject='fsaverage',
    hemi='split', 
    subjects_dir=subjects_dir,
    views='lat',
    size=(800, 400),
    background='w',
    show_traces=False,
    time_viewer=False,
    colorbar=False,
)

for label in selected_labels:
    brain.add_label(label, color='green', alpha=0.8)  

