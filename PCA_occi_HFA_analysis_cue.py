#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 17:30:53 2025

@author: freya
phase-amplitude coupling analysis between occipital HFA (two ROI from source analysis) and whole brain theta
"""
import mne
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, hilbert
from mne.minimum_norm import apply_inverse_epochs, make_inverse_operator
from itertools import zip_longest
# from mne.stats import permutation_t_test
from scipy.stats import ttest_1samp
from tqdm import tqdm

# In[]
''' dataloading and selection'''
data_dir = '/Volumes/Freya/PhD_data/attention_MEG/python_epoched_data/' # epoch data
fwd_dir = '/Volumes/Freya/PhD_data/attention_MEG/python_fwd_data' # path of fwd data

folder = Path(data_dir) 
epoch_list = list(folder.glob('*cue_epo.fif'))
name_list = [f.name[0:2] for f in epoch_list] # subjID list 

def compute_pac_tort(theta_phase, hfa, n_bins=18):
    n_theta_ch, n_time=theta_phase.shape
    n_hfa_ch=hfa.shape[0]
    
    phase_bins=np.linspace(-np.pi, np.pi, n_bins + 1)
    pac_matrix=np.zeros((n_theta_ch, n_hfa_ch))
    
    bin_idx=np.digitize(theta_phase, phase_bins) - 1  # [0 ~ n_bins-1]
    
    for i_hfa in range(n_hfa_ch):
        amp=hfa[i_hfa]
        amp_means=np.zeros((n_theta_ch, n_bins))
    
        for b in range(n_bins):
            mask=(bin_idx == b)
            # mean across time with masking
            amp_means[:, b]=np.sum(amp * mask, axis=1)/(np.sum(mask, axis=1) + 1e-10)
    
            # Normalize & compute entropy
        amp_means /= amp_means.sum(axis=1, keepdims=True)
        entropy=-np.sum(amp_means * np.log(amp_means + 1e-10), axis=1)
        entropy_max=np.log(n_bins)
        pac_matrix[:, i_hfa]=(entropy_max - entropy) / entropy_max
    return pac_matrix


# In[loop analysis]
''' for loop '''
PAC_P = []
PAC_R = []
for sub_ID in range(31):
    print(sub_ID)
    epoch_data = mne.read_epochs(epoch_list[sub_ID])
    epoch_data = epoch_data.crop(tmin=-1.0, tmax=1.5)  # crop data to -1~1.5
    
    # load fwd
    folder = Path(fwd_dir)
    filename = '*'+name_list[sub_ID]+'-fwd.fif'
    fwd_path = next(folder.rglob(filename), None)
    fwd = mne.read_forward_solution(fwd_path)
        
    ''' anatomy roi'''
    labels = mne.read_labels_from_annot(
        subject='fsaverage',      
        parc='aparc.a2009s',             
        hemi='both'
    )
    roi1 = [lab for lab in labels if lab.name == "G_oc-temp_med-Lingual-lh"][0]
    roi2 = [lab for lab in labels if lab.name == "G_cuneus-lh"][0]
    
    # invasive
    noise_data_raw_file = '/Users/freya/Study/self-learning/python/MEG_example/emptyroom.fif'
    noiseData = mne.io.read_raw_fif(noise_data_raw_file) #can use preload = false and load_data()to load it into memory later
    noise_cov = mne.compute_raw_covariance(noiseData, tmin=0, tmax=None)
    
    inverse_operator = make_inverse_operator(
        epoch_data.info, fwd, noise_cov, loose=0.2, depth=0.8
    )
    
    # source analysis
    pa_idx = np.where(epoch_data.events[:,1]==20)[0] # PA trials
    re_idx = np.where(epoch_data.events[:,1]==10)[0] # rest trials
    
    snr = 1.0
    lambda2 = 1.0 / snr**2
    method = "dSPM"
    
# PA condition
    # HFA roi time course
    stcs_pa = apply_inverse_epochs(
        epoch_data[pa_idx], inverse_operator, lambda2, method, pick_ori="normal", return_generator=True
    ) #set generator as true to avoid conflict between stcs_pa and pa_ts 
    
    pa_ts = mne.extract_label_time_course(
        stcs_pa, [roi1, roi2], src=fwd['src'], mode="mean_flip", return_generator=False
    )
    stcs_pa = apply_inverse_epochs(
        epoch_data[pa_idx], inverse_operator, lambda2, method, pick_ori="normal", return_generator=True
    )
    fs = 500
    theta_b,theta_a = butter(4, [4/(fs/2),8/(fs/2)],btype ='band') # theta band 
    hfa_b,hfa_a = butter(4, [80/(fs/2),150/(fs/2)],btype ='band') # HFA band
    time = np.linspace(-1.0,1.5, int(500*2.5)+1)
    indices = np.where((time > 0) & (time < 0.65))[0] # PAC from 0~0.65s
    
    pac_pa = np.zeros((8196,2))
    for ts, stc in tqdm(zip_longest(pa_ts, stcs_pa), total=len(pa_ts), desc="Processing STCs"): #zip_longest(pa_ts, stcs_pa):
        # theta phase
        theta_filt = filtfilt(theta_b, theta_a, stc.data)
        theta_hilbert = hilbert(theta_filt)
        theta_phase = np.angle(theta_hilbert)
        theta_phase = theta_phase[:,indices]
        # HFA amplitude
        hfa_filt = filtfilt(hfa_b, hfa_a, ts)
        hfa_hilbert = hilbert(hfa_filt)
        hfa = np.abs(hfa_hilbert)
        hfa = hfa[:,indices]
        pac_pa += compute_pac_tort(theta_phase, hfa, n_bins=18)
    
    del pa_ts, stcs_pa
    PAC_P.append(pac_pa/len(pa_idx))

    
# rest condition
    stcs_rest = apply_inverse_epochs(
        epoch_data[re_idx], inverse_operator, lambda2, method, pick_ori="normal", return_generator=True
    )
    rest_ts = mne.extract_label_time_course(
        stcs_rest, [roi1, roi2], src=fwd['src'], mode="mean_flip", return_generator=False
    )
    stcs_rest = apply_inverse_epochs(
        epoch_data[re_idx], inverse_operator, lambda2, method, pick_ori="normal", return_generator=True
    )
    pac_rest = np.zeros((8196,2))
    for ts, stc in tqdm(zip_longest(rest_ts, stcs_rest), total=len(rest_ts), desc="Processing STCs"): #zip_longest(rest_ts, stcs_rest):
        theta_filt = filtfilt(theta_b, theta_a, stc.data)
        theta_hilbert = hilbert(theta_filt)
        theta_phase = np.angle(theta_hilbert)
        theta_phase = theta_phase[:,indices]
        hfa_filt = filtfilt(hfa_b, hfa_a, ts)
        hfa_hilbert = hilbert(hfa_filt)
        hfa = np.abs(hfa_hilbert)
        hfa = hfa[:,indices]
        pac_rest += compute_pac_tort(theta_phase, hfa, n_bins=18)
    
    del rest_ts, stcs_rest

    PAC_R.append(pac_rest/len(re_idx))
    np.savez('/Users/freya/Documents/GitHub/python_MNE_attention_project/PAC_cue.npz', PAC_R=PAC_R, PAC_P=PAC_P)# save every round in case crush    


# In[statistic analysis: permutation test] PA-rest
from scipy import sparse
from mne import spatial_src_adjacency
from scipy.sparse.csgraph import connected_components

data = np.load('/Users/freya/Documents/GitHub/python_MNE_attention_project/PAC_cue.npz')
PAC_R = data['PAC_R']
PAC_P = data['PAC_P']

pac_p = np.array([d[:,1]+d[:,0] for d in PAC_P]) # sum results from two ROIs
pac_r = np.array([d[:,1]+d[:,0] for d in PAC_R])
data = pac_p-pac_r # calculate difference between PA-Rest

# permutation test
T_obs, p_values = ttest_1samp(data, popmean=0)

n_perm = 10000 # permutation time
n_subj = pac_r.shape[0]
T_perm = np.zeros((n_perm, pac_r.shape[1]))
[n_sub,n_chan] = data.shape
for i in range(n_perm):
    data_1d = data.reshape(-1)
    np.random.shuffle(data_1d)
    perm_data = data_1d.reshape(n_sub,n_chan)
    T_perm[i, :] = perm_data.mean(axis=0) / (perm_data.std(axis=0) / np.sqrt(n_subj))

p_values_perm = np.mean(np.abs(T_perm) >= np.abs(T_obs), axis=0)
sig_mask = p_values_perm < 0.05

# cluster results
adjacency = spatial_src_adjacency(fwd['src'])
adjacency_sparse = adjacency.tocsr()

sig_mask_sparse = sparse.csr_matrix(sig_mask.reshape(-1, 1))
adj_sub = adjacency_sparse[sig_mask, :][:, sig_mask]

n_clusters, cluster_labels = connected_components(adj_sub)
min_cluster_size = 5
cluster_sizes = np.bincount(cluster_labels)
large_clusters = np.where(cluster_sizes >= min_cluster_size)[0]

stcs = []
clus = np.zeros(np.shape(T_obs))
clus_data = np.zeros(np.shape(T_obs))
ave_data = p_values_perm
for clu in large_clusters:
    clu_mask = cluster_labels == clu
    clu_vertices = np.where(sig_mask)[0][clu_mask]

    clu_t = np.zeros(len(T_obs))
    clu_t[clu_vertices] = T_obs[clu_vertices]
    clus += clu_t

    clu_data = np.zeros(len(T_obs))
    clu_data[clu_vertices] = ave_data[clu_vertices]
    clus_data += clu_data
    
print(clus[np.where(clus>0)].mean())
print(clus_data[np.where(clus_data>0)])

stc = mne.SourceEstimate(
   clus,
   vertices=[fwd['src'][0]['vertno'], fwd['src'][1]['vertno']],
   tmin=0, tstep=1, subject="fsaverage",
)


''' visualize'''
subjects_dir = '/Users/freya/mne_data/MNE-fsaverage-data' # for MRI data
# lateral plot
brain = stc.plot(
    subject='fsaverage',
    hemi='split',
    subjects_dir=subjects_dir,
    views=['lat'],
    size=(800, 400),
    background="w",
    # time_viewer=False,
    show_traces=False,
    colorbar=False,
)
brain.add_annotation("HCPMMP1")
# brain.save_image('/Users/freya/Documents/GitHub/python_MNE_attention_project/PAC_cue_source_plot.png')


''' find ROIs'''
labels = mne.read_labels_from_annot(
    subject='fsaverage',      
    parc='aparc.a2009s',        #HCPMMP1_combined,aparc.a2009s       
    hemi='both'
)

mean_amp = []
for label in labels:
    stc_label = stc.in_label(label)
    mean_amp.append(stc_label.data.mean())
    
sorted_idx = np.argsort(mean_amp)[::-1]

selected_labels = []  
for i in sorted_idx[:7]:
    selected_labels.append(labels[i])
    print(f"{labels[i].name}: {mean_amp[i]:.3f}")

# In[brain plot] averaged 
pac_p = np.array([d[:,1]+d[:,0] for d in PAC_P]) # sum results from two ROIs
pac_r = np.array([d[:,1]+d[:,0] for d in PAC_R])
data = pac_p+pac_r # calculate sum
norm_data = np.zeros((data.shape))
for f in range(pac_p.shape[0]):
    norm_data[f,:] = (data[f,:] - np.min(data[f,:])) / (np.max(data[f,:]) - np.min(data[f,:]))

# permutation test
data = norm_data - 0.5
T_obs, p_values = ttest_1samp(data, popmean=0)

n_perm = 10000 # permutation time
n_subj = pac_r.shape[0]
T_perm = np.zeros((n_perm, pac_r.shape[1]))
[n_sub,n_chan] = data.shape
for i in range(n_perm):
    data_1d = data.reshape(-1)
    np.random.shuffle(data_1d)
    perm_data = data_1d.reshape(n_sub,n_chan)
    T_perm[i, :] = perm_data.mean(axis=0) / (perm_data.std(axis=0) / np.sqrt(n_subj))

p_values_perm = np.mean(np.abs(T_perm) >= np.abs(T_obs), axis=0)
sig_mask = (p_values_perm < 0.05) & (T_obs > 0)

# cluster results
adjacency = spatial_src_adjacency(fwd['src'])
adjacency_sparse = adjacency.tocsr()

sig_mask_sparse = sparse.csr_matrix(sig_mask.reshape(-1, 1))
adj_sub = adjacency_sparse[sig_mask, :][:, sig_mask]

n_clusters, cluster_labels = connected_components(adj_sub)
min_cluster_size = 5
cluster_sizes = np.bincount(cluster_labels)
large_clusters = np.where(cluster_sizes >= min_cluster_size)[0]

stcs = []
clus = np.zeros(np.shape(T_obs))
for clu in large_clusters:
    clu_mask = cluster_labels == clu
    clu_vertices = np.where(sig_mask)[0][clu_mask]

    clu_data = np.zeros(len(T_obs))
    clu_data[clu_vertices] = T_obs[clu_vertices]
    clus += clu_data

stc = mne.SourceEstimate(
   clus,
   vertices=[fwd['src'][0]['vertno'], fwd['src'][1]['vertno']],
   tmin=0, tstep=1, subject="fsaverage",
)


''' visualize'''
subjects_dir = '/Users/freya/mne_data/MNE-fsaverage-data' # for MRI data
# lateral plot
brain = stc.plot(
    subject='fsaverage',
    hemi='split',
    subjects_dir=subjects_dir,
    views=['lat'],
    size=(800, 400),
    background="w",
    # time_viewer=False,
    show_traces=False,
    colorbar=False,
)

# brain.save_image('/Users/freya/Documents/GitHub/python_MNE_attention_project/PAC_cue_source_plot.png')


''' find ROIs'''
labels = mne.read_labels_from_annot(
    subject='fsaverage',      
    parc='aparc.a2009s',        #HCPMMP1_combined,aparc.a2009s       
    hemi='both'
)

mean_amp = []
for label in labels:
    stc_label = stc.in_label(label)
    mean_amp.append(stc_label.data.mean())
    
sorted_idx = np.argsort(mean_amp)[::-1]

selected_labels = []  
for i in sorted_idx[:7]:
    selected_labels.append(labels[i])
    print(f"{labels[i].name}: {mean_amp[i]:.3f}")



