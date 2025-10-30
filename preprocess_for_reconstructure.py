#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Fri Aug 16 13:37:48 2024

@author: freya 
data:
previous preprocessed Matlab data 
and template MRI

'''
import mne
import numpy as np
from mne.coreg import Coregistration
from mne.io import read_info
import h5py
from pathlib import Path

# In[1] epoch data
''' dataloading and selection'''
datadir = '/Volumes/Freya/PhD_data/attention_MEG/python_epoched_data/' # saving path
raw_dir = '/Volumes/Freya/PhD_data/attention_MEG/original_data' # path of raw data
megdata_dir = '/Volumes/Freya/PhD_data/attention_MEG/preprocessed/' # path of MATLAB preprocessed data 

folder = Path(megdata_dir) 
megdata_list = list(folder.glob('*attention_*'))
name_list = [f.name[10:12] for f in megdata_list] # subjID list 


''' read data from matlab''' 
def epoch_data(dataName,epochName,epochs,filename):
    with h5py.File(filename, 'r') as mat_file:
        meg_structure = mat_file['meg']
        epochs._data  = np.transpose(meg_structure[dataName][:],(0,2,1))
        behav_structure = mat_file['behave']
        choice     = behav_structure['choice'][:]
        grat       = behav_structure['grat_dirc'][:]
        events2    = behav_structure['motion'][:]
    
    events1 = np.equal(choice,grat).astype(int) # correction
    Ntrial = events1.shape[0]
    events[0:Ntrial,2:3] = events1
    events[0:Ntrial,1:2] = (events2+1)*10 # motion
    epochs.events = events[0:Ntrial,0:3]
    epochs.selection = np.arange(0,np.shape(epochs._data)[0],1)
    epochs.event_id = dict([('1',1), ('0',0),('10',10),('20',20)])
    epochs.drop_log = tuple(() for _ in range(Ntrial))


    ''' MEG data: filter and baseline correction''' # the filter or other progress may need to be done in matlab
    # filtered = epochs.filter(14,18) 
    # baseline = (-0.5, 0)
    # epochs = epochs.apply_baseline(baseline)
    # epochs = epochs.crop(tmin=-0.5, tmax=2)
    filename = datadir+epochName
    epochs.save(filename, overwrite=True)


for subj_n in range(26,31): # number of subject [0 31]
    ''' get the data path'''
    megdata = megdata_list[subj_n] # get preprocessed data
    print(megdata.name)
    
    folder = Path(raw_dir)
    name_idx = '*_'+name_list[subj_n]+'_*_ds2.fif'
    sample_data_raw_file = next(folder.rglob(name_idx), None) # get the raw data according to the subject ID
    print(sample_data_raw_file)
    
    ''' read raw data as template'''
    rawData = mne.io.read_raw_fif(sample_data_raw_file) #can use preload = false and load_data()to load it into memory later
    raw_temp = rawData.copy()
    rawData = rawData.pick(picks = ['meg'])
    megchName = rawData.ch_names[0::3] # trigerchannel:STI101,STI102,SYS201
    megData = rawData.pick(megchName) # can use drop to delete channels
    events = mne.find_events(raw_temp,stim_channel='STI102',shortest_event=1)
    
    # target data
    epochs = mne.Epochs(megData, events[0:2,:], tmin=-2, tmax=3, preload=True) #template for loading data , preload=True
    file_name = name_list[subj_n]+'_targ_epo.fif'
    epoch_data('data_targ',file_name,epochs,megdata)
    
    # cue data
    epochs = mne.Epochs(megData, events[0:2,:], tmin=-2, tmax=3, preload=True) #template for loading data , preload=True
    file_name = name_list[subj_n]+'_cue_epo.fif'
    epoch_data('data_cue',file_name,epochs,megdata)


# In[2] MRI template coregistration（need raw file）
'''coregistration saved as trans'''
subjects_dir = '/Users/freya/mne_data/MNE-fsaverage-data' # for MRI data
subject = 'fsaverage' # subject name for mri data 
datadir = '/Volumes/Freya/PhD_data/attention_MEG/python_fwd_data/'

subj_ID = '40'
folder = Path(raw_dir)
name_idx = '*_'+subj_ID+'_*_ds2.fif'
# get the raw data according to the subject ID
sample_data_raw_file = next(folder.rglob(name_idx), None)
print(sample_data_raw_file)

info = read_info(sample_data_raw_file)  # read info from MEG
plot_kwargs = dict(
    subject=subject,
    subjects_dir=subjects_dir,
    surfaces="head-dense",
    dig=True,
    eeg=[],
    meg="sensors",
    show_axes=True,
    coord_frame="meg",
)
view_kwargs = dict(azimuth=45, elevation=90,
                   distance=0.6, focalpoint=(0.0, 0.0, 0.0))

# coregistration
fiducials = "estimated"  # get fiducials from fsaverage
coreg = Coregistration(
    info, subject, subjects_dir=subjects_dir, fiducials='auto')
# fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)
coreg.fit_icp(n_iterations=200, lpa_weight=1.0, nasion_weight=15.0, rpa_weight=1.0,
              hsp_weight=1.0, eeg_weight=1.0, hpi_weight=20.0,  verbose=True)
fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)

coreg.omit_head_shape_points(distance=5.0 / 1000)
dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
print(
    f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "
    f"/ {np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
)

# save coregistration
filename = datadir+'attention_'+subj_ID+'-trans.fif'
mne.write_trans(filename, coreg.trans, overwrite=True)


# In[4] forward (need raw file) 
src = mne.read_source_spaces('/Users/freya/Study/self-learning/python/MEG_example/average-src.fif')  
fwddir = '/Volumes/Freya/PhD_data/attention_MEG/python_fwd_data/'

subj_ID = '39'
folder = Path(raw_dir)
name_idx = '*_'+subj_ID+'_*_ds2.fif'
# get the raw data according to the subject ID
sample_data_raw_file = next(folder.rglob(name_idx), None)
print(sample_data_raw_file)


filename = '*'+subj_ID+'-trans.fif' # get forward data
folder = Path(fwddir)
transdata = next(folder.rglob(filename), None)
print(transdata)
trans = mne.read_trans(transdata)

''' forward solution'''
conductivity = (0.3,)  # for single layer
model = mne.make_bem_model(
    subject=subject, ico=4, conductivity=conductivity, subjects_dir=subjects_dir
)
bem = mne.make_bem_solution(model)

fwd = mne.make_forward_solution(
    sample_data_raw_file,
    trans=trans,
    src=src,
    bem=bem,
    meg=True,
    eeg=False,
    mindist=5.0,
    n_jobs=None,
    verbose=True,
)
print(fwd)
print(f"Before: {src}")
print(f'After:  {fwd["src"]}')

leadfield = fwd["sol"]["data"]
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

filename = fwddir+'attention_'+subj_ID+'-fwd.fif'
mne.write_forward_solution(filename, fwd, overwrite=True)


