# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 11:40:02 2020

@author: Talha
"""
PD_ID=[804,805,806,807,808,809,810,811,813,814,815,816,817,818,819,820,821,822,823,824,825,826,827,828,829]
CTL_ID=[906,903,8060,893,909,911,895,913,900,896,899,914,910,890,891,912,905,904,892,902,901,898,897,8070,907]

col=['Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8', 'CPz']
import scipy.io
import re
from glob import glob
import numpy  as np
import matplotlib.pyplot as plt
from scipy import signal
from feature_extractor import concatenate_features
nyq = 0.5 * 500
l=0.2
low = l / nyq
high = 60 / nyq
b, a = signal.butter(4, [low,high], 'band')


path=glob('D:/Datasets/EEG dataset/parkinson/edf data/PD Oddball Data/*.mat')

PD_data=[]
#PD_channels=[]
#PD_time=[]

CTL_data=[]
#CTL_channels=[]
#CTL_time=[]

def read_file(file_name):
    file=scipy.io.loadmat(file_name)['EEG']
    data=file['data'][0][0]
    time=file['times'][0][0][0]
    #fs=file['srate'][0][0][0]
    channel=file['chanlocs']
    channels=[]
    for i in range(len(channel[0][0][0])):
        channels.append(str(channel[0][0][0][i][0][0]))
    #return data[:,:,0:180],time,channels
    return data.T[0:180,:,:]



for i in path:
    if int(i[i.find('_')+1:i.find('_')+2])!=2:
        if int(re.findall(r"\d{3}", i)[0]) in PD_ID:
            PD_data.append(read_file(i))
        elif int(re.findall(r"\d{3}", i)[0]) in CTL_ID:
            CTL_data.append(read_file(i))


CTL_data=np.array(CTL_data)
PD_data=np.array(PD_data)
data=np.concatenate((np.mean(CTL_data,axis=1),np.mean(PD_data,axis=1)))
X=np.concatenate(data)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
plt.figure(figsize=(20,10))
pca = PCA().fit(X)
plt.axhline(0.95)
plt.axvline(9)
#Plotting the Cumulative Summation of the Explained Variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Parkinson Disease Dataset')
plt.show()
        
data = pd.DataFrame.from_records(X)
data.columns=col
X_1=data.var()
idx=X_1.sort_values()[-30::].index
print(idx)
ch=len(idx)

indx = [i for i, val in enumerate(col) if val in set(idx) ] 
  



X_data=data[idx].values.reshape(-1,2000,ch)
X_data.shape
PD_data=PD_data[:,:,:,indx]


PD_features=[]
for i in PD_data:
    feature=[]
    for j in i:
        data=signal.filtfilt(b, a, j.T).T        
        feature.append(concatenate_features(data))
    PD_features.append(np.mean(np.array(feature),axis=0))

CTL_data=CTL_data[:,:,:,indx]

CTL_features=[]
for i in CTL_data:
    feature=[]
    for j in i:
        data=signal.filtfilt(b, a, j.T).T        
        feature.append(concatenate_features(data))
    CTL_features.append(np.mean(np.array(feature),axis=0))

PD_features_array=np.array(PD_features)    
CTL_features_array=np.array(CTL_features)    
    

features=np.concatenate((PD_features_array,CTL_features_array))
label=np.concatenate((np.zeros(len(PD_features_array)),np.ones(len(CTL_features_array))))
np.save('features_30_channels.npy',features)
np.save('label_30_channels.npy',label)

from sklearn.preprocessing import scale,maxabs_scale
feature=scale(features)
feature.shape,label.shape

#from sklearn.decomposition import PCA
#pca=PCA(5)
#feature=pca.fit_transform(feature)




from classifiers import dtree_param_selection,svc_param_selection

svc_param_selection(feature,label)
print(dtree_param_selection(feature,label))



#PD_ERP=[]        

