# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:54:36 2020

@author: Talha
"""
from classifiers import dtree_param_selection,svc_param_selection
import numpy as np
def feature_selector(feature,label,ch,param_selection):

    feature_list=['mean','std','ptp','var','minim','maxim','argmin','argmax','mean_square','rms','abs_diffs_signal','skewness','kurtosis','zero_crossing',
    'app_epy','perm_epy','svd_epy','spectral_epy','sample_epy','katz','higuchi','petrosian','teager',
    'hjorth_mobility','hjorth_complexity']
    
    
    feature_selected=[]
    feat_non_sorted=[]
    score_non_sorted=[]
    for i ,j in zip(range(0,feature.shape[1],ch),feature_list):
      acc=param_selection(feature[:,i:i+ch],label)
      #print(j," : ",acc)
      feat_non_sorted.append(j)
      score_non_sorted.append(acc)
      #if acc>0.67:
       # feature_selected.append(j)
    
    score,feat  = zip(*sorted(zip(score_non_sorted, feat_non_sorted),reverse=True))
    
    #skf = StratifiedKFold(n_splits=10, random_state=2020, shuffle=False)
    acc=0
    deleted_item=[]
    for i in range(1,20):
      feature_selected=list(feat[:i])
      # feature_selected=del_item(feature_selected,deleted_item)
    
      X_good=[]
      for key,val in zip(feature_list,range(0,feature.shape[1],ch)):
          for fe in feature_selected:
              if key==fe:     
                  #print('key',key,'value',val,":",val+ch) 
                  X_good.append(feature[:,val:val+ch])
      good_feature=np.concatenate((X_good),axis=1)
      #good_feature=np.concatenate((good_feature,non_eeg),1)
    
    
      acc_new=param_selection(good_feature,label)
      print(i,' : ', acc_new)
    return feat