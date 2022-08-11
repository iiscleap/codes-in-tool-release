#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 14:32:50 2021

@author: srikanthr
"""
import sys
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scoring import *
from utils import *
import os
import random
random.seed(0)
np.random.seed(0)

#%%
datadir=sys.argv[1]
outdir=sys.argv[2]
if not os.path.exists(outdir):
    os.mkdir(outdir)

#%%
symptoms_keys = open(datadir+"/symptoms").readlines()
symptoms_keys =[line.strip() for line in symptoms_keys]

all_data = pd.read_csv('{}/dev_metadata.csv'.format(datadir))

train_labels = {}
temp = open(f'{datadir}/train').readlines()
for line in temp:
    key, val = line.strip().split()
    train_labels[key] = val
train_ids = list(train_labels.keys())
train_data = all_data[all_data.id.isin(train_ids)]
train_data = train_data[["id"]+symptoms_keys]
train_data.reset_index()

for key in symptoms_keys:
    train_data[key].fillna(False,inplace=True)
        
FL = []
for idx,item in train_data.iterrows():
    pid = item['id']
    f = [item[key]*1 for key in symptoms_keys]
    f.append(train_labels[pid])
    FL.append(f)    
FL = np.array(FL)
    
classifier = DecisionTreeClassifier(criterion='gini',
                                    min_samples_leaf=25,
                                    class_weight='balanced', 
                                    random_state=42)
classifier.fit(FL[:,:-1],FL[:,-1])
with open(outdir+'/model.pkl','wb') as f:
    pickle.dump({'scaler':None,'classifier':classifier},f)
