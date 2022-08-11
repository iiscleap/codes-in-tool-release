#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 

@author: srikanthr
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scoring import *
import sys
#%%
datadir = sys.argv[1]
tset = sys.argv[2]

outdir=sys.argv[3]

all_data = pd.read_csv(f'{datadir}/{tset}_metadata.csv')
all_data.g.replace('female',0,inplace=True)
all_data.g.replace('male',1,inplace=True)
all_data.g.replace('other',-1,inplace=True)

symptoms_keys = ['id'] + [line.strip() for line in open(f'{datadir}/symptoms').readlines()]

#%%
categories = {'n':0,'p':1}
 
#%%
if True:
    f = pickle.load(open(f'{outdir}/model.pkl','rb'))
    classifier = f['classifier']
    #%%
    test_list = [line.split(' ')[0] for line in open(f'{datadir}/{tset}').readlines()]
    test_data = all_data[all_data.id.isin(test_list)]
    test_data = test_data[symptoms_keys]
    test_data.reset_index()
    
    for key in test_data.keys():
        test_data[key].fillna(False,inplace=True)
        
    scores={}
    for idx,item in test_data.iterrows():
        pid = item['id']
        f = [item[key]*1 for key in symptoms_keys[1:]]
        sc=classifier.predict_proba(np.array(f,ndmin=2))
        scores[pid]=sc[0][1]
        
    with open(f'{outdir}/{tset}_scores.txt','w') as f:
        for item in scores: f.write('{} {}\n'.format(item,scores[item]))            