#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 14:32:50 2021

@author: srikanthr
"""

import os,sys
from scoring import *
from utils import *

def do_score_fusion(ffolders,outscores,normalize=False,weights=None):
#%%
    ''' Function to do score fusion
    Normalizes the scores if normalize is set to True
    The scores are averaged and written to outscores
    '''
    scores={}
    for i in range(len(ffolders)):
    	temp=to_dict(ffolders[i])
    	if normalize:
    		# Map the scores between 0-1
    		min_score = temp[min(temp,key=temp.get)]
    		max_score = temp[max(temp,key=temp.get)]
    		score_range = max_score-min_score
    		for item in temp:
    			temp[item] = (temp[item]-min_score)/score_range
    	scores[i] = temp
    # Fused score is an average of the test scores
    fused_scores=ArithemeticMeanFusion([scores[i] for i in range(len(ffolders)) ],weights)
    with open(outscores,'w') as f:
    	for item in fused_scores: f.write('{} {}\n'.format(item,fused_scores[item]))    

#%%
normalize = False if sys.argv[-1]=='False' else True
outscores=sys.argv[-2]
ffolders = [sys.argv[i] for i in range(1,len(sys.argv)-2)]
R = do_score_fusion(ffolders,outscores,normalize=normalize)
