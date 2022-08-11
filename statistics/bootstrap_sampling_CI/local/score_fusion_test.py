#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 14:32:50 2021

@author: srikanthr
"""

import os,sys
from scoring import *
from utils import *

def do_score_fusion(ffolders, outscores, label_map, normalize=False, weights=None):
#%%
	''' Function to do score fusion
	Normalizes the scores if normalize is set to True
	The scores are averaged and written to outscores
	'''

	breathing_to_fusion = {}
	cough_to_fusion = {}
	speech_to_fusion = {}
	label_map = open(label_map).readlines()

	for line in label_map:
		b,c,s,f,l,g = line.strip().split()
		breathing_to_fusion[b] = f
		cough_to_fusion[c] = f
		speech_to_fusion[s] = f		
	x_to_fusion={'breathing':breathing_to_fusion, 'cough':cough_to_fusion, 'speech':speech_to_fusion}

	scores={}
	for audio in ['breathing','cough','speech']:
		temp=to_dict(ffolders[audio])
		if normalize:
			# Map the scores between 0-1
			min_score = temp[min(temp,key=temp.get)]
			max_score = temp[max(temp,key=temp.get)]
			score_range = max_score-min_score
			for item in temp:
				temp[item] = (temp[item]-min_score)/score_range
		tempx={}
		for key in temp:
			tempx[x_to_fusion[audio][key]] = temp[key]
		scores[audio] = tempx

	# Fused score is an average of the test scores
	fused_scores=ArithemeticMeanFusion([scores[i] for i in scores.keys() ],weights)
	with open(outscores,'w') as f:
		for item in fused_scores: f.write('{} {}\n'.format(item,fused_scores[item]))    

#%%
if __name__=='__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--labelmap','-l',required=True)
	parser.add_argument('--normalize','-n',required=False,default='False')
	parser.add_argument('--outscores','-o',required=True)
	parser.add_argument('--breathing','-b',required=True)
	parser.add_argument('--cough','-c',required=True)
	parser.add_argument('--speech','-s',required=True)
	args = parser.parse_args()
	args.normalize=False if args.normalize=='False' else True

	R = do_score_fusion({'breathing':args.breathing, 'cough':args.cough, 'speech':args.speech}, args.outscores, args.labelmap, normalize=args.normalize)
