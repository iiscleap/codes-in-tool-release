import argparse, configparser
import pickle
import logging
import torch
import torch.nn as nn
from scoring import *
import numpy as np
import matplotlib.pyplot as plt
from models import *
from utils import *

def main(modelfil,file_list,outfil,config,feature_config):
	''' Script to do inference using trained model
	config, feature_config: model coonfiguration and feature configuration files
	modelfil: trained model stored as a .mdl file
	file_list: list of files as in "<id> <file-path>" format
	outfil: output file, its content will be "<id> <probability-score>"
	'''

	# Load model, use CPU for inference
	model = torch.load(modelfil,map_location='cpu')
	device = torch.device('cpu')
	model = model.to(device)
	model.eval()

	# Feature extractor
	FE = feature_extractor(feature_config['default'])

	# Loop over all files
	file_list = open(file_list).readlines()
	file_list = [line.strip().split() for line in file_list]	 
	scores={}
	for fileId,path in file_list:
		
		# Prepare features
		try:
			F = FE.extract(path)	
		except:
			print('failed for '+fileId)
			continue
		if config['training_dataset'].get('apply_mean_norm',False): F = F - torch.mean(F,dim=0)
		if config['training_dataset'].get('apply_var_norm',False): F = F / torch.std(F,dim=0)
		feat = F.to(device)

		# Input mode
		seg_mode = config['training_dataset'].get('mode','file')
		if seg_mode=='file':
			feat = [feat]
		elif seg_mode=='segment':
			segment_length = int(config['training_dataset'].get('segment_length',300))
			segment_hop = int(config['training_dataset'].get('segment_hop',10))
			feat = [feat[i:i+segment_length,:] for i in range(0,max(1,F.shape[0]-segment_length),segment_hop)]
		else:
			raise ValueError('Unknown eval model')
		with torch.no_grad():
			output = model.predict_proba(feat)

		# Average the scores of all segments from the input file
		scores[fileId]= sum(output)[0].item()/len(output)

	# Write output scores
	with open(outfil,'w') as f:
		for item in scores:
			f.write(item+" "+str(scores[item])+"\n")

if __name__=='__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--modelfil','-m',required=True)
	parser.add_argument('--config','-c',required=True)
	parser.add_argument('--feature_config','-f',required=True)
	parser.add_argument('--file_list','-i',required=True)
	parser.add_argument('--outfil','-o',required=True)	
	args = parser.parse_args()

	config = configparser.ConfigParser()
	config.read(args.config)

	feature_config = configparser.ConfigParser()
	feature_config.read(args.feature_config)

	main(args.modelfil, args.file_list, args.outfil, config, feature_config)
