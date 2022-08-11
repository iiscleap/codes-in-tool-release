import argparse
import pickle
import numpy as np
from models import *
from pdb import set_trace as bp

def main(modelfil,file_list,featsfil,outfil):

	# load model
	model = pickle.load(open(modelfil,'rb'))

	file_list = open(file_list).readlines()
	file_list = [line.strip().split() for line in file_list]
	# 
	feats_list = open(featsfil).readlines()
	feats_list = [line.strip().split() for line in feats_list]
	feats={}
	for fileId,file_path in feats_list:
		feats[fileId] = file_path

	scores={}
	for fileId,_ in file_list:
		F = pickle.load(open(feats[fileId],'rb'))
		#bp()
		score = model.validate([F])
		score = np.mean(score[0],axis=0)[1]
		scores[fileId]=score
	#bp()
	with open(outfil,'w') as f:
		for item in scores:
			f.write(item+" "+str(scores[item])+"\n")
		f.close()

if __name__=='__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--modelfil','-m',required=True)
	parser.add_argument('--featsfil','-f',required=True)
	parser.add_argument('--file_list','-i',required=True)
	parser.add_argument('--outfil','-o',required=True)	
	args = parser.parse_args()

	main(args.modelfil, args.file_list, args.featsfil, args.outfil)
