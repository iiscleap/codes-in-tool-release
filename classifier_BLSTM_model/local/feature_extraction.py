import pickle
from utils import *
import argparse,configparser
import os
#%%
def main(config,filelist,outdir):
	''' This function just loops over all the files in the file list
	The file list is in the form:
	<id1> <file_path>
	<id2> <file_path>
	......
	<idN> <file_path>
	
	Extracted features are stored in outdir as pkl file with id*.pkl as the file name
	'''
	temp = open(filelist).readlines()
	filepaths={}
	for line in temp:
		idx,path = line.strip().split()
		filepaths[idx]=path
	FE = feature_extractor(config['default'])
	featlist = []
	for item in filepaths:
		outname = '{}/{}.pkl'.format(outdir,item)
		if not os.path.exists(outname):
			F = FE.extract(filepaths[item])
			with open(outname,'wb') as f: 
				pickle.dump(F,f)	
		featlist.append('{} {}/{}.pkl'.format(item,outdir,item))

	with open('{}/feats.scp'.format(outdir),"w") as f:
		for item in featlist: 
			f.write(item+'\n')

if __name__=='__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--config','-c',required=True)
	parser.add_argument('--filelist','-f',required=True)
	parser.add_argument('--outdir','-o',required=True)	
	args = parser.parse_args()

	config = configparser.ConfigParser()
	config.read(args.config)

	main(config, args.filelist, args.outdir)
