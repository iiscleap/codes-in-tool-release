import os
import random
import string
import subprocess
import numpy as np
import torch.nn as nn
import torch, torchaudio
from sklearn.metrics import auc

from scoring import *

from pdb import set_trace as bp


#%%
def to_dict(filename):
	''' Convert a file with "key value" pair data to dictionary with type conversion'''
	data = open(filename).readlines()
	D = {}
	for line in data:
		key,val=line.strip().split()
		try:
			val = int(val)
		except:
			try:
				val = float(val)
			except:
				pass
		D[key] = val
	return D

def ArithemeticMeanFusion(scores,weight=None):
	'''
	Artihmetic mean fusion of scores
	scores: is a list of dictionaries with scores as key,value pairs
	weights: list of weights
	'''
	if weight==None:
		weight=[1/len(scores) for i in range(len(scores))]
	assert len(weight)==len(scores)

	if len(scores)==1: 
		# Nothing to fuse
		return scores
	else:
		keys = set(scores[0].keys())
		# get common participants
		for i in range(1,len(scores)):
			if keys != set(scores[i].keys()): 
				print("WARNING: Expected all scores to come from same set of participants")
				keys = keys.intersection(set(scores[i].keys()))
		# do weighted sum for each participant
		fused_scores={}
		for key in keys:
			s = [weight[i]*scores[i][key] for i in range(len(scores))]
			fused_scores[key]=sum(s)/sum(weight)
		return fused_scores

#%%
def compute_SAD(sig,fs,threshold=0.0001,sad_start_end_sil_length=100, sad_margin_length=50):
	''' Compute threshold based sound activity '''
	# Leading/Trailing margin
	sad_start_end_sil_length = int(sad_start_end_sil_length*1e-3*fs)
	# Margin around active samples
	sad_margin_length = int(sad_margin_length*1e-3*fs)

	sample_activity = np.zeros(sig.shape)
	sample_activity[np.power(sig,2)>threshold] = 1
	sad = np.zeros(sig.shape)
	for i in range(sample_activity.shape[1]):
		if sample_activity[0,i] == 1: sad[0,i-sad_margin_length:i+sad_margin_length] = 1
	sad[0,0:sad_start_end_sil_length] = 0
	sad[0,-sad_start_end_sil_length:] = 0
	return sad

class feature_extractor():
	''' Class for feature extraction 
	args: input arguments dictionary
	Mandatory arguments: resampling_rate, feature_type, window_size, hop_length
	For MFCC: f_max, n_mels, n_mfcc
	For MelSpec/logMelSpec: f_max, n_mels		
	Optional arguments: compute_deltas, compute_delta_deltas
	'''

	def __init__(self,args):

		self.args=args	
		self.resampling_rate = int(self.args['resampling_rate'])
		assert (args['feature_type'] in ['MFCC', 'MelSpec', 'logMelSpec', 'ComParE_2016_llds', 'ComParE_2016_voicing','ComParE_2016_spectral',
										 'ComParE_2016_mfcc','ComParE_2016_rasta','ComParE_2016_basic_spectral','ComParE_2016_energy'
										]),('Expected the feature_type to be MFCC / MelSpec / logMelSpec / ComParE_2016')

		if self.args['feature_type'] == 'MFCC':
			self.feature_transform = torchaudio.transforms.MFCC(sample_rate=self.resampling_rate, 
		                                                        n_mfcc=int(self.args['n_mfcc']), 
		                                                        melkwargs={
		                                                            'n_fft': int(float(self.args['window_size'])*1e-3*self.resampling_rate), 
		                                                            'n_mels': int(self.args['n_mels']), 
		                                                            'f_max': int(self.args['f_max']), 
		                                                            'hop_length': int(float(self.args['hop_length'])*1e-3*self.resampling_rate)})
		elif self.args['feature_type'] in ['MelSpec','logMelSpec'] :
			self.feature_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.resampling_rate,
		                                                            n_fft= int(float(self.args['window_size'])*1e-3*self.resampling_rate), 
		                                                            n_mels= int(self.args['n_mels']), 
		                                                            f_max= int(self.args['f_max']), 
		                                                            hop_length= int(float(self.args['hop_length'])*1e-3*self.resampling_rate))
		elif 'ComParE_2016' in self.args['feature_type']:
			import opensmile
			self.feature_transform = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
																feature_level=opensmile.FeatureLevel.LowLevelDescriptors, sampling_rate=self.resampling_rate)
		else:
			raise ValueError('Feature type not implemented')

	def _read_audio(self,filepath):
		''' This code does the following:
		1. Read audio, 
		2. Resample the audio if required, 
		3. Perform waveform normalization,
		4. Compute sound activity using threshold based method
		5. Discard the silence regions
		'''
		s,fs = torchaudio.load(filepath)
		if fs != self.resampling_rate: 
			s,fs = torchaudio.sox_effects.apply_effects_tensor(s,fs,[['rate',str(self.resampling_rate)]])           
		if s.shape[0]>1:
			s = s.mean(dim=0).unsqueeze(0)
		s = s/torch.max(torch.abs(s))
		sad = compute_SAD(s.numpy(),self.resampling_rate)
		s = s[np.where(sad==1)]     
		return s,fs

	def _do_feature_extraction(self,s):
		''' Feature preparation
		Steps:
		1. Apply feature extraction to waveform
		2. Convert amplitude to dB if required
		3. Append delta and delta-delta features
		'''

		if 	'ComParE_2016' in self.args['feature_type']:
			
			# get a random string
			file_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
			while os.path.exists(file_name):
				file_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
			torchaudio.save(file_name + '.wav', s, sample_rate=self.resampling_rate)
			F = self.feature_transform.process_file(file_name + '.wav')

			# columns based selection
			os.remove(file_name + '.wav')

			# feature subsets
			feature_subset = {}
			if self.args['feature_type'] == 'ComParE_2016_voicing':
				feature_subset['subset'] = ['F0final_sma', 'voicingFinalUnclipped_sma', 'jitterLocal_sma', 'jitterDDP_sma', 'shimmerLocal_sma', 'logHNR_sma']

			if self.args['feature_type'] == 'ComParE_2016_energy':
				feature_subset['subset'] = ['audspec_lengthL1norm_sma', 'audspecRasta_lengthL1norm_sma','pcm_RMSenergy_sma', 'pcm_zcr_sma']

			if self.args['feature_type'] == 'ComParE_2016_spectral':
				feature_subset['subset'] = [ 'audSpec_Rfilt_sma[0]', 'audSpec_Rfilt_sma[1]', 'audSpec_Rfilt_sma[2]', 'audSpec_Rfilt_sma[3]',
				'audSpec_Rfilt_sma[4]', 'audSpec_Rfilt_sma[5]', 'audSpec_Rfilt_sma[6]','audSpec_Rfilt_sma[7]', 'audSpec_Rfilt_sma[8]', 'audSpec_Rfilt_sma[9]',
				'audSpec_Rfilt_sma[10]', 'audSpec_Rfilt_sma[11]','audSpec_Rfilt_sma[12]', 'audSpec_Rfilt_sma[13]',
				'audSpec_Rfilt_sma[14]', 'audSpec_Rfilt_sma[15]','audSpec_Rfilt_sma[16]', 'audSpec_Rfilt_sma[17]',
				'audSpec_Rfilt_sma[18]', 'audSpec_Rfilt_sma[19]', 'audSpec_Rfilt_sma[20]', 'audSpec_Rfilt_sma[21]',
				'audSpec_Rfilt_sma[22]', 'audSpec_Rfilt_sma[23]','audSpec_Rfilt_sma[24]', 'audSpec_Rfilt_sma[25]',
				'pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma','pcm_fftMag_spectralRollOff25.0_sma',
				'pcm_fftMag_spectralRollOff50.0_sma', 'pcm_fftMag_spectralRollOff75.0_sma','pcm_fftMag_spectralRollOff90.0_sma', 'pcm_fftMag_spectralFlux_sma',
				'pcm_fftMag_spectralCentroid_sma', 'pcm_fftMag_spectralEntropy_sma','pcm_fftMag_spectralVariance_sma', 'pcm_fftMag_spectralSkewness_sma',
				'pcm_fftMag_spectralKurtosis_sma', 'pcm_fftMag_spectralSlope_sma','pcm_fftMag_psySharpness_sma', 'pcm_fftMag_spectralHarmonicity_sma',
				'mfcc_sma[1]', 'mfcc_sma[2]', 'mfcc_sma[3]', 'mfcc_sma[4]','mfcc_sma[5]', 'mfcc_sma[6]', 'mfcc_sma[7]', 'mfcc_sma[8]',
				'mfcc_sma[9]', 'mfcc_sma[10]', 'mfcc_sma[11]', 'mfcc_sma[12]','mfcc_sma[13]', 'mfcc_sma[14]']

			if self.args['feature_type'] == 'ComParE_2016_mfcc':
				feature_subset['subset'] = ['mfcc_sma[1]', 'mfcc_sma[2]', 'mfcc_sma[3]', 'mfcc_sma[4]','mfcc_sma[5]', 'mfcc_sma[6]', 'mfcc_sma[7]', 'mfcc_sma[8]',
				'mfcc_sma[9]', 'mfcc_sma[10]', 'mfcc_sma[11]', 'mfcc_sma[12]','mfcc_sma[13]', 'mfcc_sma[14]']

			if self.args['feature_type'] == 'ComParE_2016_rasta':
				feature_subset['subset'] = ['audSpec_Rfilt_sma[0]', 'audSpec_Rfilt_sma[1]', 'audSpec_Rfilt_sma[2]', 'audSpec_Rfilt_sma[3]',
				'audSpec_Rfilt_sma[4]', 'audSpec_Rfilt_sma[5]', 'audSpec_Rfilt_sma[6]','audSpec_Rfilt_sma[7]', 'audSpec_Rfilt_sma[8]', 'audSpec_Rfilt_sma[9]',
				'audSpec_Rfilt_sma[10]', 'audSpec_Rfilt_sma[11]','audSpec_Rfilt_sma[12]', 'audSpec_Rfilt_sma[13]',
				'audSpec_Rfilt_sma[14]', 'audSpec_Rfilt_sma[15]','audSpec_Rfilt_sma[16]', 'audSpec_Rfilt_sma[17]',
				'audSpec_Rfilt_sma[18]', 'audSpec_Rfilt_sma[19]', 'audSpec_Rfilt_sma[20]', 'audSpec_Rfilt_sma[21]',
				'audSpec_Rfilt_sma[22]', 'audSpec_Rfilt_sma[23]','audSpec_Rfilt_sma[24]', 'audSpec_Rfilt_sma[25]']

			if self.args['feature_type'] == 'ComParE_2016_basic_spectral':
				feature_subset['subset'] = ['pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma','pcm_fftMag_spectralRollOff25.0_sma',
				'pcm_fftMag_spectralRollOff50.0_sma', 'pcm_fftMag_spectralRollOff75.0_sma','pcm_fftMag_spectralRollOff90.0_sma', 'pcm_fftMag_spectralFlux_sma',
				'pcm_fftMag_spectralCentroid_sma', 'pcm_fftMag_spectralEntropy_sma','pcm_fftMag_spectralVariance_sma', 'pcm_fftMag_spectralSkewness_sma',
				'pcm_fftMag_spectralKurtosis_sma', 'pcm_fftMag_spectralSlope_sma','pcm_fftMag_psySharpness_sma', 'pcm_fftMag_spectralHarmonicity_sma']

			if self.args['feature_type'] == 'ComParE_2016_llds':
				feature_subset['subset'] = list(F.columns)

			F = F[feature_subset['subset']].to_numpy()
			F = np.nan_to_num(F)
			F = torch.from_numpy(F).T

		if self.args['feature_type'] == 'MelSpec': 
			F = self.feature_transform(s)		

		if self.args['feature_type'] == 'logMelSpec': 
			F = self.feature_transform(s)		
			F = torchaudio.functional.amplitude_to_DB(F,multiplier=10,amin=1e-10,db_multiplier=0)

		if self.args['feature_type'] == 'MFCC': 
			F = self.feature_transform(s)		

		if self.args.get('compute_deltas',False) == 'True' :
			FD = torchaudio.functional.compute_deltas(F)
			F = torch.cat((F,FD),dim=0)

		if self.args.get('compute_delta_deltas',False) == 'True':
			FDD = torchaudio.functional.compute_deltas(FD)
			F = torch.cat((F,FDD),dim=0)
		return F.T

	def extract(self,filepath):
		''' Interface to other codes for this class
		Steps:
		1. Read audio
		2. Do feature extraction
		'''
		self.audio_path = filepath
		s,fs = self._read_audio(filepath)
		return self._do_feature_extraction(s)

#%%
def convertType(val):
	def subutil(val):
		try: 
			val = int(val)
		except:
			try:
				val = float(val)
			except:
				if val in ['True', 'TRUE', 'true']:
					val = True
				elif val in ['False','FALSE','false']:
					val = False
				elif val in ['None']:
					val = None
				else:
					val = val
		return val

	if ',' in val:
		val = val.split(',')
		val = [subutil(item) for item in val]
	else:
		val = subutil(val)
	return val

#%%
def get_freegpu():
	A,_= subprocess.Popen('nvidia-smi --format=csv,noheader --query-gpu=utilization.gpu,memory.used,index',shell=True,stdout=subprocess.PIPE).communicate()
	A = A.decode('utf-8').strip().split('\n')
	indices =  [int(item.split(",")[-1]) for item in A]
	order = np.argsort([int(item.split(' ')[0]) for item in A])
	return indices[order[0]]

#%%
def train(model, data_loader, optimizer, epoch, debug=False):
	''' Fuction to train for an epoch '''
	model.train()
	total_train_loss = 0
	for batch_id, (feats,labels) in enumerate(data_loader):
		optimizer.zero_grad()
		# Zero the gradients before each batch
		loss = model(feats,labels)
		# Forward pass through the net and compute loss
		loss.backward()
		# Backward pass and get gradients
		optimizer.step()
		# Update the weights
		if debug and np.isnan(loss.detach().item()):
			print('found a nan at {}'.format(batch_id))
		total_train_loss += loss.detach().item()
		# Accumulate loss for reporting
	return total_train_loss/(batch_id+1)

#%%
def validate(model, data_loader):
	''' Do validation and compute loss 
		The scores and labels are also stacked to facilitate AUC computation
	'''
	model.eval()
	y_scores, y_val = [], []
	total_val_scores = 0
	num_egs = 0
	with torch.no_grad():
		for batch_id, (feats,labels) in enumerate(data_loader):
			output = model.predict(feats)
			# Predict the score
			val_loss = model.criterion(torch.stack(output),torch.stack(labels))	
			# Compute loss
			y_val.extend([item.cpu().numpy() for item in labels])
			y_scores.extend([torch.sigmoid(item).cpu().numpy() for item in output]) 
			#Get proba
			total_val_scores +=  val_loss.detach().item()*len(labels)
			num_egs+=len(labels)
	AUC, TPR, TNR=score(y_val,y_scores)		
	# Compute AUC, TPR, TNR
	return total_val_scores/num_egs, AUC, TPR, TNR

#%%
def reset_model(model):
	for submodel in model.children():
		for layers in submodel.children():
			for layer in layers:
				if hasattr(layer, 'reset_parameters'): layer.reset_parameters()
	return model
