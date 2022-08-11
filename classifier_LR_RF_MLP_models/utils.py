import subprocess
import numpy as np
import torch.nn as nn
import torch, torchaudio
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from scoring import *
import torch.nn.functional as F
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
		assert (args['feature_type'] in ['MFCC', 'MelSpec', 'logMelSpec']),('Expected the feature_type to be MFCC / MelSpec / logMelSpec ')

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
		F = self.feature_transform(s)
		if self.args['feature_type'] == 'logMelSpec': 
			F = torchaudio.functional.amplitude_to_DB(F,multiplier=10,amin=1e-10,db_multiplier=0)
		Fo=F
		if self.args.get('compute_deltas',False) == 'True' :
			FD = torchaudio.functional.compute_deltas(F)
			Fo = torch.cat((F,FD),dim=0)
		if self.args.get('compute_delta_deltas',False) == 'True':
			FD = torchaudio.functional.compute_deltas(F)
			FDD = torchaudio.functional.compute_deltas(FD)
			Fo = torch.cat((F,FD,FDD),dim=0)
		return Fo.T

	def extract(self,filepath):
		''' Interface to other codes for this class
		Steps:
		1. Read audio
		2. Do feature extraction
		'''
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
def mixup_data(x, y, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
	# x must be a python list containing tensors as elements
	# y must be a python list containing 0 or 1 as elements, in tensor format
    batch_size = len(x)
    #bp()
    if alpha > 0:
        #lam = np.random.beta(alpha, alpha)
        lam = torch.from_numpy(np.random.beta(alpha, alpha, (1,batch_size))[0])
    else:
        lam = 1
    lam=lam.to(device)
    index = torch.randperm(batch_size)
    try:
    	mixed_x = torch.mul(lam.reshape(-1,1,1), torch.stack(x)) + torch.mul((1 - lam).reshape(-1,1,1), torch.stack(x)[index, :])
    except:
        #bp()
        seg_dim = max([x[s].shape[0] for s in range(batch_size)])
        for s_idx in range(batch_size):
            if x[s_idx].shape[0]!=seg_dim:
                #bp()
                dim_diff = seg_dim-x[s_idx].shape[0]
                n_pad = int((dim_diff)/2)
                x[s_idx]=F.pad(input=x[s_idx], pad=(0, 0, n_pad, dim_diff-n_pad), mode='constant', value=0)
        #bp()
        mixed_x = torch.mul(lam.reshape(-1,1,1), torch.stack(x)) + torch.mul((1 - lam).reshape(-1,1,1), torch.stack(x)[index, :])
    mixed_x = mixed_x.to(torch.float)
    #bp()
    y_a, y_b = torch.stack(y), torch.stack(y)[index]
    return list(mixed_x), list(y_a), list(y_b), lam

def train(model, data_loader, optimizer, epoch, device, n_epochs, debug=False):
	''' Fuction to train for an epoch '''
	model.train()
	total_train_loss = 0
	for batch_id, (feats,labels) in enumerate(data_loader):
		optimizer.zero_grad()
        #feats_raw, labels_raw = feats.detach().clone(), labels.detach().clone()
		# Zero the gradients before each batch
		#bp()
		######## mixup ##########
		#bp()
		feats, labels_a, labels_b, lam = mixup_data(feats, labels, device, alpha=0.4)#only mixup
		#................................................
		#bp()
		'''feats_all, labels_a_all, labels_b_all, lam_all = [], [], [], []
		labels_a = labels.copy()
		labels_b = list(torch.zeros(torch.stack(labels_a).shape).to(device))
		lam = torch.ones(len(labels_a),).to(device)
		feats_all+=feats
		labels_a_all+=labels_a
		labels_b_all+=labels_b
		lam_all+=list(lam)
		feats, labels_a, labels_b, lam = mixup_data(feats, labels, device, alpha=0.4)
		feats_all+=feats
		labels_a_all+=labels_a
		labels_b_all+=labels_b
		lam_all+=list(lam)

		lam_all = torch.stack(lam_all)'''

		#................................................
		'''if epoch<n_epochs-2:
			feats, labels_a, labels_b, lam = mixup_data(feats, labels, device, alpha=0.4)
			#bp()
		else:
			#bp()
			labels_a = labels.copy()
			labels_b = list(torch.zeros(torch.stack(labels_a).shape).to(device))
			lam = torch.ones(len(labels_a),).to(device)'''
		##########################

		#print(f'lambda:{lam}')
		#bp()
		#loss = model(feats,labels)
		bp()
		loss = model(feats, labels_a, labels_b, lam, temp=0.01)
		#loss = model(feats_all, labels_a_all, labels_b_all, lam_all)
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
