#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 21:15:54 2020

@author: DiCOVA Team
"""


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from pdb import set_trace as bp
#%%
class sklearnModel():
	# parent class with training and forward pass methods
	def __init__(self):
		self.classifier = None

	def run_fit(self,x_train, y_train):
		self.classifier.fit(x_train, y_train)
		return

	def validate(self,x_val):
		y_scores=[]
		for item in x_val:
			y_scores.append(self.classifier.predict_proba(item))
		return y_scores
	
#%% Logistic regression
class LR(sklearnModel):
	def __init__(self, model_args):
		super().__init__()				
		self.classifier = LogisticRegression(C=float(model_args['c']), max_iter=int(model_args['max_iter']), solver=model_args['solver'], penalty = model_args['penalty'], class_weight = model_args['class_weight'], random_state=model_args['random_state'])
    
#%% Random forest
class RF(sklearnModel):
	def __init__(self, model_args):
		super().__init__()				
		self.classifier = RandomForestClassifier(n_estimators=int(model_args['n_estimators']), class_weight = model_args['class_weight'], random_state=model_args['random_state'])

#%% Multi-layer perceptron 
class MLP(sklearnModel):
	def __init__(self, model_args): 
		super().__init__()				
		self.classifier = MLPClassifier(hidden_layer_sizes=model_args['hidden_layer_sizes'],solver=model_args['solver'],alpha=model_args['alpha'],learning_rate_init=model_args['learning_rate_init'], verbose=model_args['verbose'], activation=model_args['activation'], max_iter=int(model_args['max_iter']), random_state=model_args['random_state'])
        
class linSVM(sklearnModel):
	def __init__(self, model_args):
		super().__init__()	
		#bp()			
		self.classifier = SVC(C=model_args['C'], kernel=model_args['kernel'],class_weight=model_args['class_weight'],verbose=model_args['verbose'],probability=model_args['probability'],random_state=model_args['random_state'])

class LR(sklearnModel):
	def __init__(self, model_args):
		super().__init__()				
		self.classifier = LogisticRegression(C=float(model_args['c']), max_iter=int(model_args['max_iter']), solver=model_args['solver'], penalty = model_args['penalty'], class_weight = model_args['class_weight'], random_state=model_args['random_state'])

# %%
