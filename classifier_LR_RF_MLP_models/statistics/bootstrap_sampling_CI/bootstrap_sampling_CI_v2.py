#created by DebarpanB
#date 25th July, 2022

import argparse, configparser
import pandas as pd
import numpy as np
import os
import random
from local.scoring import score
from pdb import set_trace as bp

random.seed(42)
np.random.seed(42)

def scoring_AUC(reference_labels, sys_scores,specificities_chosen=[0.5,0.95]):

    thresholds=np.arange(0,1,0.0001)

    AUC, TPR, TNR = score(reference_labels,sys_scores,thresholds=thresholds)

    specificities=[]
    sensitivities=[]

    decision_thresholds = []
    for specificity_threshold in specificities_chosen:
        ind = np.where(TNR>specificity_threshold)[0]
        sensitivities.append( TPR[ind[0]])
        specificities.append( TNR[ind[0]])
        decision_thresholds.append( thresholds[ind[0]])

    # pack the performance metrics in a dictionary to save & return
    # Each performance metric (except AUC) is a array for different threshold values
    # Specificity at 90% sensitivity
    scores={'TPR':TPR,
            'FPR':1-TNR,
            'AUC':AUC,
            'sensitivity':sensitivities,
            'specificity':specificities,
            'operatingPts':decision_thresholds,
			'thresholds':thresholds}

    return AUC,sensitivities[0],specificities[0],sensitivities[1],specificities[1], decision_thresholds[1]

def draw_bs_replicates_AUC(data,size):
    """creates a bootstrap sample, computes replicates and returns replicates array"""
    # Create an empty array to store replicates
    bs_replicates = np.empty(size)
    
    # Create bootstrap replicates as much as size
    for i in range(size):
        if i%100==0: print(i)
        # Create a bootstrap sample
        bs_sample = data[np.random.choice(data.shape[0], data.shape[0], replace=True), :]
        # Get bootstrap replicate and append to bs_replicates
        AUC,_,_,_,_,_ = scoring_AUC(list(bs_sample[:,2]), list(bs_sample[:,1]))
        bs_replicates[i] = AUC
    
    return bs_replicates

def main(samples, input_dir, score_file, label_file, stat_measure, confidence_interval):
        df_score = pd.read_csv(os.path.join(input_dir, score_file), header=None, delimiter=' ')
        df_label = pd.read_csv(label_file, header=None, delimiter=' ')
        df_score.columns = ['id', 'score']
        df_label.columns = ['id', 'label']
        #bp()
        df_score['label'] = [1 if df_label[df_label.id==i].label.values[0]=='p' else 0 for i in df_score.id.values]

        if stat_measure=='AUC':
            true_AUC,sens1,spec1,sens2,spec2,_ = scoring_AUC(list(df_score.label.values), list(df_score.score.values))
            print(f'True AUC: {true_AUC,sens1,spec1,sens2,spec2}')

            bs_replicates_AUC = draw_bs_replicates_AUC(df_score.to_numpy(), int(samples))

            # Get the corresponding values of CI
            conf_interval = np.percentile(bs_replicates_AUC,[100-float(confidence_interval), float(confidence_interval)])

            # Print the interval
            print("The confidence interval(x100): ",[round(i*100, 1) for i in conf_interval])
            print(f'Mean(x100): {round(np.mean(bs_replicates_AUC)*100, 1)}')
            print(f'Std. Dev.(x100): {round(np.std(bs_replicates_AUC)*100, 1)}')

        else:
            print('Unknown stat measure! exiting.')
            exit()

        
        


if __name__=='__main__':

        parser = argparse.ArgumentParser()
        parser.add_argument('--samples','-s',required=True)
        parser.add_argument('--input_dir','-i',required=True)
        parser.add_argument('--score_file_name','-n',required=True)
        parser.add_argument('--label_file_name','-l',required=True)
        parser.add_argument('--stat_measure','-m',required=True)
        parser.add_argument('--confidence_interval','-c',required=True)
        args = parser.parse_args()

        main(args.samples, args.input_dir, args.score_file_name, args.label_file_name, args.stat_measure, args.confidence_interval)