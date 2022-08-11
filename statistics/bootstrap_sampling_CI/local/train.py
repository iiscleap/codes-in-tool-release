import argparse, configparser
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from dataset import *
from models import *
from utils import *
import os,sys
import datetime
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def main(config, model_configfil, featsfil, train_fil, val_fil, outdir, init_model, init_encoder, init_classifier, save_encoder):

    # Setup device
    if config['training']['use_gpu']=='True':
        gpu_id=get_freegpu()	
        device = torch.device('cuda:'+str(gpu_id))
    else:
        device = torch.device('cpu')

    # Training dataset and dataloader
    trds_args={}
    for key in list(config['training_dataset'].keys()):
        val = config['training_dataset'][key]
        trds_args[key] = convertType(val)

    if config['default']['datasetModule']=='DiCOVA_2_Dataset':
        dataset_class = DiCOVA_2_Dataset
    else:
        raise ValueError('Unexpected dataset module')

    augmentation_args={}
    if config.has_section('augmentation'):
        for key in list(config['augmentation'].keys()):
            val = config['augmentation'][key]
            augmentation_args[key] = convertType(val)

    train_dataset = dataset_class({'file_list':featsfil, 
                    'label_file': train_fil, 
                    'device': device,
                    'shuffle': True,
                    'dataset_args':trds_args,
                    'augmentation_args':augmentation_args})

    # Validation dataset
    vlds_args={}
    for key in list(config['validation_dataset'].keys()):
        val = config['validation_dataset'][key]
        vlds_args[key] = convertType(val)

    validation_dataset = dataset_class({ 'file_list':featsfil, 
                    'label_file': val_fil, 
                    'device': device,
                    'dataset_args':vlds_args })

    #
    n_epochs = int(config['training']['epochs'])

    #
    model_args = {'input_dimension':train_dataset.egs[0][0].shape[1]}
    if model_configfil != 'None':
        temp = configparser.ConfigParser()
        temp.read(model_configfil)
        for key in temp['default'].keys():
            model_args[key]=convertType(temp['default'][key])
    else:
        raise ValueError('Expected an architecture')

    model = getNet(model_args['architecture'])(model_args)

    if init_model is not None:
        model.load_state_dict(torch.load(init_model,map_location='cpu'))
        print('initialized using the model {}'.format(init_model))

    if init_encoder is not None:
        encoder_weights = torch.load(init_encoder,map_location='cpu')
        model.init_encoder(encoder_weights)
        print("Encoder initialized using "+init_encoder)

    if init_classifier is not None:
        classifier_weights = torch.load(init_classifier,map_location='cpu')
        model.init_classifier(classifier_weights)
        print("Encoder initialized using "+init_classifier)

    model = model.to(device)
    print(model.parameters)

    training_loss = []
    validation_loss = []
    AUCs = []

    val_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=int(config['training']['batch_size']), shuffle=False, collate_fn=validation_dataset.collate)
    val_loss , AUC, _, _ = validate(model,val_data_loader)
    print("Initial validation loss: {}".format(round(val_loss,6)))
    print("Initial metrics: AUC {:.3f}".format(AUC,))
    sys.stdout.flush()
    sys.stderr.flush()
    best_validation_loss = 1e8
    best_validation_auc = 0
    lr = float(config['training']['learning_rate'])

    optimizer=optim.Adam(model.parameters(), lr=lr, weight_decay=float(config['training']['weight_decay']), amsgrad=True)

    print(optimizer)

    if config['training']['lr_scheme']=='ExponentialLR':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=float(config['training']['learning_rate_decay']))
    elif config['training']['lr_scheme']=='ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=float(config['training']['learning_rate_decay']))
    elif config['training']['lr_scheme']=='custom':
        min_val_loss = 1e8
        gt_min_val_cnt = 0
    else:
        raise ValueError('Unknown LR scheme')

    n_resets=0
    for epoch in range(1,n_epochs+1):			
        print(datetime.datetime.now())

        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(config['training']['batch_size']), shuffle=True, collate_fn=train_dataset.collate)

        val_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=int(config['training']['batch_size']), shuffle=False, collate_fn=validation_dataset.collate)

        train_loss = train(model, train_data_loader, optimizer, epoch)			
        training_loss.append(train_loss)

        val_loss ,AUC, _, _ = validate(model,val_data_loader)
        print("Training loss at epoch {} is {}, lr={}".format(epoch,round(train_loss,6),optimizer.param_groups[0]['lr']))
        print("Validation loss at epoch: {} is {}".format(epoch,round(val_loss,6)))
        print("Val metrics: AUC {:.3f}".format(AUC))
        validation_loss.append(val_loss)

        if config['training']['lr_scheme'] == 'custom':
            if val_loss <=min_val_loss:
                min_val_loss=val_loss
                gt_min_val_cnt = 0
            else:
                gt_min_val_cnt+=1
                if gt_min_val_cnt>2:
                    min_val_loss=val_loss
                    for gid,g in enumerate(optimizer.param_groups):
                        g['lr']*=float(config['training']['learning_rate_decay'])
                    if g['lr']<1e-8: break
        else:
            scheduler.step(val_loss)
        AUCs.append(AUC)


        # Write models to output and save loss and AUC plots
        if not os.path.exists(outdir+'/models'):
            os.mkdir(outdir+'/models')

        if not os.path.exists(outdir+'/lossplots'):
            os.mkdir(outdir+'/lossplots')
        loss_curve_path = outdir+'/lossplots/lossplot.png'
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(training_loss, label='train')
        ax.plot(validation_loss, label='valid')
        lines, labels = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc='best')
        plt.savefig(loss_curve_path)
        plt.close()

        loss_curve_path = outdir+'/lossplots/AUC.png'
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(AUCs, label='valid')
        lines, labels = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc='best')
        plt.savefig(loss_curve_path)
        plt.close()

        # Reset the model if nan is encountered in the loss
        if np.isnan(train_loss):
            n_resets+=1
            model = getNet(model_args['architecture'])(model_args)
            model = model.to(device)
            if n_resets>5: raise ValueError('Too many resets: check model parameters')

    # Save model weights, model file and the encoder part of the model
    model_filename = outdir+'/models/final.pt'
    torch.save(model.state_dict(), model_filename)

    model_filename = outdir+'/models/final.mdl'
    torch.save(model, model_filename)

    if save_encoder:
        torch.save(model.encoder.state_dict(),outdir+'/models/encoder.pt')

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config','-c',required=True)
    parser.add_argument('--model_config','-m',required=True)
    parser.add_argument('--init',required=False,default=None)	
    parser.add_argument('--init_encoder',required=False,default=None)
    parser.add_argument('--save_encoder',required=False,default=True)
    parser.add_argument('--init_classifier',required=False,default=None)	
    parser.add_argument('--featsfil','-f',required=True)
    parser.add_argument('--trainfil','-t',required=True)
    parser.add_argument('--valfil','-v',required=True)
    parser.add_argument('--outdir','-o',required=True)	
    args = parser.parse_args()

    cfg = open(args.config).readlines()

    config = configparser.ConfigParser()
    config.read(args.config)
    main(config, args.model_config, args.featsfil,args.trainfil,args.valfil,args.outdir, args.init, args.init_encoder, args.init_classifier, args.save_encoder)
