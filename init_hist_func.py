import os
import matplotlib.pyplot as plt
import jax
from jax import random
import jax.numpy as np
from jax import vmap
from pdb import set_trace
import numpy
from model import generate_noise, vmap_ori_discrimination
from util import create_grating_pairs
from analysis import plot_histograms, plot_all_sig



def evaluate_accuracy(ssn_layer_pars, readout_pars, constant_pars, data, noise_ref, noise_target):

    losses, all_losses, pred_label, sig_in, sig_out, _ = vmap_ori_discrimination(ssn_layer_pars, readout_pars, constant_pars, data, noise_ref, noise_target)
    
    #Find accuracy based on predicted labels
    true_accuracy = np.sum(data['label'] == pred_label)/len(data['label'])
    
    #Find input/output to sigmoid
    sig_input = np.max(sig_in)
    sig_output = np.mean(sig_out)
    
    return true_accuracy, sig_input, sig_output

evaluate_acc_over_w_sig = vmap(evaluate_accuracy, in_axes = ({'J_2x2_m': None, 'J_2x2_s':None, 'c_E':None, 'c_I':None, 'f_E':None, 'f_I':None, 'kappa_pre':None, 'kappa_post':None}, {'w_sig':0, 'b_sig':None}, None, {'ref':None, 'target':None, 'label':None}, None, None) )
    


def initial_acc(ssn_layer_pars, readout_pars, constant_pars, stimuli_pars, list_noise, list_w_std, p = 0.7, save_fig = None, trials=100, batch_size = 100):
    
    '''
    Find initial accuracy for varying scale of w and noise levels. 
    Inputs:
        model parameters
        ranges of noise and w_std to calculate performance
        trials: number of random w_sig to generate for each parameter combination
        batch_size: number of stimuli pairs to generate per w_sig
    Outputs:
        accuracies achieved at each noise and w_std value
        low_acc : parameter combinations that give desired starting accuracy
    Plots:
        1) for each parameter combination, histogram of initial accuracies using random w_sig initialisations
        2) for each parameter combination, histogram of size of input to sigmoid
    
    '''
   
    low_acc=[]
    all_accuracies=[]
    all_sig_inputs = []
    all_sig_outputs = []
    low_acc = []
    N_neurons = 25
            
    
    for sig_noise in list_noise:
        for w_std in list_w_std:
            
            #Generate random ww_sig
            readout_pars['w_sig']= numpy.random.normal(scale = w_std, size = (trials, N_neurons)) / np.sqrt(N_neurons)
            
    
            #Create data + noise
            train_data = create_grating_pairs(stimuli_pars = stimuli_pars, n_trials =batch_size)
            noise_ref = generate_noise(sig_noise = sig_noise, batch_size =batch_size, length= readout_pars['w_sig'].shape[1]) 
            noise_target = generate_noise(sig_noise = sig_noise, batch_size = batch_size, length= readout_pars['w_sig'].shape[1]) 

            #Calculate performance and loss
            true_acc, sig_input, sig_output = evaluate_acc_over_w_sig(ssn_layer_pars, readout_pars, constant_pars, train_data, noise_ref, noise_target)
            
            #Append values to lists
            all_sig_inputs.append(sig_input)
            all_sig_outputs.append(sig_output)
            all_accuracies.append([w_std, sig_noise, true_acc])
            
            print(sig_input.max(), sig_input.mean())
            
            #Check what percentage of accuracies lie in between 45 and 55%
            criteria = ((0.45 < true_acc) & (true_acc < 0.55)).sum() /len(true_acc)

            #Save parameter combination where most accuracies are within 45-55
            if criteria >p:
                    low_acc.append([w_std, sig_noise])
            
            print('grating sig_noise = {}, scale = {}, acc (45-55% > {} ) = {}'.format(sig_noise, w_std, p, criteria))

                    
    #Make plots
    plot_histograms(all_accuracies, save_fig = save_fig)
    plot_all_sig(all_sig_inputs, axis_title = 'Max sig input', save_fig = save_fig)
    
    return all_accuracies, low_acc

