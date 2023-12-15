import os
import matplotlib.pyplot as plt
import time, os, json
import pandas as pd
from scipy import stats 
from tqdm import tqdm
import seaborn as sns
import jax

from jax import random
from jax.config import config 
import jax.numpy as np
from jax import vmap
import pdb
import optax
from functools import partial

import math
import csv
import time
#from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy
from training import create_data, take_log, model
from SSN_classes_jax_jit import SSN2DTopoV1_ONOFF


def test_acc_func(stimuli_pars, offset, ref_ori, J_2x2, s_2x2, c_E, c_I, w_sig, b_sig, sigma_oris, ssn_pars, grid_pars, conn_pars, gE, gI, filter_pars, conv_pars, loss_pars, sig_noise, noise_type, save=None, number_trials = 20, batch_size = 500, vmap_model = None):
    '''
    Given network parameters, function generates random trials of data and calculates the accuracy per batch. 
    Input: 
        network parameters, number of trials and batch size of each trial
    Output:
        histogram of the accuracies 
    
    '''
    if vmap_model ==None:
        vmap_model = vmap(model, in_axes = (None, None, None, None, None, None, None, None, None, None, None, None, None, {'ref':0, 'target':0, 'label':0}, None, None, None, None, None) )
    
    all_accs = []
    
    ssn=SSN2DTopoV1_ONOFF(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars, filter_pars=filter_pars, J_2x2=J_2x2, s_2x2=s_2x2, gE=gE, gI=gI, sigma_oris=sigma_oris)
    ssn_ori_map = ssn.ori_map
    print(w_sig)
    
    logJ_2x2 =take_log(J_2x2)
    logs_2x2 = np.log(s_2x2)
    sigma_oris = np.log(sigma_oris)
    
    for i in range(number_trials):
        
        testing_data = create_data(stimuli_pars, number = number_trials, offset = offset, ref_ori = ref_ori)
        
        _, _, pred_label, _, _ =vmap_model(ssn_ori_map, logJ_2x2, logs_2x2, c_E, c_I, w_sig, b_sig, sigma_oris, ssn_pars, grid_pars, conn_pars, gE, gI, testing_data, filter_pars, conv_pars, loss_pars, sig_noise, noise_type)
                         
        true_accuracy = np.sum(testing_data['label'] == pred_label)/len(testing_data['label']) 
        all_accs.append(true_accuracy)
   
    plt.hist(all_accs)
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
   
    
    if save:
            plt.savefig(save+'.png')
    
    plt.show()  
    plt.close()