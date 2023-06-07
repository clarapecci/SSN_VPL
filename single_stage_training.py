import os
import matplotlib.pyplot as plt
import time, os, json
import pandas as pd
import jax

from jax import random
from jax.config import config 
import jax.numpy as np
from jax import vmap
import optax
from functools import partial
import math
import csv
import gc
import time

from torch.utils.data import DataLoader
import numpy
from SSN_classes_jax_jit import SSN2DTopoV1_ONOFF_local
from SSN_classes_jax_on_only import SSN2DTopoV1
from util import GaborFilter, BW_Grating, find_A, create_gratings, param_ratios, create_data, take_log
from IPython.core.debugger import set_trace
import util 
from two_layer_training import create_data, exponentiate, model, test_accuracy, plot_w_sig, loss, _new_model, sep_exponentiate


def single_stage_training(J_2x2_m, J_2x2_s, s_2x2_s, sigma_oris_s, c_E, c_I, f_E, f_I, w_sig, b_sig, constant_ssn_pars, stimuli_pars, epochs_to_save, results_filename = None, batch_size=20, ref_ori = 55, offset = 5, epochs=1, eta=10e-4, sig_noise = None, test_size = None, noise_type='additive', results_dir = None,  ssn_ori_map=None):
    
    '''
    Training function for two layer model. Readout layer and SSN parameters are updated simultaneously. 
    '''
    
    #Initialize loss
    val_loss_per_epoch = []
    training_losses=[]
    train_accs = []
    train_sig_input = []
    train_sig_output = []
    val_sig_input = []
    val_sig_output = []
    val_accs=[]
    save_w_sigs = []
    r_refs = []
    save_w_sigs.append(w_sig[:5])
    epoch = 0
    
  
    #Take logs of parameters
    logJ_2x2_s = take_log(J_2x2_s)
    logs_2x2 = np.log(s_2x2_s)
    logJ_2x2_m = take_log(J_2x2_m)
    logJ_2x2 = [logJ_2x2_m, logJ_2x2_s]
    sigma_oris = np.log(sigma_oris_s)
    constant_ssn_pars['key'] = random.PRNGKey(42) #ADD SEED
    
    print('Loss pars ', constant_ssn_pars['loss_pars'])
    if ssn_ori_map == None:
        #Initialise networks
        print('Creating new orientation map')
        ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=constant_ssn_pars['ssn_pars'], grid_pars=constant_ssn_pars['grid_pars'], conn_pars=constant_ssn_pars['conn_pars_m'], filter_pars=constant_ssn_pars['filter_pars'],  J_2x2=J_2x2_m, gE = constant_ssn_pars['gE'][0], gI=constant_ssn_pars['gI'][0])
        constant_ssn_pars['ssn_mid_ori_map'] = ssn_mid.ori_map
        constant_ssn_pars['ssn_sup_ori_map'] = ssn_mid.ori_map
    else:
        print('Loading orientation map')
        constant_ssn_pars['ssn_mid_ori_map'] = ssn_ori_map
        constant_ssn_pars['ssn_sup_ori_map'] = ssn_ori_map
        
        
    
    
    #Reassemble parameters into corresponding dictionaries
    constant_ssn_pars['logs_2x2'] = logs_2x2
    constant_ssn_pars['sigma_oris'] = sigma_oris
    
    readout_pars = dict(w_sig = w_sig, b_sig = b_sig)
    ssn_layer_pars = dict(logJ_2x2 = logJ_2x2, c_E = c_E, c_I = c_I, f_E = f_E, f_I = f_I)
    loss_pars = constant_ssn_pars['loss_pars']
    
    print(constant_ssn_pars['ssn_mid_ori_map'])
    
    test_size = batch_size if test_size is None else test_size
        
    #Initialise optimizer
    opt_pars = [ssn_layer_pars, readout_pars]
    optimizer = optax.adam(eta)
    opt_state = optimizer.init(opt_pars)
    
    print('Training model for {} epochs  with learning rate {}, sig_noise {} ({}) at offset {}, lam_w {}, batch size {}, noise_type {}'.format(epochs, eta, sig_noise, noise_type, offset, loss_pars.lambda_w, batch_size, noise_type))
    
    #Saving warning
    if results_filename:
        print('Saving results to csv ', results_filename)
    else:
        print('#### NOT SAVING! ####')
    
    #Define gradient function
    loss_and_grad_single_stage= jax.value_and_grad(loss_single_stage, argnums=0, has_aux = True)
    
    #Test accuracy before training
    #test_accuracy(ssn_layer_pars, readout_pars, constant_ssn_pars, stimuli_pars, offset, ref_ori, save=os.path.join(results_dir+ '_before_training'), number_trials = 20, batch_size = 500)
   
    print(opt_pars)
    for epoch in range(0, epochs+1):
      
        start_time = time.time()
        epoch_loss = 0 
           
        #Generate new batch of data
        train_data = create_data(stimuli_pars, number = batch_size, offset = offset, ref_ori = ref_ori)
        
        debug_flag = False
        #Compute loss and gradient
        constant_ssn_pars['key'], _ = random.split(constant_ssn_pars['key'])
        [epoch_loss, [epoch_all_losses, train_true_acc, train_delta_x, train_x, train_r_ref]], grad =loss_and_grad_single_stage(opt_pars, constant_ssn_pars, train_data , debug_flag)

        training_losses.append(epoch_loss)
        if epoch==0:
            all_losses = epoch_all_losses
        else:
            all_losses = np.hstack((all_losses, epoch_all_losses)) 
        train_accs.append(train_true_acc)
        train_sig_input.append(train_delta_x)
        train_sig_output.append(train_x)
        r_refs.append(train_r_ref)
 
        epoch_time = time.time() - start_time
        

        #Save the parameters given a number of epochs
        if epoch in epochs_to_save:

            #Evaluate model 
            test_data = create_data(stimuli_pars, number = test_size, offset = offset, ref_ori = ref_ori)
            
            
            start_time = time.time()
            constant_ssn_pars['key'], _ = random.split(constant_ssn_pars['key'])
            [val_loss, [val_all_losses, true_acc, val_delta_x, val_x, _ ]], _= loss_and_grad_single_stage(opt_pars, constant_ssn_pars, test_data)
            val_time = time.time() - start_time
            
            print('Training loss: {} ¦ Validation -- loss: {}, true accuracy: {}, at epoch {}, (time {}, {}), w_sig {}'.format(epoch_loss, val_loss, true_acc, epoch, epoch_time, val_time, opt_pars[1]['w_sig'][:3]))
            if epoch%50 ==0:
                    print('Training accuracy: {}, all losses{}'.format(np.mean(np.asarray(train_accs[-20:])), epoch_all_losses))
            val_loss_per_epoch.append([val_loss, int(epoch)])
            val_sig_input.append([val_delta_x, epoch])
            val_sig_output.append(val_x)
            val_accs.append(true_acc)
            
            if results_filename:
                save_params = save_params_single(opt_pars, true_acc, epoch)
                
                #Initialise results file
                if epoch==0:
                        results_handle = open(results_filename, 'w')
                        results_writer = csv.DictWriter(results_handle, fieldnames=save_params.keys(), delimiter=',')
                        results_writer.writeheader()
                        
                results_writer.writerow(save_params)


            updates, opt_state = optimizer.update(grad, opt_state)
            opt_pars = optax.apply_updates(opt_pars, updates)

            save_w_sigs.append(opt_pars[1]['w_sig'][:5])
            plot_w_sig(save_w_sigs, epochs_to_save[:len(save_w_sigs)], save = os.path.join(results_dir+'_w_sig_evolution') )
    
    plot_r_ref(r_refs, epoch_c = epoch_c, save = os.path.join(results_dir+'_noise') )
         
    return opt_pars, np.vstack([val_loss_per_epoch]), all_losses, train_accs, train_sig_input, train_sig_output, val_sig_input, val_sig_output, save_w_sigs






def loss_single_stage(opt_pars,  constant_ssn_pars, data, debug_flag=False):
    
    '''
    Wrapper function for loss: unwraps optimization parameters when doing single stage training
    '''
    ssn_layer_pars = opt_pars[0]
    readout_pars = opt_pars[1]
    
    total_loss, all_losses, pred_label, sig_input, x, r_ref= single_model(ssn_layer_pars = ssn_layer_pars, readout_pars = readout_pars, constant_ssn_pars = constant_ssn_pars, data = data, debug_flag = debug_flag)
    
    loss= np.mean(total_loss)
    all_losses = np.mean(all_losses, axis = 0)
    #r_ref = np.mean(r_ref, axis = 0)
        
    true_accuracy = np.sum(data['label'] == pred_label)/len(data['label']) 
    
    return loss, [all_losses, true_accuracy, sig_input, x, r_ref]
        




def save_params_single(opt_pars, true_acc, epoch ):
    
    '''
    Assemble trained parameters and epoch information into single dictionary for saving
    Inputs:
        dictionaries containing trained parameters
        other epoch parameters (accuracy, epoch number)
    Outputs:
        single dictionary concatenating all information to be saved
    '''
    
    ssn_layer_pars =opt_pars[0]
    readout_pars = opt_pars[1]
    
    save_params = {}
    save_params= dict(epoch = epoch, val_accuracy= true_acc)
    
    
    J_2x2_m = sep_exponentiate(ssn_layer_pars['logJ_2x2'][0])
    Jm = dict(J_EE_m= J_2x2_m[0,0], J_EI_m = J_2x2_m[0,1], 
                              J_IE_m = J_2x2_m[1,0], J_II_m = J_2x2_m[1,1])
            
    J_2x2_s = sep_exponentiate(ssn_layer_pars['logJ_2x2'][1])
    Js = dict(J_EE_s= J_2x2_s[0,0], J_EI_s = J_2x2_s[0,1], 
                              J_IE_s = J_2x2_s[1,0], J_II_s = J_2x2_s[1,1])
            
    save_params.update(Jm)
    save_params.update(Js)
    save_params['c_E'] = ssn_layer_pars['c_E']
    save_params['c_I'] = ssn_layer_pars['c_I']
    save_params['f_E'] = ssn_layer_pars['f_E']
    save_params['f_I'] = ssn_layer_pars['f_I']
    
    
    if 'sigma_oris' in ssn_layer_pars.keys():
        if len(ssn_layer_pars['sigma_oris']) ==1:
            save_params[key] = np.exp(ssn_layer_pars[key])
        else:
            sigma_oris = dict(sigma_orisE = np.exp(ssn_layer_pars['sigma_oris'][0]), sigma_orisI = np.exp(ssn_layer_pars['sigma_oris'][1]))
            save_params.update(sigma_oris)
        
    #Add readout parameters
    save_params.update(readout_pars)

    
    return save_params

















jitted_model_single = jax.jit(_new_model, static_argnums = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
#Vmap implementation of model function
vmap_model_jit_single = vmap(jitted_model_single, in_axes = ([None, None], None, None, None, None, None, None
                            , None, None, None, None, {'ref':0, 'target':0, 'label':0}, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None) )




def single_model(ssn_layer_pars, readout_pars, constant_ssn_pars, data, debug_flag=False):
    
    '''
    Wrapper function for model.
    Inputs: 
        parameters assembled  into dictionaries
    Output:
        output of model using unwrapped parameters
    '''
    
    #Obtain variables from dictionaries
    logJ_2x2 = ssn_layer_pars['logJ_2x2']
    c_E = ssn_layer_pars['c_E']
    c_I = ssn_layer_pars['c_I']
    f_E = ssn_layer_pars['f_E']
    f_I = ssn_layer_pars['f_I']
    
    sigma_oris = constant_ssn_pars['sigma_oris']
    
    w_sig = readout_pars['w_sig']
    b_sig = readout_pars['b_sig']
    
    _, subkey = random.split(constant_ssn_pars['key'])

    ssn_mid_ori_map = constant_ssn_pars['ssn_mid_ori_map']
    logs_2x2 = constant_ssn_pars['logs_2x2']
    ssn_pars = constant_ssn_pars['ssn_pars']
    grid_pars = constant_ssn_pars['grid_pars']
    conn_pars_m = constant_ssn_pars['conn_pars_m']
    conn_pars_s =constant_ssn_pars['conn_pars_s']
    gE_m =constant_ssn_pars['gE'][0]
    gE_s =constant_ssn_pars['gE'][1]
    gI_m = constant_ssn_pars['gI'][0]
    gI_s = constant_ssn_pars['gI'][1]
    filter_pars = constant_ssn_pars['filter_pars']
    conv_pars = constant_ssn_pars['conv_pars']
    loss_pars = constant_ssn_pars['loss_pars']
    sig_noise = constant_ssn_pars['sig_noise']
    noise_type = constant_ssn_pars['noise_type']
    
    
    return vmap_model_jit_single(logJ_2x2, logs_2x2, c_E, c_I, f_E, f_I, w_sig, b_sig, sigma_oris, ssn_mid_ori_map, ssn_mid_ori_map, data, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, gE_m, gI_m, gE_s, gI_s, filter_pars, conv_pars, loss_pars, sig_noise, noise_type, subkey, debug_flag)