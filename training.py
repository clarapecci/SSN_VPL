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

from torch.utils.data import DataLoader
import numpy
from SSN_classes_jax_jit import SSN2DTopoV1_AMPAGABA_ONOFF
from util import GaborFilter, BW_Grating, find_A, create_gratings, param_ratios, create_data


    
    
def create_data(stimuli_pars, number=100, offset = 5, ref_ori=55):
    
    '''
    Create data for given jitter and noise value for testing (not dataloader)
    '''
    data = create_gratings(ref_ori=ref_ori, number=number, offset=offset, **stimuli_pars)
    train_data = next(iter(DataLoader(data, batch_size=len(data), shuffle=False)))
    train_data['ref'] = train_data['ref'].numpy()
    train_data['target'] = train_data['target'].numpy()
    train_data['label'] = train_data['label'].numpy()
    
    return train_data


def save_params_dict(opt_pars, true_acc, ber_acc, epoch ):
    J_2x2, s_2x2 = exponentiate(opt_pars)
     
    save_params= dict(val_accuracy= true_acc, 
                      ber_accuracy = ber_acc,
                J_EE= J_2x2[0,0], J_EI = J_2x2[0,1], 
                              J_IE = J_2x2[1,0], J_II = J_2x2[1,1], 
                s_EE= s_2x2[0,0], s_EI = s_2x2[0,1], 
                              s_IE = s_2x2[1,0], s_II = s_2x2[1,1],
                c_E = opt_pars['c_E'], c_I = opt_pars['c_I'], sigma_oris = opt_pars['sigma_oris'],
                 epoch = epoch, w_sig = opt_pars['w_sig'], b_sig=opt_pars['b_sig'])
    
    return save_params

def constant_to_vec(c_E, c_I):
    
    matrix_E = np.zeros((9,9))
    matrix_E = matrix_E.at[2:7, 2:7].set(c_E)
    vec_E = np.ravel(matrix_E)
    
    matrix_I = np.zeros((9,9))
    matrix_I = matrix_I.at[2:7, 2:7].set(c_I)
    vec_I = np.ravel(matrix_I)
    
    constant_vec = np.hstack((vec_E, vec_E, vec_I, vec_I))
    return constant_vec

def sigmoid(x, epsilon = 0.001):
    
    '''
    Introduction of epsilon stops asymptote from reaching 1 (avoids NaN)
    '''
   
    sig = 1/(1+np.exp(x))
    
    return (1 - 2*epsilon)*sig + epsilon


def binary_loss(n, x):
    return - (n*np.log(x) + (1-n)*np.log(1-x))

def exponentiate(opt_pars):
    signs=np.array([[1, -1], [1, -1]]) 
    
    J_2x2 =np.exp(opt_pars['logJ_2x2'])*signs
    s_2x2 = np.exp(opt_pars['logs_2x2'])
    
    return J_2x2, s_2x2

def our_max(x, beta=0.5):
    max_val = np.log(np.sum(np.exp(x*beta)))/beta
    return max_val



#@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5 , 7 , 8, 9, 10, 11))#, device = jax.devices()[1]) #+ ADD 9 FOR TRAINING
def model(opt_pars, ssn_pars, grid_pars, conn_pars, gE, gI, train_data, filter_pars,  conv_pars, loss_pars, sig_noise, noise_type='no_noise'):
    
    J_2x2, s_2x2 = exponentiate(opt_pars)
    
    #Initialise network
    ssn=SSN2DTopoV1_AMPAGABA_ONOFF(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars, filter_pars=filter_pars, J_2x2=J_2x2, s_2x2=s_2x2, gE=gE, gI=gI, sigma_oris= opt_pars['sigma_oris'])
    
    #Create vector using extrasynaptic constants
    constant_vector = constant_to_vec(opt_pars['c_E'], opt_pars['c_I'])
    
    #Apply Gabor filters to stimuli
    output_ref=np.matmul(ssn.gabor_filters, train_data['ref']) + constant_vector
    output_target=np.matmul(ssn.gabor_filters, train_data['target']) + constant_vector
    
    #Rectify output
    SSN_input_ref=np.maximum(0, output_ref)
    SSN_input_target=np.maximum(0, output_target)

    #Find the fixed point 
    x_ref, r_max_ref, avg_dx_ref = obtain_fixed_point(ssn, SSN_input_ref, conv_pars)
    x_target, r_max_target, avg_dx_target = obtain_fixed_point(ssn, SSN_input_target, conv_pars)
    
    #Add additional noise before sigmoid layer
    #if sig_noise:
    delta_x = x_ref.ravel() - x_target.ravel() 
    
    if noise_type =='additive':
        external_noise = sig_noise*numpy.random.normal(size=((x_target.ravel()).shape))
        delta_x = delta_x + external_noise
    
    elif noise_type == 'multiplicative':
        external_noise = 1 + sig_noise*numpy.random.normal(size=((x_target.ravel()).shape))
        delta_x = delta_x * external_noise
        
    elif noise_type =='poisson':
        external_noise = sig_noise*np.sqrt(delta_x)*numpy.random.normal(size=((x_target.ravel()).shape))
        delta_x = delta_x + external_noise
   
    elif noise_type =='no_noise':
        pass
    else:
        raise Exception('Noise type must be one of: additive, mulitiplicative, poisson')
    
    #Apply sigmoid function - combine ref and target
    x = sigmoid( np.dot(opt_pars['w_sig'], (delta_x)) + opt_pars['b_sig'])

    #Calculate losses
    loss_binary=binary_loss(train_data['label'], x)
    loss_avg_dx = loss_pars.lambda_1*(avg_dx_ref + avg_dx_target)/2
    loss_r_max =  loss_pars.lambda_2*(r_max_ref + r_max_target)/2
    loss_w = loss_pars.lambda_w*(np.linalg.norm(opt_pars['w_sig'])**2)
    loss_b = loss_pars.lambda_b*(opt_pars['b_sig']**2)
    
    #Combine all losses
    loss = loss_binary +  loss_avg_dx + loss_r_max  + loss_w + loss_b
    all_losses = np.vstack((loss_binary, loss_avg_dx, loss_r_max, loss_w, loss_b, loss))
    
    pred_label = np.round(x) 
    
    #Calculate predicted label using Bernoulli distribution
    key_int = numpy.random.randint(low = 0, high =  10000)
    key = random.PRNGKey(key_int)
    pred_label_b = np.sum(jax.random.bernoulli(key, p=x, shape=None))
    pred_label = [pred_label, pred_label_b]

    return loss, all_losses, pred_label


def obtain_fixed_point(ssn, ssn_input, conv_pars,  Rmax_E = 50, Rmax_I = 100, inhibition = False):
    
    r_init = np.zeros(ssn_input.shape[0])
    
    dt = conv_pars.dt
    xtol = conv_pars.xtol
    Tmax = conv_pars.Tmax
    verbose = conv_pars.verbose
    silent = conv_pars.silent
    
    if conv_pars.Rmax_E:
        Rmax_E = conv_pars.Rmax_E
        Rmax_I = conv_pars.Rmax_I
    
    #Find fixed point  
   
    fp, _, avg_dx = ssn.fixed_point_r(ssn_input, r_init=r_init, dt=dt, xtol=xtol, Tmax=Tmax, verbose = verbose, silent=silent)
    avg_dx = np.maximum(0, (avg_dx -1))
    
    #Apply bounding box to data
    x_box = ssn.apply_bounding_box(fp, size=3.2)
    
    #Obtain inhibitory response 
    if inhibition ==True:
        x_box_i = ssn.apply_bounding_box(fp, size=3.2, select='I_ON')
        x_box = [x_box, x_box_i]
        
    r_max = np.maximum(0, (our_max(fp[:ssn.Ne])/Rmax_E - 1)) + np.maximum(0, (our_max(fp[ssn.Ne:-1])/Rmax_I - 1))
    
    return x_box, r_max, avg_dx



def loss(opt_pars, ssn_pars, grid_pars, conn_pars, gE, gI, train_data, filter_pars,  conv_pars, loss_pars, sig_noise, noise_type):
    '''
    Calculate parallelized loss for batch of data through vmap.
    Output:
        mean loss of all the input images
    '''
    
    vmap_model = vmap(model, in_axes = ({'b_sig': None,  'c_E':None, 'c_I': None,  'logJ_2x2': None, 'logs_2x2': None, 'sigma_oris':None, 'w_sig': None}, None, None, None, None, None, {'ref':0, 'target':0, 'label':0}, None, None, None, None, None) )                   
    total_loss, all_losses , _= vmap_model(opt_pars, ssn_pars, grid_pars, conn_pars, gE, gI, train_data, filter_pars, conv_pars, loss_pars, sig_noise, noise_type)
    loss= np.sum(total_loss)
    all_losses = np.mean(all_losses, axis = 0)
    
    return loss, all_losses


def vmap_eval(opt_pars, ssn_pars, grid_pars, conn_pars, gE, gI, test_data, filter_pars,  conv_pars, loss_pars, sig_noise, noise_type='additive'):
    
    eval_vmap = vmap(model, in_axes = ({'b_sig': None,  'c_E':None, 'c_I': None,  'logJ_2x2': None, 'logs_2x2': None, 'sigma_oris': None, 'w_sig': None}, None, None, None, None, None, {'ref':0, 'target':0, 'label':0}, None, None, None, None, None) )
    losses, _, pred_labels = eval_vmap(opt_pars, ssn_pars, grid_pars, conn_pars, gE, gI, test_data, filter_pars, conv_pars, loss_pars, sig_noise, noise_type) 

    #Find accuracy based on predicted labels
    true_accuracy = np.sum(test_data['label'] == pred_labels[0])/len(test_data['label']) 
    ber_accuracy = np.sum(test_data['label'] == pred_labels[1])/len(test_data['label']) 
    
    vmap_loss= np.mean(losses)
    
    
    return vmap_loss, true_accuracy, ber_accuracy




def train_SSN_vmap(opt_pars, ssn_pars, grid_pars, conn_pars, gE, gI, stimuli_pars, filter_pars, conv_pars, loss_pars, epochs_to_save, results_filename = None, batch_size=20, ref_ori = 55, offset = 5, epochs=1, eta=10e-4, sig_noise = None, test_size = 100, noise_type='additive'):
    
    #Initialize loss
    val_loss_per_epoch = []
    training_losses=[]
    
    #Initialise optimizer
    optimizer = optax.adam(eta)
    opt_state = optimizer.init(opt_pars)
    
    print('Training model with learning rate {}, sig_noise {} at offset {}, lam_w {}, batch size {}'.format(eta, sig_noise, offset, loss_pars.lambda_w, batch_size))
    
    #Define test data - no need to iterate
    test_data = create_data(stimuli_pars, number = test_size, offset = offset, ref_ori = ref_ori)
    val_loss, true_acc, ber_acc= vmap_eval(opt_pars, ssn_pars, grid_pars, conn_pars, gE, gI, test_data, filter_pars,  conv_pars, loss_pars, sig_noise, noise_type)
    print('Before training  -- loss: {}, true accuracy: {} , Bernoulli accuracy: {} (learning rate: {})'.format(np.round(float(val_loss), 3), np.round(true_acc, 3), np.round(ber_acc, 3), eta))
    val_loss_per_epoch.append(val_loss)
    
    #Save initial parameters
    save_params = save_params_dict(opt_pars=opt_pars, true_acc=true_acc, ber_acc = ber_acc, epoch=0 )
    
    #Initialise csv file
    if results_filename:
        results_handle = open(results_filename, 'w')
        results_writer = csv.DictWriter(results_handle, fieldnames=save_params.keys())
        results_writer.writeheader()
        results_writer.writerow(save_params)
        print('Saving results to csv ', results_filename)
    else:
        print('#### NOT SAVING! ####')
    
    for epoch in range(1, epochs+1):
        start_time = time.time()
        epoch_loss = 0 
           
        #Load next batch of data and convert
        train_data = create_data(stimuli_pars, number = batch_size, offset = offset, ref_ori = ref_ori)

        #Compute loss and gradient
        epoch_loss, grad =jax.value_and_grad(loss, has_aux = True)(opt_pars, ssn_pars, grid_pars, conn_pars, gE, gI, train_data, filter_pars, conv_pars, loss_pars, sig_noise, noise_type)

        #Apply SGD through Adam optimizer per batch
        updates, opt_state = optimizer.update(grad, opt_state)
        opt_pars = optax.apply_updates(opt_pars, updates)
        training_losses.append(epoch_loss[0])
    
        #Save all losses
        if epoch==1:
            all_losses = epoch_loss[1]
        else:
            all_losses = np.hstack((all_losses, epoch_loss[1]))
        
        epoch_time = time.time() - start_time

        #Save the parameters given a number of epochs
        if epoch in epochs_to_save:
            
            #Evaluate model 
            test_data = create_data(stimuli_pars, number = test_size, offset = offset, ref_ori = ref_ori)
            start_time = time.time()
            val_loss, true_acc, ber_acc= vmap_eval(opt_pars, ssn_pars, grid_pars, conn_pars, gE, gI, test_data, filter_pars,  conv_pars, loss_pars, sig_noise, noise_type)
            val_time = time.time() - start_time
            print('Training loss: {} ¦ Validation -- loss: {}, true accuracy: {}, Bernoulli accuracy: {} at epoch {}, (time {}, {})'.format(epoch_loss[0], val_loss, true_acc, ber_acc, epoch, epoch_time, val_time))
            val_loss_per_epoch.append(val_loss)
            
            #Create dictionary of parameters to save
            save_params = save_params_dict(opt_pars, true_acc, ber_acc, epoch)
            
            #Write results in csv file
            if results_filename:
                results_writer.writerow(save_params)

    #Reparametize parameters
    signs=np.array([[1, -1], [1, -1]])    
    opt_pars['logJ_2x2'] = np.exp(opt_pars['logJ_2x2'])*signs
    opt_pars['logs_2x2'] = np.exp(opt_pars['logs_2x2'])
    
   
    return opt_pars, val_loss_per_epoch, all_losses

