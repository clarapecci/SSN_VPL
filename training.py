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


    
    
def save_params_dict(opt_pars, true_acc, ber_acc, epoch ):
    save_params = {}
    save_params= dict(epoch = epoch, val_accuracy= true_acc, 
                      ber_accuracy = ber_acc)
    
    for key in opt_pars.keys():
        
        if key =='logJ_2x2':
            J_2x2 = sep_exponentiate(opt_pars['logJ_2x2'])
            Js = dict(J_EE= J_2x2[0,0], J_EI = J_2x2[0,1], 
                              J_IE = J_2x2[1,0], J_II = J_2x2[1,1])
            save_params.update(Js)
        
        elif key =='logs_2x2':
            s_2x2 = np.exp(opt_pars['logs_2x2'])
            ss = dict(s_EE= s_2x2[0,0], s_EI = s_2x2[0,1], 
                              s_IE = s_2x2[1,0], s_II = s_2x2[1,1])
        
            save_params.update(ss)
        
        elif key=='sigma_oris':
            if len(opt_pars['sigma_oris']) ==1:
                save_params[key] = opt_pars[key]
            else:
                sigma_oris = dict(sigma_orisE = np.exp(opt_pars['sigma_oris'][0]), sigma_orisI = np.exp(opt_pars['sigma_oris'][1]))
                save_params.update(sigma_oris)
        
        elif key =='w_sig':
            save_params[key] = opt_pars[key]
            norm_w = np.linalg.norm(opt_pars[key])
            save_params['norm_w'] = norm_w
        
        else:
                save_params[key] = opt_pars[key]

    
    return save_params


def exponentiate(opt_pars):
    signs=np.array([[1, -1], [1, -1]]) 
    J_2x2 =np.exp(opt_pars['logJ_2x2'])*signs
    s_2x2 = np.exp(opt_pars['logs_2x2'])

    
    return J_2x2, s_2x2


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


def constant_to_vec(c_E, c_I):
    
    matrix_E = np.zeros((9,9))
    matrix_E = matrix_E.at[2:7, 2:7].set(c_E)
    vec_E = np.ravel(matrix_E)
    
    matrix_I = np.zeros((9,9))
    matrix_I = matrix_I.at[2:7, 2:7].set(c_I)
    vec_I = np.ravel(matrix_I)
    
    constant_vec = np.hstack((vec_E, vec_I, vec_E, vec_I))
    return constant_vec

def our_max(x, beta=0.5):
    #nscipy function
    #max_val = scipy.special.logsumexp(x*beta)/beta
    max_val = np.log(np.sum(np.exp(x*beta)))/beta
    return max_val


def sigmoid(x, epsilon = 0.001):
    '''
    Introduction of epsilon stops asymptote from reaching 1 (avoids NaN)
    '''
    sig = 1/(1+np.exp(x))
    
    return (1 - 2*epsilon)*sig + epsilon


def binary_loss(n, x):
    return - (n*np.log(x) + (1-n)*np.log(1-x))

def obtain_fixed_point(ssn, ssn_input, conv_pars,  Rmax_E = 50, Rmax_I = 100, inhibition = False):
    
    r_init = np.zeros(ssn_input.shape[0])
    dt = conv_pars.dt
    xtol = conv_pars.xtol
    Tmax = conv_pars.Tmax
    verbose = conv_pars.verbose
    silent = conv_pars.silent
    
    #Find fixed point
    fp, _, avg_dx = ssn.fixed_point_r(ssn_input, r_init=r_init, dt=dt, xtol=xtol, Tmax=Tmax, verbose = verbose, silent=silent)

    avg_dx = np.maximum(0, (avg_dx -1))
    
    #Apply bounding box to data
    r_box = (ssn.apply_bounding_box(fp, size=3.2)).ravel()
    
    #Obtain inhibitory response 
    if inhibition ==True:
        r_box_i = ssn.apply_bounding_box(fp, size=3.2, select='I_ON')
        r_box = [r_box, r_box_i.ravel()]
        
    r_max = np.maximum(0, (our_max(fp[:ssn.Ne])/Rmax_E - 1)) + np.maximum(0, (our_max(fp[ssn.Ne:-1])/Rmax_I - 1))
    return r_box, r_max, avg_dx


def take_log(J_2x2):
    
    signs=np.array([[1, -1], [1, -1]])
    logJ_2x2 =np.log(J_2x2*signs)
    
    return logJ_2x2


def sep_exponentiate(J_s):
    signs=np.array([[1, -1], [1, -1]]) 
    new_J =np.exp(J_s)*signs

    return new_J

def create_param_1(logJ_2x2, logs_2x2, sigma_oris, c_E, c_I, w_sig, b_sig, conn_pars):
    '''
    Training:logJ_2x2, log_s2x2, w_sig, b_sig, c_E, c_I
    '''
    
    opt_pars = dict(logJ_2x2 = logJ_2x2, logs_2x2 = logs_2x2, w_sig = w_sig, b_sig=b_sig, c_E = c_E, c_I = c_I)
    conn_pars.sigma_oris = sigma_oris

    return opt_pars, conn_pars

def create_param_2(logJ_2x2, logs_2x2, sigma_oris, c_E, c_I, w_sig, b_sig, conn_pars):
    '''
    Training:logJ_2x2, sigma_oris, w_sig, b_sig, c_E, c_I
    '''
    
    opt_pars = dict(logJ_2x2 = logJ_2x2, sigma_oris = sigma_oris, w_sig = w_sig, b_sig=b_sig, c_E = c_E, c_I = c_I)
    conn_pars.s_2x2 = logs_2x2

    return opt_pars, conn_pars

def create_param_3(logJ_2x2, logs_2x2, sigma_oris, c_E, c_I, w_sig, b_sig, conn_pars):
    '''
    Training:logJ_2x2, w_sig, b_sig, c_E, c_I
    '''
    
  
    opt_pars = dict(logJ_2x2 = logJ_2x2, w_sig = w_sig, b_sig=b_sig, c_E = c_E, c_I = c_I)
    conn_pars.sigma_oris = sigma_oris
    conn_pars.s_2x2 = logs_2x2
    
    return opt_pars, conn_pars
    

def create_param_4(logJ_2x2, logs_2x2, sigma_oris, c_E, c_I, w_sig, b_sig, conn_pars):
    '''
    Training: w_sig, b_sig
    '''
    conn_pars.J_2x2 = logJ_2x2
    conn_pars.s_2x2 = logs_2x2
    conn_pars.sigma_oris = sigma_oris
    conn_pars.c_E = c_E
    conn_pars.c_I = c_I
    
    opt_pars = dict(w_sig = w_sig, b_sig = b_sig)

    return opt_pars, conn_pars


def create_param_5(logJ_2x2, logs_2x2, sigma_oris, c_E, c_I, w_sig, b_sig, conn_pars):
    '''
    Training: logJ_2x2, log_s2x2, c_E, c_I
    '''
    
    opt_pars = dict(logJ_2x2 = logJ_2x2, logs_2x2 = logs_2x2, c_E = c_E, c_I = c_I)
    conn_pars.sigma_oris = sigma_oris
    conn_pars.w_sig = w_sig
    conn_pars.b_sig = b_sig
    
    return opt_pars, conn_pars


    
def separate_param_1(opt_pars, conn_pars):
    logJ_2x2 = opt_pars['logJ_2x2']
    logs_2x2 = opt_pars['logs_2x2']
    c_E =opt_pars['c_E']
    c_I =opt_pars['c_I']
    w_sig = opt_pars['w_sig']
    b_sig = opt_pars['b_sig']
    sigma_oris = conn_pars.sigma_oris
    
    return logJ_2x2, logs_2x2, c_E, c_I, w_sig, b_sig, sigma_oris

def separate_param_2(opt_pars, conn_pars):
    
    logJ_2x2 = opt_pars['logJ_2x2']
    c_E =opt_pars['c_E']
    c_I =opt_pars['c_I']
    w_sig = opt_pars['w_sig']
    b_sig = opt_pars['b_sig']
    sigma_oris=opt_pars['sigma_oris']
    log_s2x2 = conn_pars.s_2x2
    
    return logJ_2x2, logs_2x2, c_E, c_I, w_sig, b_sig, sigma_oris


def separate_param_3(opt_pars, conn_pars):
    logJ_2x2 = opt_pars['logJ_2x2']
    logs_2x2 = conn_pars.s_2x2
    c_E =opt_pars['c_E']
    c_I =opt_pars['c_I']
    w_sig = opt_pars['w_sig']
    b_sig = opt_pars['b_sig']
    sigma_oris = conn_pars.sigma_oris
    
    return logJ_2x2, logs_2x2, c_E, c_I, w_sig, b_sig, sigma_oris


def separate_param_4(opt_pars, conn_pars):
    log_J_2x2 = conn_pars.J_2x2
    log_s_2x2 = conn_pars.s_2x2
    sigma_oris = conn_pars.sigma_oris
    c_E = conn_pars.c_E
    c_I = conn_pars.c_I
    w_sig = opt_pars['w_sig']
    b_sig = opt_pars['b_sig']
    
    return logJ_2x2, logs_2x2, c_E, c_I, w_sig, b_sig, sigma_oris

def separate_param_5(opt_pars, conn_pars):
    logJ_2x2 = opt_pars['logJ_2x2']
    logs_2x2 = opt_pars['logs_2x2']
    c_E =opt_pars['c_E']
    c_I =opt_pars['c_I']
    
    w_sig = conn_pars.w_sig
    b_sig = conn_pars.b_sig
    sigma_oris = conn_pars.sigma_oris
    
    return logJ_2x2, logs_2x2, c_E, c_I, w_sig, b_sig, sigma_oris

@partial(jax.jit, static_argnums=( 7, 8, 9, 10, 11, 13, 14, 15, 17), device = jax.devices()[0])
def model(logJ_2x2, logs_2x2, c_E, c_I, w_sig, b_sig, sigma_oris, ssn_pars, grid_pars, conn_pars, gE, gI, train_data, filter_pars,  conv_pars, loss_pars, sig_noise, noise_type='no_noise'):

    
    J_2x2 = sep_exponentiate(logJ_2x2)
    s_2x2 = np.exp(logs_2x2)
    sigma_oris = np.exp(sigma_oris)

    #Create vector using extrasynaptic constants
    constant_vector = constant_to_vec(c_E, c_I)
    
    #Initialise network
    ssn=SSN2DTopoV1_AMPAGABA_ONOFF(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars, filter_pars=filter_pars, J_2x2=J_2x2, s_2x2=s_2x2, gE=gE, gI=gI, sigma_oris=sigma_oris)
   
    #Apply Gabor filters to stimuli
    output_ref=np.matmul(ssn.gabor_filters, train_data['ref']) + constant_vector
    output_target=np.matmul(ssn.gabor_filters, train_data['target']) + constant_vector
    
    #Rectify output
    SSN_input_ref=np.maximum(0, output_ref)
    SSN_input_target=np.maximum(0, output_target)

    #Find the fixed point 
    r_ref, r_max_ref, avg_dx_ref = obtain_fixed_point(ssn, SSN_input_ref, conv_pars)
    r_target, r_max_target, avg_dx_target = obtain_fixed_point(ssn, SSN_input_target, conv_pars)
   
    
    #Add additional noise before sigmoid layer
    if noise_type =='additive':
        r_ref =r_ref + sig_noise*numpy.random.normal(size=(r_ref.shape))
        r_target = r_target + sig_noise*numpy.random.normal(size=(r_target.shape))
        
    elif noise_type == 'multiplicative':
        r_ref = r_ref*(1 + sig_noise*numpy.random.normal(size=(r_ref.shape)))
        r_target = r_target*(1 + sig_noise*numpy.random.normal(size=(r_target.shape)))
         
    elif noise_type =='poisson':
        r_ref = r_ref + sig_noise*np.sqrt(r_ref)*numpy.random.normal(size=(r_ref.shape))
        r_target = r_target + sig_noise*np.sqrt(r_target)*numpy.random.normal(size=(r_target.shape))

    elif noise_type =='no_noise':
        pass
    
    else:
        raise Exception('Noise type must be one of: additive, mulitiplicative, poisson')
    
    delta_x = r_ref - r_target
    
    #Apply sigmoid function - combine ref and target
    sig_input = np.dot(w_sig, (delta_x)) + b_sig
    
    x = sigmoid( np.dot(w_sig, (delta_x)) + b_sig)

    #Calculate losses
    loss_binary=binary_loss(train_data['label'], x)
    loss_avg_dx = loss_pars.lambda_1*(avg_dx_ref + avg_dx_target)/2
    loss_r_max =  loss_pars.lambda_2*(r_max_ref + r_max_target)/2
    loss_w = loss_pars.lambda_w*(np.linalg.norm(w_sig)**2)
    loss_b = loss_pars.lambda_b*(b_sig**2)
    
    #Combine all losses
    loss = loss_binary +  loss_avg_dx + loss_r_max  + loss_w + loss_b
    all_losses = np.vstack((loss_binary, loss_avg_dx, loss_r_max, loss_w, loss_b, loss))
    
    pred_label = np.round(x) 
    
    #Calculate predicted label using Bernoulli distribution
    key_int = numpy.random.randint(low = 0, high =  10000)
    key = random.PRNGKey(key_int)
    pred_label_b = np.sum(jax.random.bernoulli(key, p=x, shape=None))

   
    return loss, all_losses, pred_label, pred_label_b, sig_input, x



def loss(opt_pars, ssn_pars, grid_pars, conn_pars, gE, gI, test_data, filter_pars,  conv_pars, loss_pars, sig_noise, noise_type, vmap_model, model_type=1, evaluate=False):
    
    #Separate parameters
    if model_type==1:
        logJ_2x2, logs_2x2, c_E, c_I, w_sig, b_sig, sigma_oris = separate_param_1(opt_pars, conn_pars)
        
    if model_type==2:
        logJ_2x2, logs_2x2, c_E, c_I, w_sig, b_sig, sigma_oris = separate_param_2(opt_pars, conn_pars)
    
    if model_type==3:
        logJ_2x2, logs_2x2, c_E, c_I, w_sig, b_sig, sigma_oris = separate_param_3(opt_pars, conn_pars)
    
    if model_type ==4:
        logJ_2x2, logs_2x2, c_E, c_I, w_sig, b_sig, sigma_oris = separate_param_4(opt_pars, conn_pars)
        
    if model_type ==5:
        logJ_2x2, logs_2x2, c_E, c_I, w_sig, b_sig, sigma_oris = separate_param_5(opt_pars, conn_pars)
    
    total_loss, all_losses, pred_label, pred_label_b, delta_x, x= vmap_model(logJ_2x2, logs_2x2, c_E, c_I, w_sig, b_sig, sigma_oris, ssn_pars, grid_pars, conn_pars, gE, gI, test_data, filter_pars,  conv_pars, loss_pars, sig_noise, noise_type)
    
    loss= np.mean(total_loss)
    all_losses = np.mean(all_losses, axis = 0)
        
    true_accuracy = np.sum(test_data['label'] == pred_label)/len(test_data['label']) 
    ber_accuracy = np.sum(test_data['label'] == pred_label_b)/len(test_data['label']) 
       
        
    return loss, [all_losses, true_accuracy, ber_accuracy, delta_x, x]
    

    
    
def train_SSN_vmap(logJ_2x2, logs_2x2, sigma_oris, c_E, c_I, w_sig, b_sig, ssn_pars, grid_pars, conn_pars, gE, gI, stimuli_pars, filter_pars, conv_pars, loss_pars, epochs_to_save, results_filename = None, batch_size=20, ref_ori = 55, offset = 5, epochs=1, eta=10e-4, sig_noise = None, test_size = None, noise_type='additive', model_type=1, readout_pars=None, early_stop = 0.6):
          
    #Initialize loss
    val_loss_per_epoch = []
    training_losses=[]
    train_accs = []
    train_sig_input = []
    train_sig_output = []
    val_sig_input = []
    val_sig_output = []
    
    test_size = batch_size if test_size is None else test_size
    
    #Initialise vmap version of model
    vmap_model = vmap(new_model, in_axes = (None, None, None, None, None, None, None, None, None, None, None, None, {'ref':0, 'target':0, 'label':0}, None, None, None, None, None) )
    
    #Separate parameters used in optimisation
    if model_type ==1:
        opt_pars, conn_pars = create_param_1(logJ_2x2, logs_2x2, sigma_oris, c_E, c_I, w_sig, b_sig, conn_pars)
    
    if model_type==2:
        opt_pars, conn_pars = create_param_2(logJ_2x2, logs_2x2, sigma_oris, c_E, c_I, w_sig, b_sig, conn_pars)
    
    if model_type==3:
        opt_pars, conn_pars = create_param_3(logJ_2x2, logs_2x2, sigma_oris, c_E, c_I, w_sig, b_sig, conn_pars)
    
    if model_type ==4:
        opt_pars, conn_pars= create_param_4(logJ_2x2, logs_2x2, sigma_oris, c_E, c_I, w_sig, b_sig, conn_pars)
        
    if model_type ==5:
        opt_pars, conn_pars = create_param_5(logJ_2x2, logs_2x2, sigma_oris, c_E, c_I, w_sig, b_sig, conn_pars)
    
    #Initialise optimizer
    optimizer = optax.adam(eta)
    opt_state = optimizer.init(opt_pars)
    
    print(opt_pars)
    
    print('Training model with learning rate {}, sig_noise {} at offset {}, lam_w {}, batch size {}, noise_type {}'.format(eta, sig_noise, offset, loss_pars.lambda_w, batch_size, noise_type))
    
    #Define test data - no need to iterate
    test_data = create_data(stimuli_pars, number = test_size, offset = offset, ref_ori = ref_ori)
    val_loss, [all_losses, true_acc, ber_acc, delta_x, x]= loss(opt_pars, ssn_pars, grid_pars, conn_pars, gE, gI, test_data, filter_pars,  conv_pars, loss_pars, sig_noise, noise_type, vmap_model, model_type, evaluate = True)
    print('Before training  -- loss: {}, true accuracy: {} , Bernoulli accuracy: {}'.format(np.round(float(val_loss), 3), np.round(true_acc, 3), np.round(ber_acc, 3)))
    val_loss_per_epoch.append(val_loss)
    train_sig_input.append(delta_x)
    val_sig_input.append(delta_x)
    train_sig_output.append(x)
    val_sig_output.append(x)
    
    #Save initial parameters
    initial_save_params = save_params_dict(opt_pars=opt_pars, true_acc=true_acc, ber_acc = ber_acc, epoch=0)
    
    #Initialise csv file
    if results_filename:
        results_handle = open(results_filename, 'w')
        results_writer = csv.DictWriter(results_handle, fieldnames=initial_save_params.keys(), delimiter=',')
        results_writer.writeheader()
        results_writer.writerow(initial_save_params)
        print('Saving results to csv ', results_filename)
    else:
        print('#### NOT SAVING! ####')
    
    loss_and_grad = jax.value_and_grad(loss, has_aux = True)
    
    for epoch in range(1, epochs+1):
        start_time = time.time()
        epoch_loss = 0 
           
        #Load next batch of data and convert
        train_data = create_data(stimuli_pars, number = batch_size, offset = offset, ref_ori = ref_ori)

        #Compute loss and gradient
        [epoch_loss, [epoch_all_losses, train_true_acc, train_ber_acc, train_delta_x, train_x]], grad =loss_and_grad(opt_pars, ssn_pars, grid_pars, conn_pars, gE, gI, train_data, filter_pars, conv_pars, loss_pars, sig_noise, noise_type, vmap_model, model_type)
        training_losses.append(epoch_loss)
        all_losses = np.hstack((all_losses, epoch_all_losses))
        train_accs.append(train_true_acc)
        train_sig_input.append(train_delta_x)
        train_sig_output.append(train_x)
        
        
        epoch_time = time.time() - start_time
        
        if model_type ==4 and epoch>7 and np.mean(np.asarray(train_accs[-7:]))>early_stop:
                print('Early stop: {} accuracy achieved at epoch {}'.format(early_stop, epoch))
                break

        #Save the parameters given a number of epochs
        if epoch in epochs_to_save:
            
            #Evaluate model 
            test_data = create_data(stimuli_pars, number = test_size, offset = offset, ref_ori = ref_ori)
            start_time = time.time()
            val_loss, [val_all_losses, true_acc, ber_acc, val_delta_x, val_x ]= loss(opt_pars, ssn_pars, grid_pars, conn_pars, gE, gI, test_data, filter_pars,  conv_pars, loss_pars, sig_noise, noise_type, vmap_model, model_type, evaluate = True)
            val_time = time.time() - start_time
            print('Training loss: {} ¦ Validation -- loss: {}, true accuracy: {}, Bernoulli accuracy: {} at epoch {}, (time {}, {})'.format(epoch_loss, val_loss, true_acc, ber_acc, epoch, epoch_time, val_time))
            val_loss_per_epoch.append(val_loss)
            val_sig_input.append(val_delta_x)
            val_sig_output.append(val_x)
            
                
        updates, opt_state = optimizer.update(grad, opt_state)
        opt_pars = optax.apply_updates(opt_pars, updates)
    
        #Save new optimized parameters
        if epoch in epochs_to_save:
            if results_filename:
                save_params = save_params_dict(opt_pars=opt_pars, true_acc=true_acc, ber_acc=ber_acc, epoch=epoch)
                results_writer.writerow(save_params)
    
    #Reparametize parameters
    signs=np.array([[1, -1], [1, -1]])
    if 'logJ_2x2' in opt_pars.keys():
        opt_pars['logJ_2x2'] = np.exp(opt_pars['logJ_2x2'])*signs
    if 'logs_2x2' in opt_pars.keys():
        opt_pars['logs_2x2'] = np.exp(opt_pars['logs_2x2'])
    if 'sigma_oris' in opt_pars.keys():
        opt_pars['sigma_oris'] = np.exp(opt_pars['sigma_oris'])
   
    return opt_pars, val_loss_per_epoch, all_losses, train_accs, train_sig_input, train_sig_output, val_sig_input, val_sig_output


