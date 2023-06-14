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


def constant_to_vec(c_E, c_I, ssn, sup = False):
    
    edge_length = ssn.grid_pars.gridsize_Nx

    matrix_E = np.ones((edge_length, edge_length)) * c_E
    vec_E = np.ravel(matrix_E)
    
    matrix_I = np.ones((edge_length, edge_length))* c_I
    vec_I = np.ravel(matrix_I)
    
    constant_vec = np.hstack((vec_E, vec_I, vec_E, vec_I))
    
    if sup:
        constant_vec = np.hstack((vec_E, vec_I))
        
    return constant_vec

def our_max(x, beta=0.5):
    max_val = np.log(np.sum(np.exp(x*beta)))/beta
    return max_val


def sigmoid(x, epsilon = 0.01):
    '''
    Introduction of epsilon stops asymptote from reaching 1 (avoids NaN)
    '''
    sig = 1/(1+np.exp(x))
    
    return (1 - 2*epsilon)*sig + epsilon


def binary_loss(n, x):
    return - (n*np.log(x) + (1-n)*np.log(1-x))

def obtain_fixed_point(ssn, ssn_input, conv_pars, PLOT=False, save=None, inds=None, print_dt = False):
    
    r_init = np.zeros(ssn_input.shape[0])
    dt = conv_pars.dt
    xtol = conv_pars.xtol
    Tmax = conv_pars.Tmax
    verbose = conv_pars.verbose
    silent = conv_pars.silent
    
    #Find fixed point
    if PLOT==True:
        fp, avg_dx = ssn.fixed_point_r_plot(ssn_input, r_init=r_init, dt=dt, xtol=xtol, Tmax=Tmax, verbose = verbose, silent=silent, PLOT=PLOT, save=save, inds=inds, print_dt = print_dt)
    else:
        fp, _, avg_dx = ssn.fixed_point_r(ssn_input, r_init=r_init, dt=dt, xtol=xtol, Tmax=Tmax, verbose = verbose, silent=silent, PLOT=PLOT, save=save)

    avg_dx = np.maximum(0, (avg_dx -1))
    
    return fp, avg_dx


def middle_layer_fixed_point(ssn, ssn_input, conv_pars,  Rmax_E = 50, Rmax_I = 100, inhibition = False, PLOT=False, save=None, inds=None, return_fp = False, print_dt = False):
    
    fp, avg_dx = obtain_fixed_point(ssn=ssn, ssn_input = ssn_input, conv_pars = conv_pars, PLOT = PLOT, save = save, inds = inds, print_dt = print_dt)
    
    #Add responses from E and I neurons
    fp_E_on = ssn.select_type(fp, select='E_ON').ravel()
    fp_E_off = ssn.select_type(fp, select='E_OFF').ravel()
    
    layer_output = fp_E_on + fp_E_off
    
    r_max = np.maximum(0, (our_max(fp[:ssn.Ne])/Rmax_E - 1)) + np.maximum(0, (our_max(fp[ssn.Ne:-1])/Rmax_I - 1))
    
    if return_fp ==True:
        return layer_output, r_max, avg_dx, fp
    else:
        return layer_output, r_max, avg_dx
    

def obtain_fixed_point_centre_E(ssn, ssn_input, conv_pars,  Rmax_E = 50, Rmax_I = 100, inhibition = False, PLOT=False, save=None, inds=None):
    #Obtain fixed point
    fp, avg_dx = obtain_fixed_point(ssn=ssn, ssn_input = ssn_input, conv_pars = conv_pars, PLOT = PLOT, save = save, inds = inds)
    
    #Apply bounding box to data
    r_box = (ssn.apply_bounding_box(fp, size=3.2)).ravel()
    
    #Obtain inhibitory response 
    if inhibition ==True:
        r_box_i = ssn.apply_bounding_box(fp, size=3.2, select='I_ON').ravel()
        r_box = [r_box, r_box_i]
    
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



#@partial(jax.jit, static_argnums=( 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22), device = jax.devices()[0]) 
def _new_model(logJ_2x2, logs_2x2, c_E, c_I, f_E, f_I, w_sig, b_sig, sigma_oris, ssn_mid_ori_map, ssn_sup_ori_map, train_data, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, gE_m, gI_m, gE_s, gI_s, filter_pars, conv_pars, loss_pars, sig_noise, noise_ref, noise_target, debug_flag=False):
    
    '''
    SSN two-layer model. SSN layers are regenerated every run. Static arguments in jit specify parameters that stay constant throughout training. Static parameters cant be dictionaries
    Inputs:
        individual parameters - having taken logs of differentiable parameters
        noise_type: select different noise models
        debug_flag: to be used in pdb mode allowing debugging inside function
    Outputs:
        losses to take gradient with respect to
        sig_input, x: I/O values for sigmoid layer
    '''

    
    J_2x2_m = sep_exponentiate(logJ_2x2[0])
    J_2x2_s = sep_exponentiate(logJ_2x2[1])
    s_2x2_s = np.exp(logs_2x2)
    sigma_oris_s = np.exp(sigma_oris)
    f_E = np.exp(f_E)
    f_I = np.exp(f_I)
    

    #Initialise network
    ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_m, filter_pars=filter_pars, J_2x2=J_2x2_m, gE = gE_m, gI=gI_m, ori_map = ssn_mid_ori_map)
    ssn_sup=SSN2DTopoV1(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_s, filter_pars=filter_pars, J_2x2=J_2x2_s, s_2x2=s_2x2_s, gE = gE_s, gI=gI_s, sigma_oris = sigma_oris_s, ori_map = ssn_sup_ori_map)

    
    #Create vector using extrasynaptic constants
    constant_vector = constant_to_vec(c_E = c_E, c_I = c_I, ssn= ssn_mid)
    constant_vector_sup = constant_to_vec(c_E = c_E, c_I = c_I, ssn = ssn_sup, sup=True)
    
    #Apply Gabor filters to stimuli
    output_ref=np.matmul(ssn_mid.gabor_filters, train_data['ref'])
    output_target=np.matmul(ssn_mid.gabor_filters, train_data['target'])
    
    #Rectify output
    SSN_input_ref = np.maximum(0, output_ref) + constant_vector
    SSN_input_target = np.maximum(0, output_target) + constant_vector

    #Find fixed point for middle layer
    r_ref_mid, r_max_ref_mid, avg_dx_ref_mid = middle_layer_fixed_point(ssn_mid, SSN_input_ref, conv_pars)
    r_target_mid, r_max_target_mid, avg_dx_target_mid = middle_layer_fixed_point(ssn_mid, SSN_input_target, conv_pars)
    
 
    #Input to superficial layer
    sup_input_ref = np.hstack([r_ref_mid*f_E, r_ref_mid*f_I]) + constant_vector_sup
    sup_input_target = np.hstack([r_target_mid*f_E, r_target_mid*f_I]) + constant_vector_sup
    
    #Find fixed point for superficial layer
    r_ref, r_max_ref_sup, avg_dx_ref_sup= obtain_fixed_point_centre_E(ssn_sup, sup_input_ref, conv_pars)
    r_target, r_max_target_sup, avg_dx_target_sup= obtain_fixed_point_centre_E(ssn_sup, sup_input_target, conv_pars)
    
    noise_type = None
    #Add additional noise before sigmoid layer
    if noise_type =='additive':
        noise_add = sig_noise*numpy.random.normal(size=(r_ref.shape))
        r_ref =r_ref + noise_add
        r_target = r_target + sig_noise*numpy.random.normal(size=(r_target.shape))
        
    elif noise_type == 'multiplicative':
        r_ref = r_ref*(1 + sig_noise*numpy.random.normal(size=(r_ref.shape)))
        r_target = r_target*(1 + sig_noise*numpy.random.normal(size=(r_target.shape)))
        
    elif noise_type =='poisson':
        #noise_add = sig_noise*np.sqrt(jax.nn.softplus(r_ref))*numpy.random.normal(size=(r_ref.shape))
        
        noise_add = sig_noise*jax.random.normal(key, shape=(r_ref.shape))
        r_ref = r_ref + noise_add*np.sqrt(jax.nn.softplus(r_ref))

        r_target = r_target + sig_noise*np.sqrt(jax.nn.softplus(r_target))*numpy.random.normal(size=(r_target.shape))
       
    elif noise_type =='no_noise':
        pass
    
    #else:
    #    raise Exception('Noise type must be one of: additive, mulitiplicative, poisson')
  
    r_ref = r_ref + noise_ref
    r_target = r_target + noise_target
    delta_x = r_ref - r_target

    
    sig_input = np.dot(w_sig, (delta_x)) + b_sig
    
    #Apply sigmoid function - combine ref and target
    x = sigmoid(sig_input)
    
    #Calculate losses
    loss_binary=binary_loss(train_data['label'], x)
    loss_avg_dx = loss_pars.lambda_dx*(avg_dx_ref_mid + avg_dx_target_mid + avg_dx_ref_sup + avg_dx_target_sup )/4
    loss_r_max =  loss_pars.lambda_r_max*(r_max_ref_mid + r_max_target_mid + r_max_ref_sup + r_max_target_sup )/4
    loss_w = loss_pars.lambda_w*(np.linalg.norm(w_sig)**2)
    loss_b = loss_pars.lambda_b*(b_sig**2)
    
    #Combine all losses
    loss = loss_binary + loss_w + loss_b +  loss_avg_dx + loss_r_max
    all_losses = np.vstack((loss_binary, loss_avg_dx, loss_r_max, loss_w, loss_b, loss))
    pred_label = np.round(x) 

   
    return loss, all_losses, pred_label, sig_input, x,  noise_ref






jitted_model = jax.jit(_new_model, static_argnums = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], device = jax.devices()[0])
#Vmap implementation of model function
vmap_model_jit = vmap(jitted_model, in_axes = ([None, None], None, None, None, None, None, None
                            , None, None, None, None, {'ref':0, 'target':0, 'label':0}, None, None, None, None, None, None, None, None, None, None, None, None, 0, 0, None) )

vmap_model = vmap(_new_model, in_axes = ([None, None], None, None, None, None, None, None
                            , None, None, None, None, {'ref':0, 'target':0, 'label':0}, None, None, None, None, None, None, None, None, None, None, None, None, 0, 0, None) )





def model(ssn_layer_pars, readout_pars, constant_ssn_pars, data, debug_flag=False):
    
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
    
    #sigma_oris = ssn_layer_pars['sigma_oris']
    sigma_oris = constant_ssn_pars['sigma_oris']
    w_sig = readout_pars['w_sig']
    b_sig = readout_pars['b_sig']

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
    noise_ref = constant_ssn_pars['noise_ref']
    noise_target = constant_ssn_pars['noise_target']
    
    return vmap_model(logJ_2x2, logs_2x2, c_E, c_I, f_E, f_I, w_sig, b_sig, sigma_oris, ssn_mid_ori_map, ssn_mid_ori_map, data, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, gE_m, gI_m, gE_s, gI_s, filter_pars, conv_pars, loss_pars, sig_noise, noise_ref, noise_target, debug_flag)
    



def test_accuracy(ssn_layer_pars, readout_pars, constant_ssn_pars, stimuli_pars, offset, ref_ori, save=None, number_trials = 5, batch_size = 5):
    '''
    Given network parameters, function generates random trials of data and calculates the accuracy per batch. 
    Input: 
        network parameters, number of trials and batch size of each trial
    Output:
        histogram of accuracies 
    
    '''
    
    all_accs = []
    noises = []
        
    for i in range(number_trials):
        
        testing_data = create_data(stimuli_pars, number = number_trials, offset = offset, ref_ori = ref_ori)
        
        constant_ssn_pars['key'], _ = random.split(constant_ssn_pars['key'])
        _, _, pred_label, _, _, trial_noise=model(ssn_layer_pars = ssn_layer_pars, readout_pars = readout_pars, constant_ssn_pars = constant_ssn_pars, data = testing_data)
        
        true_accuracy = np.sum(testing_data['label'] == pred_label)/len(testing_data['label']) 
        all_accs.append(true_accuracy)
        noises.append(trial_noise)
   
    plt.hist(all_accs)
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
   
    
    if save:
            plt.savefig(save+'.png')
    
    plt.show()  
    plt.close() 
    
    noises = np.vstack(np.asarray(noises))

    #plot_r_ref(noises, epoch_c = None, save = os.path.join(save+'_noise') )
    
    
    
    
    
def plot_w_sig(w_sig,  epochs_to_save , epoch_c = None,save=None):
    
    plt.plot(w_sig)
    plt.xlabel('Epoch')
    plt.ylabel('Values of w')
    if epoch_c:
        plt.axvline(x=epoch_c, c='r', label='criterion')
    if save:
            plt.savefig(save+'.png')
    plt.show()
    plt.close()
    
    
    


    
    
    
def new_two_stage_training(J_2x2_m, J_2x2_s, s_2x2_s, sigma_oris_s, c_E, c_I, f_E, f_I, w_sig, b_sig, constant_ssn_pars, stimuli_pars, epochs_to_save, results_filename = None, batch_size=20, ref_ori = 55, offset = 5, epochs=1, eta=10e-4, second_eta=None, sig_noise = None, test_size = None, noise_type='additive', results_dir = None, early_stop = 0.7, extra_stop = 20, ssn_ori_map=None):
    
    '''
    Training function for two layer model in two stages: once readout layer is trained until early_stop (first stage), extra epochs are ran without updating, and then SSN layer parameters are trained (second stage). Second stage is nested in first stage. Accuracy is calculated on testing set before training and after first stage. 
    Inputs:
        individual parameters of the model
    Outputs:
        
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
    r_refs = []
    save_w_sigs = []
    save_w_sigs.append(w_sig[:5])
    epoch = 0
    
  
    #Take logs of parameters
    logJ_2x2_s = take_log(J_2x2_s)
    logs_2x2 = np.log(s_2x2_s)
    logJ_2x2_m = take_log(J_2x2_m)
    logJ_2x2 = [logJ_2x2_m, logJ_2x2_s]
    sigma_oris = np.log(sigma_oris_s)
    f_E = np.log(f_E)
    f_I = np.log(f_I)
    
    constant_ssn_pars['key'] = random.PRNGKey(42) #ADD SEED

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
    constant_ssn_pars['sigma_oris']=sigma_oris
    readout_pars = dict(w_sig = w_sig, b_sig = b_sig)
    ssn_layer_pars = dict(logJ_2x2 = logJ_2x2, c_E = c_E, c_I = c_I, f_E = f_E, f_I = f_I)#, sigma_oris = sigma_oris)
    loss_pars = constant_ssn_pars['loss_pars']
    
    print(constant_ssn_pars['ssn_mid_ori_map'])
    
    test_size = batch_size if test_size is None else test_size
        
    #Initialise optimizer
    optimizer = optax.adam(eta)
    readout_state = optimizer.init(readout_pars)
    
    print('Training model for {} epochs  with learning rate {}, sig_noise {} at offset {}, lam_w {}, batch size {}, noise_type {}'.format(epochs, eta, sig_noise, offset, loss_pars.lambda_w, batch_size, noise_type))
    print('Loss parameters dx {}, w {} '.format( constant_ssn_pars['loss_pars'].lambda_dx, constant_ssn_pars['loss_pars'].lambda_w))

    epoch_c = epochs
    loop_epochs  = epochs
    flag=True
    
    #Initialise csv file
    if results_filename:
        print('Saving results to csv ', results_filename)
    else:
        print('#### NOT SAVING! ####')
    
    loss_and_grad_readout = jax.value_and_grad(loss, argnums=1, has_aux = True)
    loss_and_grad_ssn = jax.value_and_grad(loss, argnums=0, has_aux = True)
    
    #Test accuracy before training

    #test_accuracy(ssn_layer_pars, readout_pars, constant_ssn_pars, stimuli_pars, offset, ref_ori, save=os.path.join(results_dir+ '_before_training'), number_trials = 20, batch_size = 500)
   

    while epoch < loop_epochs+1:
      
        start_time = time.time()
        epoch_loss = 0 
           
        #Load next batch of data and convert
        train_data = create_data(stimuli_pars, number = batch_size, offset = offset, ref_ori = ref_ori)
        #train_data = util.load(os.path.join(os.getcwd(), 'results', 'data', 'two_stage', 'noisy_2',  'train_data'+str(epoch)))
        #util.save_h5(os.path.join(os.getcwd(), 'results', 'data', 'two_stage', 'noisy_2', 'train_data'+str(epoch)), train_data)
        
       
        if epoch ==epoch_c+extra_stop:
            debug_flag = True
        else:
            debug_flag = False
            

        constant_ssn_pars['key'], _ = random.split(constant_ssn_pars['key'])
        constant_ssn_pars['noise_ref'] =  sig_noise*jax.random.normal(constant_ssn_pars['key'], shape=(batch_size, w_sig.shape[0]))
        constant_ssn_pars['key'], _ = random.split(constant_ssn_pars['key'])
        constant_ssn_pars['noise_target'] =  sig_noise*jax.random.normal(constant_ssn_pars['key'], shape=(batch_size, w_sig.shape[0]))
        
        #Compute loss and gradient
        [epoch_loss, [epoch_all_losses, train_true_acc, train_delta_x, train_x, train_r_ref]], grad =loss_and_grad_readout(ssn_layer_pars, readout_pars, constant_ssn_pars, train_data , debug_flag)
        
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
            #Compute loss and gradient
            constant_ssn_pars['key'], _ = random.split(constant_ssn_pars['key'])
            constant_ssn_pars['noise_ref'] =  sig_noise*jax.random.normal(constant_ssn_pars['key'], shape=(batch_size, w_sig.shape[0]))
            constant_ssn_pars['key'], _ = random.split(constant_ssn_pars['key'])
            constant_ssn_pars['noise_target'] =  sig_noise*jax.random.normal(constant_ssn_pars['key'], shape=(batch_size, w_sig.shape[0]))
            [val_loss, [val_all_losses, true_acc, val_delta_x, val_x, _ ]], _= loss_and_grad_readout(ssn_layer_pars, readout_pars, constant_ssn_pars, test_data)
            val_time = time.time() - start_time
            
            print('Training loss: {} ¦ Validation -- loss: {}, true accuracy: {}, at epoch {}, (time {}, {}), '.format(epoch_loss, val_loss, true_acc, epoch, epoch_time, val_time))

            if epoch%50 ==0:
                    print('Training accuracy: {}, all losses{}'.format(np.mean(np.asarray(train_accs[-20:])), epoch_all_losses))
            val_loss_per_epoch.append([val_loss, int(epoch)])
            val_sig_input.append([val_delta_x, epoch])
            val_sig_output.append(val_x)
            val_accs.append(true_acc)
            
            if results_filename:
                save_params = save_params_dict_two_stage(ssn_layer_pars, readout_pars, true_acc, epoch)
                
                #Initialise results file
                if epoch==0:
                        results_handle = open(results_filename, 'w')
                        results_writer = csv.DictWriter(results_handle, fieldnames=save_params.keys(), delimiter=',')
                        results_writer.writeheader()
                        
                results_writer.writerow(save_params)

            
        #Early stop in first stage of training
        if epoch>20 and flag and np.mean(np.asarray(train_accs[-20:]))>early_stop:
            epoch_c = epoch
            print('Early stop: {} accuracy achieved at epoch {}'.format(early_stop, epoch))
            loop_epochs = epoch_c + extra_stop
            save_dict = dict(training_accuracy = train_true_acc)
            save_dict.update(readout_pars)
            #util.save_h5(os.path.join(results_dir+'dict_0'), save_dict) #save  true_acc accuracy here
       
            flag=False

        #Only update parameters before criterion
        if epoch < epoch_c:        

            updates, readout_state = optimizer.update(grad, readout_state)
            readout_pars = optax.apply_updates(readout_pars, updates)
            save_w_sigs.append(readout_pars['w_sig'][:5])
        
        
    
        #Start second stage of training after reaching criterion or after given number of epochs
        if (flag == False and epoch>=epoch_c+extra_stop) or (flag == True and epoch==loop_epochs):
            
   
            #Final save before second stagenn
            if results_filename:
                save_params = save_params_dict_two_stage(ssn_layer_pars, readout_pars, true_acc, epoch)
                results_writer.writerow(save_params)
            

            final_epoch = epoch
            print('Entering second stage at epoch {}'.format(epoch))
            
            #test_accuracy(ssn_layer_pars, readout_pars, constant_ssn_pars, stimuli_pars, offset, ref_ori, save=os.path.join(results_dir+'_breaking'), number_trials = 20, batch_size = 500)
            
#############START TRAINING NEW STAGE ##################################
            
            #Initialise second optimizer
            ssn_layer_state = optimizer.init(ssn_layer_pars)
            epoch = 1                                                  
            second_eta = eta if second_eta is None else second_eta
            
            for epoch in range(epoch, epochs+1):
                
                #Load next batch of data and convert
                train_data = create_data(stimuli_pars, number = batch_size, offset = offset, ref_ori = ref_ori)
              
                
                
                #Compute loss and gradient
                constant_ssn_pars['key'], _ = random.split(constant_ssn_pars['key'])
                constant_ssn_pars['noise_ref'] =  sig_noise*jax.random.normal(constant_ssn_pars['key'], shape=(batch_size, w_sig.shape[0]))
                constant_ssn_pars['key'], _ = random.split(constant_ssn_pars['key'])
                constant_ssn_pars['noise_target'] =  sig_noise*jax.random.normal(constant_ssn_pars['key'], shape=(batch_size, w_sig.shape[0]))
                
                [epoch_loss, [epoch_all_losses, train_true_acc, train_delta_x, train_x, train_r_ref]], grad =loss_and_grad_ssn(ssn_layer_pars, readout_pars, constant_ssn_pars, train_data, debug_flag)

                all_losses = np.hstack((all_losses, epoch_all_losses))
                training_losses.append(epoch_loss)
                train_accs.append(train_true_acc)
                train_sig_input.append(train_delta_x)
                train_sig_output.append(train_x)
                r_refs.append(train_r_ref)
               
                #Save the parameters given a number of epochs
                if epoch in epochs_to_save:

                    #Evaluate model 
                    test_data = create_data(stimuli_pars, number = test_size, offset = offset, ref_ori = ref_ori)
                    
                    
                    start_time = time.time()
                    constant_ssn_pars['key'], _ = random.split(constant_ssn_pars['key'])
                    constant_ssn_pars['noise_ref'] =  sig_noise*jax.random.normal(constant_ssn_pars['key'], shape=(batch_size, w_sig.shape[0]))
                    constant_ssn_pars['key'], _ = random.split(constant_ssn_pars['key'])
                    constant_ssn_pars['noise_target'] =  sig_noise*jax.random.normal(constant_ssn_pars['key'], shape=(batch_size, w_sig.shape[0]))
                    
                    [val_loss, [val_all_losses, true_acc, val_delta_x, val_x, _]], _= loss_and_grad_ssn(ssn_layer_pars, readout_pars, constant_ssn_pars, test_data)
                    val_time = time.time() - start_time
                    print('Training loss: {} ¦ Validation -- loss: {}, true accuracy: {}, at epoch {}, (time {}, {})'.format(epoch_loss, val_loss, true_acc, epoch, epoch_time, val_time))
                    
                    if epoch%50 ==0:
                        print('Training accuracy: {}, all losses{}'.format(train_true_acc, epoch_all_losses))
                    
                    val_loss_per_epoch.append([val_loss, epoch+final_epoch])
                    val_sig_input.append([val_delta_x, epoch+final_epoch])
                    val_sig_output.append(val_x)
                
                    if results_filename:
                            save_params = save_params_dict_two_stage(ssn_layer_pars, readout_pars, true_acc, epoch = epoch+final_epoch)
                            results_writer.writerow(save_params)
                            
                #Update parameters
                updates, ssn_layer_state = optimizer.update(grad, ssn_layer_state)
                ssn_layer_pars = optax.apply_updates(ssn_layer_pars, updates)
                
                if epoch ==1:
                    save_dict = dict(training_accuracy = train_true_acc)
                    save_dict.update(readout_pars)
                    #util.save_h5(os.path.join(results_dir+'dict_4'), save_dict)
                    
            final_epoch_2 =  epoch+final_epoch       
           
            break
################################################################################

                
        epoch+=1
    save_w_sigs = np.asarray(np.vstack(save_w_sigs))
    plot_w_sig(save_w_sigs, epochs_to_save[:len(save_w_sigs)], epoch_c, save = os.path.join(results_dir+'_w_sig_evolution') )


    #THIRD STAGE
    print('Entering third stage')
    for epoch in range(final_epoch_2+1, final_epoch_2+2):
        print(epoch)
        train_data = create_data(stimuli_pars, number = batch_size, offset = offset, ref_ori = ref_ori)

        #Compute loss and gradient
        constant_ssn_pars['key'], _ = random.split(constant_ssn_pars['key'])
        constant_ssn_pars['noise_ref'] =  sig_noise*jax.random.normal(constant_ssn_pars['key'], shape=(batch_size, w_sig.shape[0]))
        constant_ssn_pars['key'], _ = random.split(constant_ssn_pars['key'])
        constant_ssn_pars['noise_target'] =  sig_noise*jax.random.normal(constant_ssn_pars['key'], shape=(batch_size, w_sig.shape[0]))
        [epoch_loss, [epoch_all_losses, train_true_acc, train_delta_x, train_x, train_r_ref]], grad =loss_and_grad_readout(ssn_layer_pars, readout_pars, constant_ssn_pars, train_data , debug_flag)

        training_losses.append(epoch_loss)
        train_accs.append(train_true_acc)
        train_sig_input.append(train_delta_x)
        train_sig_output.append(train_x)
        r_refs.append(train_r_ref)

        '''
        if epoch %10 == 0:

                #Evaluate model 
                test_data = create_data(stimuli_pars, number = test_size, offset = offset, ref_ori = ref_ori)
                start_time = time.time()
                [val_loss, [val_all_losses, true_acc, val_delta_x, val_x ]], _= loss_and_grad_ssn(ssn_layer_pars, readout_pars, constant_ssn_pars, test_data)
                val_time = time.time() - start_time
                print('Training loss: {} ¦ Validation -- loss: {}, true accuracy: {}, at epoch {}, (time {}, {}) J_2x2 {}'.format(epoch_loss, val_loss, true_acc, epoch, epoch_time, val_time, np.exp(ssn_layer_pars['logJ_2x2'][0][0,0])))

                val_loss_per_epoch.append([val_loss, epoch+final_epoch])
                val_sig_input.append([val_delta_x, epoch+final_epoch])
                val_sig_output.append(val_x)

                if results_filename:
                        save_params = save_params_dict_two_stage(ssn_layer_pars, readout_pars, true_acc, epoch = epoch+final_epoch)
                        results_writer.writerow(save_params)
        '''        
        #Update parameters
        updates, readout_state = optimizer.update(grad, readout_state)
        readout_pars = optax.apply_updates(readout_pars, updates)

   
    if flag==False:
        epoch_c = [epoch_c, extra_stop, final_epoch_2]
    #r_refs = np.vstack(np.asarray(r_refs))

    #plot_r_ref(r_refs, epoch_c = epoch_c, save = os.path.join(results_dir+'_noise') )
   
    return [ssn_layer_pars, readout_pars], np.vstack([val_loss_per_epoch]), all_losses, train_accs, train_sig_input, train_sig_output, val_sig_input, val_sig_output, epoch_c, save_w_sigs



def plot_r_ref(r_ref, epoch_c = None, save=None):
    
    plt.plot(r_ref)
    plt.xlabel('Epoch')
    plt.ylabel('noise')
    
    if epoch_c==None:
                pass
    else:
        if np.isscalar(epoch_c):
            plt.axvline(x=epoch_c, c = 'r')
        else:
            plt.axvline(x=epoch_c[0], c = 'r')
            plt.axvline(x=epoch_c[0]+epoch_c[1], c='r')
            plt.axvline(x=epoch_c[2], c='r')
    
    if save:
            plt.savefig(save+'.png')
    plt.show()
    plt.close() 
    
    
    

def loss(ssn_layer_pars, readout_pars, constant_ssn_pars, data, debug_flag=False):
    
    '''
    Function to take gradient with respect to. Output returned as two variables (jax grad takes gradient with respect to first output)
    Inputs:
        parameters assembled into dictionaries
    Ouputs: 
        total loss to take gradient with respect to
    '''
    
    total_loss, all_losses, pred_label, sig_input, x, r_ref = model(ssn_layer_pars = ssn_layer_pars, readout_pars = readout_pars, constant_ssn_pars = constant_ssn_pars, data = data, debug_flag = debug_flag)
   
    loss= np.mean(total_loss)
    all_losses = np.mean(all_losses, axis = 0)
    r_ref = np.mean(r_ref, axis = 0)

    true_accuracy = np.sum(data['label'] == pred_label)/len(data['label']) 
    

        
    return loss, [all_losses, true_accuracy, sig_input, x, r_ref]





def save_params_dict_two_stage(ssn_layer_pars, readout_pars, true_acc, epoch ):
    
    '''
    Assemble trained parameters and epoch information into single dictionary for saving
    Inputs:
        dictionaries containing trained parameters
        other epoch parameters (accuracy, epoch number)
    Outputs:
        single dictionary concatenating all information to be saved
    '''
    
    
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
    save_params['f_E'] = np.exp(ssn_layer_pars['f_E'])
    save_params['f_I'] = np.exp(ssn_layer_pars['f_I'])
        
    if 'sigma_oris' in ssn_layer_pars.keys():

        if len(ssn_layer_pars['sigma_oris']) ==1:
            save_params[key] = np.exp(ssn_layer_pars[key])
        else:
            sigma_oris = dict(sigma_orisE = np.exp(ssn_layer_pars['sigma_oris'][0]), sigma_orisI = np.exp(ssn_layer_pars['sigma_oris'][1]))
            save_params.update(sigma_oris)

    #Add readout parameters
    save_params.update(readout_pars)

    return save_params














    
    
'''
def two_stage_training(J_2x2_m, J_2x2_s, s_2x2_s, sigma_oris_s, c_E, c_I, f, w_sig, b_sig, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, gE, gI, stimuli_pars, filter_pars, conv_pars, loss_pars, epochs_to_save, results_filename = None, batch_size=20, ref_ori = 55, offset = 5, epochs=1, eta=10e-4, sig_noise = None, test_size = None, noise_type='additive', results_dir = None, early_stop = 0.7, extra_stop = 20):
    
    print(locals())
    
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
    save_w_sigs.append(w_sig[:5])
    model_type =4
    epoch = 0
    
    #Initialise networks
    ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_m, filter_pars=filter_pars, J_2x2=J_2x2_m, gE = gE, gI=gI)
    ssn_mid_ori_map = ssn_mid.ori_map
    ssn_sup_ori_map = ssn_mid.ori_map
    
    #Take logs of parameters
    logJ_2x2_s =take_log(J_2x2_s)
    logs_2x2 = np.log(s_2x2_s)
    logJ_2x2_m =take_log(J_2x2_m)
    logJ_2x2 = [logJ_2x2_m, logJ_2x2_s]
    sigma_oris = np.log(sigma_oris_s)
    
    test_size = batch_size if test_size is None else test_size
    
    #Initialise vmap version of model
    vmap_model = vmap(new_model, in_axes = (None, None, [None, None], None, None, None, None, None, None, None, None, None, None, None, None, None, {'ref':0, 'target':0, 'label':0}, None, None, None, None, None, None) )
    
    #Separate parameters used in optimisation
    opt_pars, conn_pars_m, conn_pars_s= create_param_4(logJ_2x2, logs_2x2, sigma_oris, c_E, c_I, f, w_sig, b_sig, conn_pars_m, conn_pars_s)
        
    #Initialise optimizer
    optimizer = optax.adam(eta)
    opt_state = optimizer.init(opt_pars)
    
    print(opt_pars)
    
    print('Training model for {} epochs  with learning rate {}, sig_noise {} at offset {}, lam_w {}, batch size {}, noise_type {}'.format(epochs, eta, sig_noise, offset, loss_pars.lambda_w, batch_size, noise_type))
    
    #Define test data - no need to iterate
    initial_data = create_data(stimuli_pars, number = test_size, offset = offset, ref_ori = ref_ori)
                                                            
    val_loss, [all_losses, true_acc, delta_x, x]= loss(opt_pars, ssn_mid_ori_map, ssn_sup_ori_map, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, gE, gI, initial_data, filter_pars, conv_pars, loss_pars, sig_noise, noise_type, vmap_model, model_type)
    print('Before training  -- loss: {}, true accuracy: {} ,'.format(np.round(float(val_loss), 3), np.round(true_acc, 3)))
    val_loss_per_epoch.append([val_loss, epoch])
    train_sig_input.append(delta_x)
    val_sig_input.append(delta_x)
    train_sig_output.append(x)
    val_sig_output.append(x)
    
    epoch_c = epochs
    flag=True
    
    #Save initial parameters
    initial_save_params = save_params_dict_two_stage(logJ_2x2, logs_2x2, sigma_oris, c_E, c_I, f, w_sig, b_sig, true_acc, epoch)
    
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
    
    #uuracy(stimuli_pars, offset, ref_ori, J_2x2_m, J_2x2_s, s_2x2_s, c_E, c_I, f, w_sig, b_sig, sigma_oris, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, gE, gI, filter_pars, conv_pars, loss_pars, sig_noise, noise_type, save =os.path.join(results_dir+ '_before_training'), number_trials = 20, batch_size = 500, vmap_model = vmap_model)
    
    for epoch in range(1, epochs+1):
        start_time = time.time()
        epoch_loss = 0 
           
        #Load next batch of data and convert
        train_data = create_data(stimuli_pars, number = batch_size, offset = offset, ref_ori = ref_ori)
    
        if epoch ==epoch_c+20:
            debug_flag = True
        else:
            debug_flag = False
        
        #Compute loss and gradient
        [epoch_loss, [epoch_all_losses, train_true_acc, train_delta_x, train_x]], grad =loss_and_grad(opt_pars, ssn_mid_ori_map, ssn_sup_ori_map, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, gE, gI, train_data, filter_pars, conv_pars, loss_pars, sig_noise, noise_type, vmap_model, model_type, debug_flag)

        training_losses.append(epoch_loss)
        all_losses = np.hstack((all_losses, epoch_all_losses))
        train_accs.append(train_true_acc)
        train_sig_input.append(train_delta_x)
        train_sig_output.append(train_x)
 
        epoch_time = time.time() - start_time
        

        #Save the parameters given a number of epochs
        if epoch in epochs_to_save:

            
            #Evaluate model 
            test_data = create_data(stimuli_pars, number = test_size, offset = offset, ref_ori = ref_ori)
            start_time = time.time()
            [val_loss, [val_all_losses, true_acc, val_delta_x, val_x ]], _= loss_and_grad(opt_pars, ssn_mid_ori_map, ssn_sup_ori_map, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, gE, gI, test_data, filter_pars, conv_pars, loss_pars, sig_noise, noise_type, vmap_model, model_type)
            val_time = time.time() - start_time
            print('Training loss: {} ¦ Validation -- loss: {}, true accuracy: {}, at epoch {}, (time {}, {})'.format(epoch_loss, val_loss, true_acc, epoch, epoch_time, val_time))
            #if epoch%50 ==0:
            #            print('Training accuracy: {}, all losses{}'.format(np.mean(np.asarray(train_accs[-20:])), epoch_all_losses))
            val_loss_per_epoch.append([val_loss, int(epoch)])
            val_sig_input.append(val_delta_x)
            val_sig_output.append(val_x)
            val_accs.append(true_acc)
        
        #Early stop in first stage of training
        if model_type ==4 and epoch>20 and flag and np.mean(np.asarray(train_accs[-20:]))>early_stop:
            epoch_c = epoch
            print('Early stop: {} accuracy achieved at epoch {}'.format(early_stop, epoch))
            flag=False

        #Only update parameters before criterion
        if epoch < epoch_c:        
            updates, opt_state = optimizer.update(grad, opt_state)
            opt_pars = optax.apply_updates(opt_pars, updates)
            if model_type==4:
                save_w_sigs.append(opt_pars['w_sig'][:5])
         
        #Start second stage of training after reaching criterion or after given number of epochs
        if (flag == False and epoch>=epoch_c+extra_stop) or (flag == True and epoch==epochs):
            
            #Histogram accuracies obtained
            if flag==False:
                data = np.histogram(np.asarray(train_accs[-20:]))
                np.save(os.path.join(results_dir+'histogram_count.npy'), data)
            
            trained_w_sig = opt_pars['w_sig']
            trained_b_sig = opt_pars['b_sig']
            final_epoch = epoch
            print('Entering second stage at epoch {}'.format(epoch))
            
            #test_accuracy(stimuli_pars, offset, ref_ori, J_2x2_m, J_2x2_s, s_2x2_s, c_E, c_I, f, w_sig, b_sig, sigma_oris, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, gE, gI, filter_pars, conv_pars, loss_pars, sig_noise, noise_type, save =os.path.join(results_dir+'_breaking'), number_trials = 20, batch_size = 500, vmap_model = vmap_model)
            
#############START TRAINING NEW STAGE ##################################
            epoch = 0
            model_type = 5
            opt_pars_2, conn_pars_m, conn_pars_s = create_param_5(logJ_2x2, logs_2x2, sigma_oris, c_E, c_I, f, trained_w_sig, trained_b_sig, conn_pars_m, conn_pars_s)
            opt_state_2 = optimizer.init(opt_pars_2)
                                                                
            
            for epoch in range(1, epochs+1):
                #Load next batch of data and convert
                train_data = create_data(stimuli_pars, number = batch_size, offset = offset, ref_ori = ref_ori)
                
                if epoch ==1:
                    debug_flag = True
                else:
                    debug_flag = False
                
                #Compute loss and gradient
                [epoch_loss, [epoch_all_losses, train_true_acc, train_delta_x, train_x]], grad =loss_and_grad(opt_pars_2, ssn_mid_ori_map, ssn_sup_ori_map, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, gE, gI, train_data, filter_pars, conv_pars, loss_pars, sig_noise, noise_type, vmap_model, model_type, debug_flag)

                all_losses = np.hstack((all_losses, epoch_all_losses))
                training_losses.append(epoch_loss)
                train_accs.append(train_true_acc)
                train_sig_input.append(train_delta_x)
                train_sig_output.append(train_x)
               
                #Save the parameters given a number of epochs
                if epoch in epochs_to_save:

                    #Evaluate model 
                    test_data = create_data(stimuli_pars, number = test_size, offset = offset, ref_ori = ref_ori)
                    start_time = time.time()
                    [val_loss, [val_all_losses, true_acc, val_delta_x, val_x ]], _= loss_and_grad(opt_pars_2, ssn_mid_ori_map, ssn_sup_ori_map, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, gE, gI, test_data, filter_pars, conv_pars, loss_pars, sig_noise, noise_type, vmap_model, model_type)
                    val_time = time.time() - start_time
                    print('Training loss: {} ¦ Validation -- loss: {}, true accuracy: {}, at epoch {}, (time {}, {})'.format(epoch_loss, val_loss, true_acc, epoch, epoch_time, val_time))
                    if epoch%50 ==0:
                        print('Training accuracy: {}, all losses{}'.format(train_true_acc, epoch_all_losses))
                    
                    val_loss_per_epoch.append([val_loss, epoch+final_epoch])
                    val_sig_input.append(val_delta_x)
                    val_sig_output.append(val_x)
                
                updates, opt_state_2 = optimizer.update(grad, opt_state_2)
                opt_pars_2 = optax.apply_updates(opt_pars_2, updates)
                    
                if epoch in epochs_to_save:
                    if results_filename:
                            save_params = save_params_dict_two_stage(opt_pars_2['logJ_2x2'], logs_2x2, opt_pars_2['sigma_oris'], opt_pars_2['c_E'], opt_pars_2['c_I'], opt_pars_2['f'], trained_w_sig, trained_b_sig, true_acc=true_acc, epoch=epoch)
                            results_writer.writerow(save_params)
                    
            break
################################################################################

            
    
        #Save new optimized parameters
        if epoch in epochs_to_save:
            if results_filename:
                save_params = save_params_dict_two_stage(logJ_2x2, logs_2x2, sigma_oris, c_E, c_I, f, opt_pars['w_sig'], opt_pars['b_sig'], true_acc=true_acc, epoch=epoch)
                results_writer.writerow(save_params)
        
        

    save_w_sigs = np.asarray(np.vstack(save_w_sigs))
    plot_w_sig(save_w_sigs, epochs_to_save[:len(save_w_sigs)], epoch_c, save = os.path.join(results_dir+'_w_sig_evolution') )

    

    trained_J_2x2_m = sep_exponentiate(opt_pars_2['logJ_2x2'][0])
    trained_J_2x2_s = sep_exponentiate(opt_pars_2['logJ_2x2'][1])
    #trained_s_2x2 = opt_pars_2['logs_2x2']
    trained_c_E = opt_pars_2['c_E']
    trained_c_I = opt_pars_2['c_I']
    trained_f = opt_pars_2['f']
    trained_sigma_oris = np.exp(sigma_oris)

    opt_pars.update(opt_pars_2)
 

    print('valloss ', val_loss_per_epoch)
    #test_accuracy(stimuli_pars, offset, ref_ori,  trained_J_2x2_m,  trained_J_2x2_s,  trained_s_2x2,  trained_c_E,  trained_c_I,  trained_f,  trained_w_sig,  trained_b_sig, trained_sigma_oris, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, gE, gI, filter_pars, conv_pars, loss_pars, sig_noise, noise_type, save =os.path.join(results_dir+'_inside_function'), number_trials = 20, batch_size = 500, vmap_model = vmap_model)
   
   
    return opt_pars, np.vstack([val_loss_per_epoch]), all_losses, train_accs, train_sig_input, train_sig_output, val_sig_input, val_sig_output, [epoch_c, extra_stop], save_w_sigs
    

def loss_old(pars, ssn_mid_ori_map, ssn_sup_ori_map, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, gE, gI, data, filter_pars, conv_pars, loss_pars, sig_noise, noise_type, vmap_model, model_type=1, debug_flag=False):
    
    
    
    #Separate parameters
    if model_type==1:
        logJ_2x2, logs_2x2, c_E, c_I, f,  w_sig, b_sig, sigma_oris = separate_param_1(pars, conn_pars_s)
        
    if model_type==2:
        logJ_2x2, logs_2x2, c_E, c_I, w_sig, b_sig, sigma_oris = separate_param_2(pars, conn_pars_m, conn_pars_s)
    
    if model_type==3:
        logJ_2x2, logs_2x2, c_E, c_I, w_sig, b_sig, sigma_oris = separate_param_3(pars, conn_pars_m, conn_pars_s)
    
    if model_type ==4:
        logJ_2x2, logs_2x2, c_E, c_I, f, w_sig, b_sig, sigma_oris = separate_param_4(pars, conn_pars_m, conn_pars_s)
        
    if model_type ==5:
        logJ_2x2, logs_2x2, c_E, c_I, f, w_sig, b_sig, sigma_oris = separate_param_5(pars, conn_pars_m, conn_pars_s)
    
    total_loss, all_losses, pred_label, sig_input, x= vmap_model(ssn_mid_ori_map, ssn_sup_ori_map, logJ_2x2, logs_2x2, c_E, c_I, f, w_sig, b_sig, sigma_oris, ssn_pars, grid_pars, 
conn_pars_m, conn_pars_s, gE, gI, data, filter_pars, conv_pars, loss_pars, sig_noise, noise_type, debug_flag)
    
    
    loss= np.mean(total_loss)
    all_losses = np.mean(all_losses, axis = 0)
        
    true_accuracy = np.sum(data['label'] == pred_label)/len(data['label']) 
       
        
    return loss, [all_losses, true_accuracy, sig_input, x]
    


def train_SSN_vmap(J_2x2_m, J_2x2_s, s_2x2_s, sigma_oris, c_E, c_I, f, w_sig, b_sig, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, gE, gI, stimuli_pars, filter_pars, conv_pars, loss_pars, epochs_to_save, results_filename = None, batch_size=20, ref_ori = 55, offset = 5, epochs=1, eta=10e-4, sig_noise = None, test_size = None, noise_type='additive', model_type=1, readout_pars=None, results_dir = None, early_stop = 0.7):
    
    
    #Initialize loss
    val_loss_per_epoch = []
    training_losses=[]
    train_accs = []
    train_sig_input = []
    train_sig_output = []
    val_sig_input = []
    val_sig_output = []
    save_w_sigs = []
    save_w_sigs.append(w_sig[:5])
    
    #Initialise networks
    ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_m, filter_pars=filter_pars, J_2x2=J_2x2_m, gE = gE, gI=gI)
    ssn_mid_ori_map = ssn_mid.ori_map
    ssn_sup_ori_map = ssn_mid.ori_map
    
    #Take logs of parameters
    logJ_2x2_s =take_log(J_2x2_s)
    logs_2x2 = np.log(s_2x2_s)
    logJ_2x2_m =take_log(J_2x2_m)
    logJ_2x2 = [logJ_2x2_m, logJ_2x2_s]
    sigma_oris = np.log(sigma_oris)
    
    test_size = batch_size if test_size is None else test_size
    
    #Initialise vmap version of model
    vmap_model = vmap(new_model, in_axes = (None, None, [None, None], None, None, None, None, None, None, None, None, None, None, None, None, None, {'ref':0, 'target':0, 'label':0}, None, None, None, None, None, None) )
    
    #Separate parameters used in optimisation
    if model_type ==1:
        opt_pars, conn_pars_s = create_param_1(logJ_2x2, logs_2x2, sigma_oris, c_E, c_I, f, w_sig, b_sig, conn_pars_s)
    
    if model_type==2:
        opt_pars, conn_pars_m, conn_pars_s = create_param_2(logJ_2x2, logs_2x2, sigma_oris, c_E, c_I, w_sig, b_sig, conn_pars_m, conn_pars_s)
    
    if model_type==3:
        opt_pars, conn_pars_m, conn_pars_s = create_param_3(logJ_2x2, logs_2x2, sigma_oris, c_E, c_I, w_sig, b_sig, conn_pars_m, conn_pars_sn_pars)
    
    if model_type ==4:
        opt_pars, conn_pars_m, conn_pars_s= create_param_4(logJ_2x2, logs_2x2, sigma_oris, c_E, c_I, f, w_sig, b_sig, conn_pars_m, conn_pars_s)
        
    if model_type ==5:
        opt_pars, conn_pars_m, conn_pars_s = create_param_5(logJ_2x2, logs_2x2, sigma_oris, c_E, c_I, f, w_sig, b_sig, conn_pars_m, conn_pars_s)
    
    #Initialise optimizer
    optimizer = optax.adam(eta)
    opt_state = optimizer.init(opt_pars)
    
    print(opt_pars)
    
    print('Training model with learning rate {}, sig_noise {} at offset {}, lam_w {}, batch size {}, noise_type {}'.format(eta, sig_noise, offset, loss_pars.lambda_w, batch_size, noise_type))
    
    #Define test data - no need to iterate
    initial_data = create_data(stimuli_pars, number = test_size, offset = offset, ref_ori = ref_ori)
                                                            
    val_loss, [all_losses, true_acc, delta_x, x]= loss(opt_pars, ssn_mid_ori_map, ssn_sup_ori_map, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, gE, gI, initial_data, filter_pars, conv_pars, loss_pars, sig_noise, noise_type, vmap_model, model_type)
    print('Before training  -- loss: {}, true accuracy: {} ,'.format(np.round(float(val_loss), 3), np.round(true_acc, 3)))
    val_loss_per_epoch.append(val_loss)
    train_sig_input.append(delta_x)
    val_sig_input.append(delta_x)
    train_sig_output.append(x)
    val_sig_output.append(x)
    
    epoch_c = epochs
    flag=True
    
    #Save initial parameters
    initial_save_params = save_params_dict(opt_pars=opt_pars, true_acc=true_acc, epoch=0)
    
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
    #test_accuracy(stimuli_pars, offset, ref_ori, J_2x2_m, J_2x2_s, s_2x2_s, c_E, c_I, f, w_sig, b_sig, sigma_oris, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, gE, gI, filter_pars, conv_pars, loss_pars, sig_noise, noise_type, save =os.path.join(results_dir+ '_before_training'), number_trials = 20, batch_size = 500, vmap_model = vmap_model)
    
    for epoch in range(1, epochs+1):
        
        start_time = time.time()
        epoch_loss = 0 
           
        #Load next batch of data and convert
        train_data = create_data(stimuli_pars, number = batch_size, offset = offset, ref_ori = ref_ori)

        #Compute loss and gradient
        [epoch_loss, [epoch_all_losses, train_true_acc, train_delta_x, train_x]], grad =loss_and_grad(opt_pars, ssn_mid_ori_map, ssn_sup_ori_map, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, gE, gI, train_data, filter_pars, conv_pars, loss_pars, sig_noise, noise_type, vmap_model, model_type)

        training_losses.append(epoch_loss)
        all_losses = np.hstack((all_losses, epoch_all_losses))
        train_accs.append(train_true_acc)
        train_sig_input.append(train_delta_x)
        train_sig_output.append(train_x)
 
        epoch_time = time.time() - start_time
        

        #Save the parameters given a number of epochs
        if epoch in epochs_to_save:
            
            #Evaluate model 
            test_data = create_data(stimuli_pars, number = test_size, offset = offset, ref_ori = ref_ori)
            start_time = time.time()
            [val_loss, [val_all_losses, true_acc, val_delta_x, val_x ]], _= loss_and_grad(opt_pars, ssn_mid_ori_map, ssn_sup_ori_map, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, gE, gI, test_data, filter_pars, conv_pars, loss_pars, sig_noise, noise_type, vmap_model, model_type)
            val_time = time.time() - start_time
            print('Training loss: {} ¦ Validation -- loss: {}, true accuracy: {} ({}), at epoch {}, (time {}, {})'.format(epoch_loss, val_loss, np.round(true_acc, 5), np.round(train_true_acc, 5), epoch, np.round(epoch_time, 3), val_time))
            val_loss_per_epoch.append(val_loss)
            val_sig_input.append(val_delta_x)
            val_sig_output.append(val_x)
            
        if model_type ==4 and epoch>7 and flag and np.mean(np.asarray(train_accs[-7:]))>early_stop:
            epoch_c = epoch
            print('Early stop: {} accuracy achieved at epoch {}'.format(early_stop, epoch))
            print(training_accs)
            flag=False
                
        if epoch < epoch_c:        
            updates, opt_state = optimizer.update(grad, opt_state)
            opt_pars = optax.apply_updates(opt_pars, updates)
            if model_type==4:
                save_w_sigs.append(opt_pars['w_sig'][:5])
            
        if epoch>=epoch_c+100:
            w_sig = opt_pars['w_sig']
            b_sig = opt_pars['b_sig']
            test_accuracy(stimuli_pars, offset, ref_ori, J_2x2_m, J_2x2_s, s_2x2_s, c_E, c_I, f, w_sig, b_sig, sigma_oris, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, gE, gI, filter_pars, conv_pars, loss_pars, sig_noise, noise_type, save =os.path.join(results_dir+'_breaking'), number_trials = 20, batch_size = 500, vmap_model = vmap_model)
            
            print('Breaking at epoch {}'.format(epoch))
            break
    
        #Save new optimized parameters
        if epoch in epochs_to_save:
            if results_filename:
                save_params = save_params_dict(opt_pars=opt_pars, true_acc=true_acc, epoch=epoch)
                results_writer.writerow(save_params)
    
    #Reparametize parameters
    signs=np.array([[1, -1], [1, -1]])
    #if 'logJ_2x2' in opt_pars.keys():
    #    opt_pars['logJ_2x2'] = np.exp(opt_pars['logJ_2x2'])*signs
    #if 'logs_2x2' in opt_pars.keys():
    #    opt_pars['logs_2x2'] = np.exp(opt_pars['logs_2x2'])
    if 'sigma_oris' in opt_pars.keys():
        opt_pars['sigma_oris'] = np.exp(opt_pars['sigma_oris'])
        
        
    if model_type ==4:
        save_w_sigs = np.asarray(np.vstack(save_w_sigs))
        plot_w_sig(save_w_sigs, epochs_to_save[:len(save_w_sigs)], epoch_c, save = os.path.join(results_dir+'_w_sig_evolution') )
        w_sig = opt_pars['w_sig']
        b_sig = opt_pars['b_sig']
    
    if model_type==5:
        J_2x2 = opt_pars['logJ_2x2']
        s_2x2 = opt_pars['logs_2x2']
        c_E = opt_pars['c_E']
        c_I = opt_pars['c_I']
        sigma_oris = np.exp(sigma_oris)

        
    if model_type ==1:

        J_2x2 = opt_pars['logJ_2x2']
        s_2x2 = opt_pars['logs_2x2']
        c_E = opt_pars['c_E']
        c_I = opt_pars['c_I']
        w_sig = opt_pars['w_sig']
        b_sig = opt_pars['b_sig']
        sigma_oris = np.exp(sigma_oris)

    #test_accuracy(stimuli_pars, offset, ref_ori, J_2x2_m, J_2x2_s, s_2x2_s, c_E, c_I, f, w_sig, b_sig, sigma_oris, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, gE, gI, filter_pars, conv_pars, loss_pars, sig_noise, noise_type, save =os.path.join(results_dir+'_inside_function'), number_trials = 20, batch_size = 500, vmap_model = vmap_model)
   
   
    return opt_pars, val_loss_per_epoch, all_losses, train_accs, train_sig_input, train_sig_output, val_sig_input, val_sig_output, epoch_c, save_w_sigs
'''