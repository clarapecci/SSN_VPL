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
from util import create_gratings
from training import exponentiate, constant_to_vec, create_data, obtain_fixed_point, exponentiate, constant_to_vec
from SSN_classes_jax_jit import SSN2DTopoV1_AMPAGABA_ONOFF

def find_response(opt_pars, ssn_pars, grid_pars, conn_pars, gE, gI, filter_pars, conv_pars, stimuli_pars, ref_ori, offset, inhibition=False):
    
    J_2x2, s_2x2 = exponentiate(opt_pars)
    constant_vector = constant_to_vec(opt_pars['c_E'], opt_pars['c_I'])
    
    ssn=SSN2DTopoV1_AMPAGABA_ONOFF(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars, gE=gE, gI=gI, filter_pars=filter_pars, J_2x2=J_2x2, s_2x2=s_2x2, sigma_oris = opt_pars['sigma_oris'])
    
    all_responses_E=[]
    all_responses_I = []
    
    if ref_ori==None:
        ref_ori = ssn.ori_map[4,4]

    for i in range(0,100):
        test_data = create_data(stimuli_pars, number = 1, offset = offset, ref_ori = ref_ori)
        stimuli = test_data['ref'][0]

        output_ref=np.matmul(ssn.gabor_filters, stimuli) + constant_vector
        SSN_input_ref=np.maximum(0, output_ref)

    #Find the fixed point 
        x_ref, _, _ = obtain_fixed_point(ssn, SSN_input_ref, conv_pars, inhibition = inhibition)
        
        all_responses_E.append(x_ref[0])
        all_responses_I.append(x_ref[1])
    
    return np.asarray(all_responses_E), np.asarray(all_responses_I)

def findRmax(opt_pars, ssn_pars, grid_pars, conn_pars, gE, gI, filter_pars, conv_pars, stimuli_pars, ref_ori, offset, inhibition=False):
    
    responses = find_response(opt_pars, ssn_pars, grid_pars, conn_pars, gE, gI, filter_pars, conv_pars, stimuli_pars, ref_ori, offset, inhibition)

    Rmax_E = responses[0].max()
    Rmax_I = responses[1].max()
    
    return Rmax_E, Rmax_I


def plot_results(results_file, title=None, bernoulli=True):
    
    results = pd.read_csv(results_file, header = 0)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,8))

    results.plot(x='epoch', y=["J_EE", "J_EI", "J_IE", "J_II"], ax=axes[0,0])
    results.plot(x='epoch', y=["s_EE", "s_EI", "s_IE", "s_II"], ax = axes[0,1])
    results.plot(x='epoch', y=["c_E", "c_I"], ax = axes[1,0])
    if bernoulli == True:
        results.plot(x='epoch', y = ['val_accuracy', 'ber_accuracy'], ax = axes[1,1])
    else:
        results.plot(x='epoch', y = ['val_accuracy'], ax = axes[1,1])
 
    
    if title:
        fig.suptitle(title)
    fig.show()
    
    
def plot_losses(training_losses, validation_losses, epochs_to_save):
    plt.plot(training_losses.T, label = ['Binary cross entropy', 'Avg_dx', 'R_max', 'w', 'b', 'Training total'] )
    plt.plot(epochs_to_save, validation_losses, label='Validation')
    plt.legend()
    plt.title('Training losses')
    plt.show()
    

    
    
def assemble_pars(all_pars, matrix = True):
    '''
    Take parameters from csv file and 
    
    '''
    pre_train = np.asarray(all_pars.iloc[0].tolist())
    post_train =  np.asarray(all_pars.iloc[-1].tolist())

    if matrix == True:
        matrix_pars = lambda Jee, Jei, Jie, Jii: np.array([[Jee, Jei], [Jie,  Jii]])

        pre_train = matrix_pars(*pre_train)
        post_train = matrix_pars(*post_train)
    
    
    return pre_train, post_train


def plot_acc_vs_param(to_plot, lambdas, type_param = None, param = None):
    '''
    Input:
        Matrix with shape (N+1, length of lambda) - each row corresponds to a different value of lambda, params at that value and 
        the accuracy obtained
    Output:
        Plot of the desired param against the accuracy 
    '''
    
    plt.scatter(np.abs(to_plot[:, param]).T, to_plot[:, 0].T, c = lambdas)
    plt.colorbar()
    
    plt.ylabel('Accuracy')
    
    if type_param == 'J':
        if param ==1:
            plt.xlabel('J_EE')
        if param ==2:
            plt.xlabel('J_EI')
        if param ==3:
            plt.xlabel('J_IE')
        if param ==4:
            plt.xlabel('J_II')
            
    if type_param == 's':
        if param ==1:
            plt.xlabel('s_EE')
        if param ==2:
            plt.xlabel('s_EI')
        if param ==3:
            plt.xlabel('s_IE')
        if param ==4:
            plt.xlabel('s_II')
    
    if type_param == 'c':
        if param ==1:
            plt.xlabel('c_E')
        if param ==2:
            plt.xlabel('c_I')

    plt.show()


    
def case_1(pre_param, post_param, ssn_pars, grid_pars, conn_pars, gE, gI, test_data, filter_pars, conv_pars, loss_pars, sig_noise):
    '''
    Interpolate all parameters and evaluate accuracy at each value.
    Input:
        list of pre and post values of J
        opt_pars for other optimisation parameters
        test_data
    Output:
        Matrix with shape (N+1, length of lambda) - each row corresponds to a different value of lambda, params at that value and 
        the accuracy obtained   
    '''
    
    values = []
    accuracy = []
    lambdas = np.linspace(0,1,10)
    for lamb in lambdas:
        new_param = {}

        for key in opt_pars.keys():
            new_param[key] =(1-lamb)*pre_param[key] + lamb*post_param[key]
        
        print(new_param)
        new_param['logJ_2x2'] = np.log(new_param['logJ_2x2']*signs)
        new_param['logs_2x2'] = np.log(new_param['logs_2x2'])
        
        
        val_loss, true_acc, _= vmap_eval(new_param, ssn_pars, grid_pars, conn_pars, gE, gI, test_data, filter_pars, conv_pars, loss_pars, sig_noise=2.5)
        print('lambda ', lamb, ', accuracy', true_acc)
        accuracy.append(true_acc)

    return accuracy, lambdas #np.column_stack([np.vstack(accuracy), np.vstack(values)])
    

def case_2(pre_param, post_param, opt_pars, test_data, type_param = None, index=None):
    '''
    Interpolate a single trained parameter and evaluate accuracy at each value. Produce plot of param against accuracy
    Input:
        list of pre and post values of J
        opt_pars for other optimisation parameters
        test_data
        desired param from the matrix (0,0) - J_EE ¦ (0,1) - J_EI, ¦ (1,0) - J_IE ¦ (1,1) - J_II
    Output:
        Matrix with shape (N+1, length of lambda) - each row corresponds to a different value of lambda, params at that value and 
        the accuracy obtained
        Plot of the changing parameter against accuracy
        
    '''
    values = []
    accuracy = []
    lambdas = np.linspace(0,1,10)
    parameter_matrix = np.asarray([[1,2],[3,4]]) 
    plot_param = parameter_matrix[index]
    
    #Create evenly spaced parameters to interpolate
    lambdas = np.linspace(0,1,10)
    
    for lamb in lambdas:
        
        #Update values of J according to interpolation
        new_param = np.copy(post_param)
        new_param = new_param.at[index].set((1-lamb)*pre_param[index] + lamb*post_param[index])
        
        #Take logs before passing through model
        if type_param =='J':
            opt_pars['logJ_2x2'] = np.log(new_param*signs)
        if type_param =='s':
            opt_pars['logs_2x2'] =  np.log(new_param)
        if type_param =='c':
            opt_pars['c_E'] = new_param[0]
            opt_pars['c_I'] = new_param[1]
            plot_param = int(index+1)

        
        #Evaluate accuracy
        val_loss, true_acc, ber_acc= vmap_eval(opt_pars, ssn_pars, grid_pars, conn_pars, gE, gI, test_data, filter_pars,  conv_pars, loss_pars, sig_noise=2.5)
        print('lambda ', lamb, ', accuracy', true_acc)
        
        #Store values of J and accuracy
        values.append([param for param in new_param.ravel()])
        accuracy.append(true_acc)

    to_plot = np.column_stack([np.vstack(accuracy), np.vstack(values)])
    
    #Plot parameters
    plot_acc_vs_param(to_plot, lambdas, type_param = type_param, param= plot_param)
    
    return to_plot

    
    
def response_matrix(opt_pars, ssn_pars, grid_pars, conn_pars, conv_pars, gE, gI, filter_pars, stimuli_pars, radius_list, ori_list):
    '''
    Construct a response matrix of sizze n_orientations x n_neurons x n_radii
    '''
    #Initialize ssn
    
    J_2x2, s_2x2 = exponentiate(opt_pars)
    constant_vector = constant_to_vec(opt_pars['c_E'], opt_pars['c_I'])
    sigma_oris = opt_pars['sigma_oris']
    ssn=SSN2DTopoV1_AMPAGABA_ONOFF(ssn_pars, grid_pars, conn_pars, filter_pars, J_2x2, s_2x2, gE, gI, sigma_oris)
    
    
    responses = []
    for i in range(len(ori_list)):
        
        #Find responses at different stimuli radii
        x_response = surround_suppression(ssn, stimuli_pars, conv_pars, radius_list, constant_vector, ref_ori = ori_list[i])
        print(x_response.shape)
        responses.append(x_response)
    
    
    return np.stack(responses, axis = 2)

def surround_suppression(ssn, stimuli_pars, conv_pars, radius_list, constant_vector, ref_ori, title= None):    
    all_responses = []
    
    if ref_ori==None:
        ref_ori = ssn.ori_map[4,4]
    
    print(ref_ori) #create stimuli in the function just input radii)
    for radii in radius_list:
        
        stimuli_pars['outer_radius'] = radii
        stimuli_pars['inner_radius'] = radii*5/6
        
        test_data = create_data(stimuli_pars, number = 1, offset = 2, ref_ori = ref_ori)
        stimuli = test_data['ref'][0]

        output_ref=np.matmul(ssn.gabor_filters, stimuli) + constant_vector
        SSN_input_ref=np.maximum(0, output_ref)

        #Find the fixed point 
        x_ref, _, _ = obtain_fixed_point(ssn, SSN_input_ref, conv_pars)
        
        centre_response = x_ref[2,2]

        all_responses.append(x_ref.ravel())
        print('Mean population response {} (max in population {}), centre neuron {}'.format(x_ref.mean(), x_ref.max(), centre_response))
    
    if title:
        plt.plot(radius_list, np.asarray(all_responses))
        plt.xlabel('Radius')
        plt.ylabel('Response of centre neuron')
        if title:
            plt.title(title)
        plt.show()
    
    return np.vstack(all_responses)


import numpy
from training import vmap_eval

def vmap_eval3(opt_pars, ssn_pars, grid_pars, conn_pars, gE, gI, test_data, filter_pars, conv_pars, loss_pars, sig_noise):
    '''
    Iterates through all values of 'w' to give the losses at each stimuli and weight, and the accuracy at each weight
    Output:
        losses: size(n_weights, n_stimuli )
        accuracy: size( n_weights)
    '''
    eval_vmap = vmap(vmap_eval, in_axes = ({'b_sig': None, 'logJ_2x2': None, 'logs_2x2': None, 'w_sig': 0, 'c_E':None, 'c_I':None, 'sigma_oris':None}, None, None, None, None, None, {'ref':None, 'target':None, 'label':None}, None, None, None, None) )
    losses, true_acc, ber_acc = eval_vmap(opt_pars, ssn_pars, grid_pars, conn_pars, gE, gI, test_data, filter_pars,  conv_pars, loss_pars, sig_noise)

    return losses, true_acc, ber_acc
                    
                    

def test_accuracies(opt_pars, ssn_pars, grid_pars, conn_pars, gE, gI, filter_pars, conv_pars, stimuli_pars, loss_pars, sig_noise, trials = 5, p = 0.9, printing=True):
    
    key = random.PRNGKey(7)
    N_neurons = 25
    accuracies = []
    key, _ = random.split(key)
    opt_pars['w_sig'] = random.normal(key, shape = (trials, N_neurons)) / np.sqrt(N_neurons)
    
    train_data = create_data(stimuli_pars, offset = 2)
    val_loss, true_acc, ber_acc = vmap_eval3(opt_pars, ssn_pars, grid_pars, conn_pars, gE, gI, train_data, filter_pars, conv_pars, loss_pars, sig_noise)
    
    #calcualate how many accuracies are above 90
    higher_90 = np.sum(true_acc[true_acc>p]) / len(true_acc)
    
    if printing:
        print('grating contrast = {}, jitter = {}, noise std={}, acc (% >90 ) = {}'.format(stimuli_pars['grating_contrast'], stimuli_pars['jitter_val'], stimuli_pars['std'], higher_90))
    print(true_acc.shape)
    
    return higher_90, true_acc, opt_pars['w_sig']


def initial_acc( opt_pars, ssn_pars, grid_pars, conn_pars, gE, gI, filter_pars,  conv_pars, stimuli_pars,  loss_pars,  min_sig_noise , max_sig_noise, min_jitter = 3, max_jitter = 5, p = 0.9, len_noise=11, len_jitters=3):
    '''
    Find initial accuracy for varying jitter and noise levels. 
    
    '''


    list_noise  = np.linspace(min_sig_noise, max_sig_noise, len_noise)
    list_jitters = np.linspace(min_jitter, max_jitter, len_jitters)
   
    
    low_acc=[]
    all_accuracies=[]
    percent_50=[]
    good_w_s=[]
    
    
    for sig_noise in list_noise:
        for jitter in list_jitters:
            
            #stimuli_pars['std'] = noise
            stimuli_pars['jitter_val'] = jitter
            higher_90, acc, w_s = test_accuracies(opt_pars, ssn_pars, grid_pars, conn_pars, gE, gI, filter_pars, conv_pars, stimuli_pars, loss_pars, sig_noise, p=p,  trials=100, printing=False)
            
            #save low accuracies
            if higher_90 < 0.05:
                low_acc.append([jitter, sig_noise, higher_90])
            
            indices = list(filter(lambda x: acc[x] == 0.5, range(len(acc))))
            w_s = [w_s[idx] for idx in indices]
            good_w_s.append(w_s)
            
            all_accuracies.append([jitter, sig_noise, acc])
            
    plot_histograms(all_accuracies)
        
    
    return all_accuracies, low_acc, percent_50, good_w_s


def plot_histograms(all_accuracies):
    
    n_rows =  int(np.sqrt(len(all_accuracies)))
    n_cols = int(np.ceil(len(all_accuracies) / n_rows))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 20))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    count = 0

    
   #plot histograms
    for k in range(n_rows):
        for j in range (n_cols):
            axs[k,j].hist(all_accuracies[count][2])
            axs[k,j].set_xlabel('Initial accuracy')
            axs[k,j].set_ylabel('Frequency')
            axs[k,j].set_title('std = '+str(np.round(all_accuracies[count][1], 2))+ ' jitter = '+str(np.round(all_accuracies[count][0], 2)), fontsize=10)
            count+=1
            if count==len(all_accuracies):
                break
    
    fig.show()
    

def plot_tuning_curves(pre_response_matrix, neuron_indices, radius_idx, post_response_matrix=None):


    colors = plt.cm.rainbow(np.linspace(0, 1, len(neuron_indices)))
    i=0

    for idx in neuron_indices:
        plt.plot(ori_list, pre_response_matrix[radius_idx, idx, :], '--' , color=colors[i])

        if post_response_matrix.all():
            plt.plot(ori_list, post_response_matrix[radius_idx, idx, :], color=colors[i])
        i+=1

    plt.show()