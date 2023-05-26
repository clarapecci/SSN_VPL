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
from two_layer_training import exponentiate, constant_to_vec, create_data, obtain_fixed_point, exponentiate, take_log, middle_layer_fixed_point
from SSN_classes_jax_jit import SSN2DTopoV1_ONOFF

def find_response(opt_pars, ssn_pars, grid_pars, conn_pars, gE, gI, filter_pars, conv_pars, stimuli_pars, ref_ori, offset, inhibition=False, s_2x2 = None):
    
    if "logs_2x2" in opt_pars:
        J_2x2, s_2x2 = exponentiate(opt_pars)
    else:
        J_2x2 = exponentiate(opt_pars)
    
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


def test_accuracy(stimuli_pars, offset, ref_ori, J_2x2, s_2x2, c_E, c_I, w_sig, b_sig, sigma_oris, ssn_pars, grid_pars, conn_pars, gE, gI, filter_pars, conv_pars, loss_pars, sig_noise, noise_type, save=None, number_trials = 20, batch_size = 500, vmap_model = None):
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
    
def plot_training_accs(training_accs, epoch_c = None, save=None):
    
    plt.plot(training_accs)
    plt.xlabel('Epoch')
    plt.ylabel('Training accuracy')
    
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
    

def plot_w_sig(w_sig, epoch_c = None, save=None):
    
    plt.plot(w_sig.T)
    plt.xlabel('Epoch')
    plt.ylabel('Values of w')
    if epoch_c==None:
        pass
    else:
        if np.isscalar(epoch_c):
            plt.axvline(x=epoch_c, c = 'r')
        else:
            plt.axvline(x=epoch_c[0], c = 'r')
            plt.axvline(x=epoch_c[0]+epoch_c[1], c='r')
        
    if save:
            plt.savefig(save+'.png')
    plt.show()
    plt.close()
    

def plot_sigmoid_outputs(train_sig_input, val_sig_input, train_sig_output, val_sig_output, epoch_c = None, save=None):
    
    #Find maximum and minimum of 
    max_train_sig_input = [item.max() for item in train_sig_input]
    mean_train_sig_input = [item.mean() for item in train_sig_input]
    min_train_sig_input = [item.min() for item in train_sig_input]


    max_val_sig_input = [item[0].max() for item in val_sig_input]
    mean_val_sig_input = [item[0].mean() for item in val_sig_input]
    min_val_sig_input = [item[0].min() for item in val_sig_input]
    
    epochs_to_plot = [item[1] for item in val_sig_input]

    max_train_sig_output = [item.max() for item in train_sig_output]
    mean_train_sig_output = [item.mean() for item in train_sig_output]
    min_train_sig_output = [item.min() for item in train_sig_output]

    max_val_sig_output = [item.max() for item in val_sig_output]
    mean_val_sig_output = [item.mean() for item in val_sig_output]
    min_val_sig_output = [item.min() for item in val_sig_output]

    #Create plots 
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,8))
    #axes.vlines(x=epoch_c)

    axes[0,0].plot(max_train_sig_input, label='Max')
    axes[0,0].plot(mean_train_sig_input, label = 'Mean')
    axes[0,0].plot(min_train_sig_input, label = 'Min')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].legend()
    axes[0,0].set_title('Input to sigmoid layer (training) ')
    #axes[0,0].vlines(x=epoch_c)

    axes[0,1].plot(epochs_to_plot, max_val_sig_input, label='Max')
    axes[0,1].plot(epochs_to_plot, mean_val_sig_input, label = 'Mean')
    axes[0,1].plot(epochs_to_plot, min_val_sig_input, label = 'Min')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].legend()
    axes[0,1].set_title('Input to sigmoid layer (validation)')
    #axes[0,1].vlines(x=epoch_c)

    axes[1,0].plot( max_train_sig_output, label='Max')
    axes[1,0].plot( mean_train_sig_output, label = 'Mean')
    axes[1,0].plot( min_train_sig_output, label = 'Min')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].legend()
    axes[1,0].set_title('Output of sigmoid layer (training)')
    #axes[1,0].vlines(x=epoch_c)

    axes[1,1].plot(epochs_to_plot, max_val_sig_output, label='Max')
    axes[1,1].plot(epochs_to_plot, mean_val_sig_output, label = 'Mean')
    axes[1,1].plot(epochs_to_plot, min_val_sig_output, label = 'Min')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].legend()
    axes[1,1].set_title('Output to sigmoid layer (validation)')
    #axes[1,1].vlines(x=epoch_c)

    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    
    if epoch_c==None:
                pass
    else:
        if np.isscalar(epoch_c):
            plt.axvline(x=epoch_c, c = 'r')
        else:
            plt.axvline(x=epoch_c[0], c = 'r')
            plt.axvline(x=epoch_c[0]+epoch_c[1], c='r')
    
    if save:
            fig.savefig(save+'.png')
    fig.show()
    plt.close()


def plot_results(results_filename, bernoulli=True, epoch_c = None, save=None, norm_w = False):
    '''
    Read csv file with results and plot parameters against epochs. Option to plot norm of w if it is saved. 
    '''
    results = pd.read_csv(results_filename, header = 0)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,8))

    if 'J_EE' in results.columns:
        results.plot(x='epoch', y=["J_EE", "J_EI", "J_IE", "J_II"], ax=axes[0,0])

    if 's_EE' in results.columns:
        results.plot(x='epoch', y=["s_EE", "s_EI", "s_IE", "s_II"], ax=axes[0,1])

    if 'c_E' in results.columns:
        results.plot(x='epoch', y=["c_E", "c_I"], ax = axes[1,0])

    if 'sigma_orisE' in results.columns:
        results.plot(x='epoch', y=["sigma_oriE", "sigma_oriI"], ax = axes[0,1])

    if 'sigma_oris' in results.columns:
        results.plot(x='epoch', y=["sigma_oris"], ax = axes[0,1])

    if 'norm_w' in results.columns and norm_w ==True:
        results.plot(x='epoch', y=["norm_w"], ax = axes[1,0])

    if bernoulli == True:
            results.plot(x='epoch', y = ['val_accuracy', 'ber_accuracy'], ax = axes[1,1])
    else:
            results.plot(x='epoch', y = ['val_accuracy'], ax = axes[1,1])
            if epoch_c:
                plt.axvline(x=epoch_c, c = 'r')
    if save:
            fig.savefig(save+'.png')
    fig.show()
    plt.close()


    
    
def param_ratios(results_file):
    results = pd.read_csv(results_file, header = 0)
    
    if 'J_EE' in results.columns:
        Js = results[['J_EE', 'J_EI', 'J_IE', 'J_II']]
        Js = Js.to_numpy()
        print("J ratios = ", np.array((Js[-1,:]/Js[0,:] -1)*100, dtype=int))

    if 's_EE' in results.columns:
        ss = results[['s_EE', 's_EI', 's_IE', 's_II']]
        ss = ss.to_numpy()
        print("s ratios = ", np.array((ss[-1,:]/ss[0,:] -1)*100, dtype=int))

    if 'c_E' in results.columns:
        cs = results[["c_E", "c_I"]]
        cs = cs.to_numpy()
        print("c ratios = ", np.array((cs[-1,:]/cs[0,:] -1)*100, dtype=int))
        
    if 'sigma_orisE' in results.columns:
        sigma_oris = results[["sigma_orisE", "sigma_orisI"]]
        sigma_oris = sigma_oris.to_numpy()
        print("sigma_oris ratios = ", np.array((sigma_oris[-1,:]/sigma_oris[0,:] -1)*100, dtype=int))
    
    if 'sigma_oris' in results.columns:
        sigma_oris = results[["sigma_oris"]]
        sigma_oris = sigma_oris.to_numpy()
        
        
        print("sigma_oris ratios = ", np.array((sigma_oris[-1,:]/sigma_oris[0,:] -1)*100, dtype=int))

        
def param_ratios_two_layer(results_file):
    results = pd.read_csv(results_file, header = 0)
    
    if 'J_EE_m' in results.columns:
        Js = results[['J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m']]
        Js = Js.to_numpy()
        print("J_m ratios = ", np.array((Js[-1,:]/Js[0,:] -1)*100, dtype=int))
    
    if 'J_EE_s' in results.columns:
        Js = results[['J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s']]
        Js = Js.to_numpy()
        print("J_s ratios = ", np.array((Js[-1,:]/Js[0,:] -1)*100, dtype=int))
        
    if 's_EE_m' in results.columns:
        ss = results[['s_EE_m', 's_EI_m', 's_IE_m', 's_II_m']]
        ss = ss.to_numpy()
        print("s_m ratios = ", np.array((ss[-1,:]/ss[0,:] -1)*100, dtype=int))
    
    if 's_EE_s' in results.columns:
        ss = results[['s_EE_s', 's_EI_s', 's_IE_s', 's_II_s']]
        ss = ss.to_numpy()
        print("s_s ratios = ", np.array((ss[-1,:]/ss[0,:] -1)*100, dtype=int))
    
    if 'c_E' in results.columns:
        cs = results[["c_E", "c_I"]]
        cs = cs.to_numpy()
        print("c ratios = ", np.array((cs[-1,:]/cs[0,:] -1)*100, dtype=int))
        
    if 'sigma_orisE' in results.columns:
        sigma_oris = results[["sigma_orisE", "sigma_orisI"]]
        sigma_oris = sigma_oris.to_numpy()
        print("sigma_oris ratios = ", np.array((sigma_oris[-1,:]/sigma_oris[0,:] -1)*100, dtype=int))
    
    if 'sigma_oris' in results.columns:
        sigma_oris = results[["sigma_oris"]]
        sigma_oris = sigma_oris.to_numpy()
        print("sigma_oris ratios = ", np.array((sigma_oris[-1,:]/sigma_oris[0,:] -1)*100, dtype=int))
        
    if 'f_E' in results.columns:
        fs = results[["f_E", "f_I"]]
        fs = fs.to_numpy()
        print("f ratios = ", np.array((fs[-1,:]/fs[0,:] -1)*100, dtype=int))
        
    
        

def plot_results_two_layers(results_filename, bernoulli=False, save=None, epoch_c=None, norm_w=False, param_sum = False):
    
    results = pd.read_csv(results_filename, header = 0)


    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,10))

    if 'J_EE_m' in results.columns:
        results.plot(x='epoch', y=["J_EE_m", "J_EI_m", "J_IE_m", "J_II_m", "J_EE_s", "J_EI_s", "J_IE_s", "J_II_s"], ax=axes[0,0])

    if 's_EE_s' in results.columns:
        results.plot(x='epoch', y=["s_EE_s", "s_EI_s", "s_IE_s", "s_II_s"], ax=axes[0,1])

    if 'c_E' in results.columns:
        results.plot(x='epoch', y=["c_E", "c_I"], ax = axes[1,0])

    if 'sigma_orisE' in results.columns:
        results.plot(x='epoch', y=["sigma_orisE", "sigma_orisI"], ax = axes[0,1])
    
    if 'sigma_oris' in results.columns:
        results.plot(x='epoch', y=["sigma_oris"], ax = axes[0,1])
        
    if 'norm_w' in results.columns and norm_w == True:
        results.plot(x='epoch', y=["norm_w"], ax = axes[0,1])    
    
    if 'f_E' in results.columns:
        results.plot(x='epoch', y=["f_E", "f_I"], ax = axes[1,1])    
    

    if bernoulli == True:
            results.plot(x='epoch', y = ['val_accuracy', 'ber_accuracy'], ax = axes[2,0])
    else:
            results.plot(x='epoch', y = ['val_accuracy'], ax = axes[2,0])
            #If passed criterion, plot both lines
            if epoch_c==None:
                pass
            else:
                if np.isscalar(epoch_c):
                    axes[2,0].axvline(x=epoch_c, c = 'r')
                else:
                    axes[2,0].axvline(x=epoch_c[0], c = 'r')
                    axes[2,0].axvline(x=epoch_c[0]+epoch_c[1], c='r')
    if save:
            fig.savefig(save+'.png')
    fig.show()
    plt.close()
    
    #Create plots of sum of parameters
    if param_sum ==True:
        fig_2, axes_2 = plt.subplots(nrows=1, ncols=3, figsize=(14, 3))

        axes_2[0].plot(results['J_IE_s'].to_numpy() + results['J_EE_s'])
        axes_2[0].set_title('Sum of J_EE_s + J_IE_s')
        
        axes_2[1].plot(results['J_IE_m'].to_numpy() + results['J_EE_m'])
        axes_2[1].set_title('Sum of J_EE_m + J_IE_m')
        
        axes_2[2].plot(results['f_E'].to_numpy() + results['f_I'])
        axes_2[2].set_title('Sum of f_E + f_I')
        
        if save:
            fig_2.savefig(save+'_param_sum.png')
        
        
        
        
        fig_2.show()
        plt.close()


        
       
    
def plot_losses(training_losses, validation_losses, epochs_to_save, epoch_c = None, save=None):
    plt.plot(training_losses.T, label = ['Binary cross entropy', 'Avg_dx', 'R_max', 'w', 'b', 'Training total'] )
    plt.plot(epochs_to_save, validation_losses, label='Validation')
    plt.legend()
    plt.title('Training losses')
    if epoch_c:
        plt.axvline(x=epoch_c, c='r')
    if save:
        plt.savefig(save+'.png')
    plt.show()
    plt.close()
    
'''
def plot_losses_two_stage(training_losses, val_loss_per_epoch, epoch_c = None, save=None):
    plt.plot(training_losses.T, label = ['Binary cross entropy', 'Avg_dx', 'R_max', 'w', 'b', 'Training total'] )
    plt.plot(val_loss_per_epoch[:,1], val_loss_per_epoch[:,0], label='Validation')
    plt.legend()
    plt.title('Training losses')
    
    if epoch_c==None:
                pass
    else:
        if np.isscalar(epoch_c):
            plt.axvline(x=epoch_c, c = 'r')
        else:
            plt.axvline(x=epoch_c[0], c = 'r')
            plt.axvline(x=epoch_c[0]+epoch_c[1], c='r')
    if save:
        plt.savefig(save+'.png')
    plt.show()
    plt.close()
'''

def plot_losses_two_stage(training_losses, val_loss_per_epoch, epoch_c = None, save=None, inset = None):
    
    fig, axs1 = plt.subplots()
    axs1.plot(training_losses.T, label = ['Binary cross entropy', 'Avg_dx', 'R_max', 'w', 'b', 'Training total'] )
    axs1.plot(val_loss_per_epoch[:,1], val_loss_per_epoch[:,0], label='Validation')
    axs1.legend()
    axs1.set_title('Training losses')
    
    
    
    if inset:    
        left, bottom, width, height = [0.2, 0.22, 0.35, 0.25]
        ax2 = fig.add_axes([left, bottom, width, height])

        ax2.plot(training_losses[0, :], label = 'Binary loss')
        ax2.legend()

    if epoch_c==None:
                pass
    else:
        if np.isscalar(epoch_c):
            axs1.axvline(x=epoch_c, c = 'r')
            if inset:
                ax2.axvline(x=epoch_c, c = 'r') 
        else:
            axs1.axvline(x=epoch_c[0], c = 'r')
            axs1.axvline(x=epoch_c[0]+epoch_c[1], c='r')
            axs1.axvline(x=epoch_c[2], c='r')
            if inset:
                ax2.axvline(x=epoch_c[0], c = 'r') 
                ax2.axvline(x=epoch_c[0]+epoch_c[1], c='r') 
                axs1.axvline(x=epoch_c[2], c='r')

    fig.show()
    if save:
        fig.savefig(save+'.png')
    plt.close()

    
    
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


    
def case_1_interpolation(pre_param, post_param, ssn_pars, grid_pars, conn_pars, gE, gI, test_data, filter_pars, conv_pars, loss_pars, sig_noise, save=None):
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

        for key in pre_param.keys():
            new_param[key] =(1-lamb)*pre_param[key] + lamb*post_param[key]
        
        
        val_loss, true_acc, _= vmap_eval(new_param, ssn_pars, grid_pars, conn_pars, gE, gI, test_data, filter_pars, conv_pars, loss_pars, sig_noise=2.5)
        print('lambda ', lamb, ', accuracy', true_acc)
        accuracy.append(true_acc)
        
    plt.plot(lambdas, accuracy)
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')

    if save:
        plt.savefig(save+'.png')
    plt.show

    return accuracy, lambdas
    

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
    
    ssn=SSN2DTopoV1_AMPAGABA_ONOFF(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars, gE=gE, gI=gI, filter_pars=filter_pars, J_2x2=J_2x2, s_2x2=s_2x2, sigma_oris = opt_pars['sigma_oris'])
    
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
        
        centre_response = x_ref[int((len(x_ref) - 1)/2)]

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


from two_layer_training import create_data, model
from jax import vmap 
import numpy


def vmap_eval_hist(ssn_layer_pars, readout_pars, constant_ssn_pars, data, debug_flag=False):

    losses, all_losses, pred_label, _, _ = model(ssn_layer_pars, readout_pars, constant_ssn_pars, data, debug_flag)
    #Find accuracy based on predicted labels
    true_accuracy = np.sum(data['label'] == pred_label)/len(data['label']) 

    vmap_loss= np.mean(losses)
    all_losses = np.mean(all_losses, axis = 0)
    
    return vmap_loss, true_accuracy
                    
                    
def vmap_eval3(ssn_layer_pars, readout_pars, constant_ssn_pars, data, debug_flag=False):
    '''
    Iterates through all values of 'w' to give the losses at each stimuli and weight, and the accuracy at each weight
    Output:
        losses: size(n_weights, n_stimuli )
        accuracy: size( n_weights)
    '''
    eval_vmap = vmap(vmap_eval_hist, in_axes = ({'c_E': None, 'c_I': None, 'f_E': None, 'f_I': None, 'logJ_2x2': [None, None], 'sigma_oris': None}, {'b_sig':None, 'w_sig': 0}, {'ssn_mid_ori_map': None, 'ssn_sup_ori_map': None, 'conn_pars_m': None, 'conn_pars_s': None, 'conv_pars': None, 'filter_pars': None, 'gE': [None, None], 'gI': [None, None], 'grid_pars': None, 'loss_pars': None, 'logs_2x2': None, 'noise_type': None, 'sig_noise': None, 'ssn_pars': None}, {'label': None, 'ref': None, 'target': None}, None))
    losses, true_acc = eval_vmap(ssn_layer_pars, readout_pars, constant_ssn_pars, data, debug_flag)

    return losses, true_acc
                    
    
    
def test_accuracies(ssn_layer_pars, readout_pars, constant_ssn_pars, stimuli_pars, offset, trials = 5, p = 0.9, printing=True):
    
    
    N_neurons = 25
    accuracies = []

    readout_pars['w_sig']= numpy.random.normal(size = (trials, N_neurons)) / np.sqrt(N_neurons)

    train_data = create_data(stimuli_pars, offset = offset)
    val_loss, true_acc = vmap_eval3(ssn_layer_pars, readout_pars, constant_ssn_pars, train_data)
    
    #calcualate how many accuracies are above 90
    higher_90 = np.sum(true_acc[true_acc>p]) / len(true_acc)
    
    if printing:
        print('grating contrast = {}, jitter = {}, noise std={}, acc (% >90 ) = {}'.format(stimuli_pars['grating_contrast'], stimuli_pars['jitter_val'], stimuli_pars['std'], higher_90))
    print(true_acc.shape)
    
    return higher_90, true_acc, readout_pars['w_sig']


def initial_acc(ssn_layer_pars, readout_pars, constant_ssn_pars, stimuli_pars, offset, min_sig_noise , max_sig_noise, min_jitter = 3, max_jitter = 5, p = 0.9, len_noise=11, len_jitters=3, save_fig = None):
    '''
    Find initial accuracy for varying jitter and noise levels. 
    
    '''

    print(constant_ssn_pars['noise_type'])
    #list_noise  =  np.logspace(start=np.log10(min_sig_noise), stop=np.log10(max_sig_noise), num=len_noise, endpoint=True, base=10.0, dtype=None, axis=0)
    list_noise = np.linspace(min_sig_noise, max_sig_noise, len_noise)
    list_jitters = np.linspace(min_jitter, max_jitter, len_jitters)
   
    
    low_acc=[]
    all_accuracies=[]
    percent_50=[]
    good_w_s=[]
    
    
    for sig_noise in list_noise:
        for jitter in list_jitters:
            
            #stimuli_pars['std'] = noise
            stimuli_pars['jitter_val'] = jitter
            constant_ssn_pars['sig_noise'] = sig_noise
            
            higher_90, acc, w_s = test_accuracies(ssn_layer_pars, readout_pars, constant_ssn_pars, stimuli_pars, offset, p=p,  trials=100, printing=False)
            print(acc.shape)
            #save low accuracies
            if higher_90 < 0.05:
                low_acc.append([jitter, sig_noise, higher_90])
            
            
            indices = list(filter(lambda x: acc[x] == 0.5, range(len(acc))))
            w_s = [w_s[idx] for idx in indices]
            good_w_s.append(w_s)
            
            all_accuracies.append([jitter, sig_noise, acc])
            
    plot_histograms(all_accuracies, save_fig = save_fig)
        
    
    return all_accuracies, low_acc, percent_50, good_w_s


def plot_histograms(all_accuracies, save_fig = None):
    
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
            axs[k,j].set_title('noise = '+str(np.round(all_accuracies[count][1], 2))+ ' jitter = '+str(np.round(all_accuracies[count][0], 2)), fontsize=10)
            count+=1
            if count==len(all_accuracies):
                break
    
    if save_fig:
        fig.savefig(save_fig+'.png')
        
    fig.show()
    plt.close()
    
    
def accuracies(all_acc, p = 0.75):
    '''
    Print accuracies and jitters that give have a probability p of having initial accuracy betwen 0.45-0.55
    '''
    
    acc_to_save = []
    for x in range(len(all_acc)):
        acc = all_acc[x][2]
        if ((0.45 < acc) & (acc < 0.55)).sum() /len(acc)>p:
            print(all_acc[x][0], all_acc[x][1])
            acc_to_save.append([all_acc[x][0], all_acc[x][1]])
        
    return acc_to_save
    

def plot_tuning_curves(pre_response_matrix, neuron_indices, radius_idx, ori_list, post_response_matrix=None, save=None):


    colors = plt.cm.rainbow(np.linspace(0, 1, len(neuron_indices)))
    i=0

    for idx in neuron_indices:
        plt.plot(ori_list, pre_response_matrix[radius_idx, idx, :], '--' , color=colors[i])

        if post_response_matrix.all():
            plt.plot(ori_list, post_response_matrix[radius_idx, idx, :], color=colors[i])
        i+=1
    plt.xlabel('Orientation (degrees)')
    plt.ylabel('Response')
    
    if save:
        plt.savefig(save+'.png')
    plt.show()
    
    
def obtain_regular_indices(ssn, number = 8, test_oris=None):
    '''
    Function takes SSN network and outputs linearly separated list of orientation indices
    '''
    
    array = ssn.ori_map[2:7, 2:7]
    array=array.ravel()
    
    if test_oris:
        pass
    else:
        test_oris = np.linspace(array.min(), array.max(), number)
    indices = []
    
    for test_ori in test_oris:
        idx = (np.abs(array - test_ori)).argmin()
        indices.append(idx)

    testing_angles = [array[idx] for idx in indices]
    print(testing_angles)
    
    return indices


def plot_vec2map(ssn, fp, save_fig=False):
    
    fp_E_on = ssn.select_type(fp, select='E_ON').ravel()
    fp_E_off = ssn.select_type(fp, select='E_OFF').ravel()
    fp_I_on = ssn.select_type(fp, select='I_ON').ravel()
    fp_I_off = ssn.select_type(fp, select='I_OFF').ravel()
    
    titles = ['E_on', 'I_on', 'E_off', 'I_off']
    all_responses = [fp_E_on,  fp_I_on, fp_E_off,  fp_I_off]
    
    fig, axes = plt.subplots(2,2, figsize=(8,8))
    count = 0
    for row in range(0,2):
        for col in range(0,2):
            ax = axes[row, col]
            im = ax.imshow(all_responses[count].reshape(9,9), vmin = fp.min(), vmax = fp.max() )
            ax.set_title(titles[count])
            ax.set_xlabel('max '+str(all_responses[count].max())+' at index '+str(np.argmax(all_responses[count])))
            count+=1
        
    fig.colorbar(im, ax=axes.ravel().tolist())
    
    if save_fig:
        fig.savefig(save_fig+'.png')
    
    plt.close()
    
    


def ori_tuning_curve_responses(ssn, conv_pars, stimuli_pars, index = None, offset = 4, c_E = 5, c_I = 5):
    
    all_responses = []
    ori_list = np.linspace(0, 180, 18*2+1)
    
    #Add preferred orientation 
    if index:
        ori_list = np.unique(np.insert(ori_list, 0, ssn.ori_vec[index]).sort())
    
    #Obtain response for different orientations
    for ori in ori_list:
        stimulus_data = create_data(stimuli_pars, number = 1, offset = offset, ref_ori = ori)
        constant_vector = constant_to_vec(c_E, c_I, ssn)
    
        output_ref=np.matmul(ssn.gabor_filters, stimulus_data['ref'].squeeze()) 
       

        #Rectify output
        SSN_input_ref=np.maximum(0, output_ref) +  constant_vector
        
        r_init = np.zeros(SSN_input_ref.shape[0])
        fp, _ = obtain_fixed_point(ssn, SSN_input_ref, conv_pars)
        
        if index==None:
            all_responses.append(fp)
        else: 
            all_responses.append(fp[index])
        
    return np.vstack(all_responses), ori_list

def obtain_min_max_indices(ssn, fp):
    idx = (ssn.ori_vec>45)*(ssn.ori_vec<65)
    indices = np.where(idx)
    responses_45_65 = fp[indices]
    j_s = []
    max_min_indices = np.concatenate([np.argsort(responses_45_65)[:3], np.argsort(responses_45_65)[-3:]])
    
    for i in max_min_indices:
        j = (indices[0][i])
        j_s.append(j)
    
    return j_s
    
def plot_mutiple_gabor_filters(ssn, fp, save_fig=None, indices=None):
    
    if indices ==None:
        indices = obtain_min_max_indices(ssn = ssn, fp = fp)
        
    fig, axes = plt.subplots(2,3, figsize=(8,8))
    count=0
    for row in range(0,2):
        for col in range(0,3):
            ax = axes[row, col]
            im = plot_individual_gabor(ax, fp, ssn, index = indices[count])
            count+=1
    if save_fig:
        fig.savefig(os.path.join(save_fig+'.png'))   
    plt.show()
    plt.close()

def plot_individual_gabor(ax, fp, ssn, index):

    if ax==None:
        fig, ax = plt.subplots(1,1, figsize=(8,8))
    labels = ['E_ON', 'I_ON', 'E_OFF', 'I_OFF']
    ax.imshow(ssn.gabor_filters[index].reshape(129, 129), cmap = 'Greys')
    ax.set_xlabel('Response '+str(fp[index]))
    ax.set_title('ori '+str(ssn.ori_vec[index])+' ' +str(label_neuron(index)))
    return ax

def plot_tuning_curves(ssn, index, conv_pars, stimuli_pars, offset = 4, all_responses = None, save_fig = None):
     
        print('Neuron preferred orientation: ', str(ssn.ori_vec[index]))
       
        if all_responses!=None:
            pass
        else:
            all_responses, ori_list = ori_tuning_curve_responses(ssn = ssn, index = index, conv_pars = conv_pars, stimuli_pars = stimuli_pars, offset = offset)
        

        plt.plot(ori_list, all_responses)
        plt.axvline(x = ssn.ori_vec[index], linestyle = 'dashed', c='r', label= 'Pref ori')
        plt.xlabel('Stimulus orientations')
        plt.ylabel('Response')
        plt.title('Neuron type ' +str(label_neuron(index)))
        plt.legend()
        if save_fig:
            plt.savefig(save_fig+'.png')
        plt.show()
        plt.close()
        
        return all_responses
    
def label_neuron(index):
    
    labels = ['E_ON', 'I_ON', 'E_OFF', 'I_OFF']
    return  labels[int(np.floor(index/81))]