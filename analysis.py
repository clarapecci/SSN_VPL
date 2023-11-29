import os
import matplotlib.pyplot as plt
import time, os
import pandas as pd
import jax
from jax import random
import jax.numpy as np
from jax import vmap
from pdb import set_trace
import time
from torch.utils.data import DataLoader
import numpy
from util  import constant_to_vec, create_grating_pairs
from model import generate_noise
from matplotlib.colors import hsv_to_rgb
import matplotlib.patches as mpatches


def plot_training_accs(training_accs, epochs_plot = None, save=None):
    
    plt.plot(np.linspace(1, len(training_accs),len(training_accs)), training_accs)
    plt.xlabel('Epoch')
    plt.ylabel('Training accuracy')
    
    if epochs_plot==None:
                pass
    else:
        if np.isscalar(epochs_plot):
            plt.axvline(x=epochs_plot, c = 'r')
        else:
            plt.axvline(x=epochs_plot[0], c = 'r')
            plt.axvline(x=epochs_plot[1], c='r')
            #plt.axvline(x=epochs_plot[2], c='r')
    
    if save:
            plt.savefig(save+'.png')
    plt.show()
    plt.close() 
    



def plot_sigmoid_outputs(train_sig_input, val_sig_input, train_sig_output, val_sig_output, epochs_plot = None, save=None):
    
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
    #axes.vlines(x=epochs_plot)

    axes[0,0].plot(max_train_sig_input, label='Max')
    axes[0,0].plot(mean_train_sig_input, label = 'Mean')
    axes[0,0].plot(min_train_sig_input, label = 'Min')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].legend()
    axes[0,0].set_title('Input to sigmoid layer (training) ')
    #axes[0,0].vlines(x=epochs_plot)

    axes[0,1].plot(epochs_to_plot, max_val_sig_input, label='Max')
    axes[0,1].plot(epochs_to_plot, mean_val_sig_input, label = 'Mean')
    axes[0,1].plot(epochs_to_plot, min_val_sig_input, label = 'Min')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].legend()
    axes[0,1].set_title('Input to sigmoid layer (validation)')
    #axes[0,1].vlines(x=epochs_plot)

    axes[1,0].plot( max_train_sig_output, label='Max')
    axes[1,0].plot( mean_train_sig_output, label = 'Mean')
    axes[1,0].plot( min_train_sig_output, label = 'Min')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].legend()
    axes[1,0].set_title('Output of sigmoid layer (training)')
    #axes[1,0].vlines(x=epochs_plot)

    axes[1,1].plot(epochs_to_plot, max_val_sig_output, label='Max')
    axes[1,1].plot(epochs_to_plot, mean_val_sig_output, label = 'Mean')
    axes[1,1].plot(epochs_to_plot, min_val_sig_output, label = 'Min')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].legend()
    axes[1,1].set_title('Output to sigmoid layer (validation)')
    #axes[1,1].vlines(x=epochs_plot)

    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    
    if epochs_plot==None:
                pass
    else:
        if np.isscalar(epochs_plot):
            plt.axvline(x=epochs_plot, c = 'r')
        else:
            plt.axvline(x=epochs_plot[0], c = 'r')
            plt.axvline(x=epochs_plot[1], c='r')
    
    if save:
            fig.savefig(save+'.png')
    fig.show()
    plt.close()
    
    
def plot_offset(offsets, epochs_plot = None, save=None):
    
    plt.plot(offsets)
    plt.axvline(x=epochs_plot, c = 'r')
    plt.xlabel('Epoch')
    plt.ylabel('Stimuli offset')
    
    if save:
        plt.savefig(save+'.png')
    plt.show()
    
    plt.close()

        
def param_ratios_two_layer(results_file, epoch = None, percent_acc = 0.85):
    results = pd.read_csv(results_file, header = 0)

    
    
    if epoch==None:
        accuracies = list(results['val_accuracy'][:20].values)
        count = 9
        while np.asarray(accuracies).mean()<percent_acc:

            count+=1
            del accuracies[0]
            if count>len(results['val_accuracy']):
                          break
            else:
                accuracies.append(results['val_accuracy'][count])


        epoch = results['epoch'][count]
        epoch_index = results[results['epoch'] == epoch].index
        print(epoch, epoch_index)
                          
    if epoch==-1:
        epoch_index = epoch
    else:
        epoch_index = results[results['epoch'] == epoch].index
    
    if 'J_EE_m' in results.columns:
        Js = results[['J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m']]
        Js = Js.to_numpy()
        print("J_m ratios = ", np.array((Js[epoch_index,:]/Js[0,:] -1)*100))
    
    if 'J_EE_s' in results.columns:
        Js = results[['J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s']]
        Js = Js.to_numpy()
        print("J_s ratios = ", np.array((Js[epoch_index,:]/Js[0,:] -1)*100))
        
    if 's_EE_m' in results.columns:
        ss = results[['s_EE_m', 's_EI_m', 's_IE_m', 's_II_m']]
        ss = ss.to_numpy()
        print("s_m ratios = ", np.array((ss[epoch_index,:]/ss[0,:] -1)*100,))
    
    if 's_EE_s' in results.columns:
        ss = results[['s_EE_s', 's_EI_s', 's_IE_s', 's_II_s']]
        ss = ss.to_numpy()
        print("s_s ratios = ", np.array((ss[epoch_index,:]/ss[0,:] -1)*100))
    
    if 'c_E' in results.columns:
        cs = results[["c_E", "c_I"]]
        cs = cs.to_numpy()
        print("c ratios = ", np.array((cs[epoch_index,:]/cs[0,:] -1)*100))
        
    if 'sigma_orisE' in results.columns:
        sigma_oris = results[["sigma_orisE", "sigma_orisI"]]
        sigma_oris = sigma_oris.to_numpy()
        print("sigma_oris ratios = ", np.array((sigma_oris[epoch_index,:]/sigma_oris[0,:] -1)*100))
    
    if 'sigma_oris' in results.columns:
        sigma_oris = results[["sigma_oris"]]
        sigma_oris = sigma_oris.to_numpy()
        print("sigma_oris ratios = ", np.array((sigma_oris[epoch_index,:]/sigma_oris[0,:] -1)*100))
        
    if 'f_E' in results.columns:
        fs = results[["f_E", "f_I"]]
        fs = fs.to_numpy()
        print("f ratios = ", np.array((fs[epoch_index,:]/fs[0,:] -1)*100))
    
    if 'kappa_preE' in results.columns:
        kappas = results[['kappa_preE', 'kappa_preI', 'kappa_postE', 'kappa_postI']]
        kappas = kappas.to_numpy()
        print('kappas = ', kappas[epoch_index, :])
        

def plot_results_two_layers(results_filename, bernoulli=False, save=None, epochs_plot=None, norm_w=False, param_sum = False):
    
    results = pd.read_csv(results_filename, header = 0)


    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,10))

    if 'J_EE_m' in results.columns:
        colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
        param_list_m = ["J_EE_m", "J_EI_m", "J_IE_m", "J_II_m"]
        param_list_s = ["J_EE_s", "J_EI_s", "J_IE_s", "J_II_s"]
        
        for i in range(4):
            results.plot(x='epoch', y=param_list_m[i],  linestyle = '-', ax=axes[0,0], c=colors[i])
            results.plot(x='epoch', y=param_list_s[i],  linestyle = '--', ax=axes[0,0], c=colors[i])
        
        
    if 's_EE_s' in results.columns:
        results.plot(x='epoch', y=["s_EE_s", "s_EI_s", "s_IE_s", "s_II_s"], ax=axes[0,1])

    if 'c_E' in results.columns:
        results.plot(x='epoch', y=["c_E", "c_I"], ax = axes[1,0])

    if 'sigma_orisE' in results.columns:
        results.plot(x='epoch', y="sigma_orisE", linestyle = '-', ax = axes[0,1], c='tab:blue')
        results.plot(x='epoch', y="sigma_orisI", linestyle = '--', ax = axes[0,1], c='tab:blue')
    
    
    if 'sigma_orisEE' in results.columns:
        results.plot(x='epoch', y="sigma_orisEE", linestyle = '-', ax = axes[0,1], c='tab:blue')
        results.plot(x='epoch', y="sigma_orisEI", linestyle = '--', ax = axes[0,1], c='tab:blue')
    
    #option 1 training
    if 'kappa_preE' in results.columns:
        colors = ['tab:green', 'tab:orange']
        param_list_E = ["kappa_preE", "kappa_postE"]
        param_list_I = ["kappa_preI", "kappa_postI"]
        
        for i in range(2):
            results.plot(x='epoch', y=param_list_E[i],  linestyle = '-', ax=axes[2,1], c=colors[i])
            results.plot(x='epoch', y=param_list_I[i],  linestyle = '--', ax=axes[2,1], c=colors[i])
    
    #option 2 training
    if 'kappa_preEE' in results.columns:
        colors = ['tab:green', 'tab:orange']
        param_list_E = ["kappa_preEE", "kappa_postEE"]
        param_list_I = ["kappa_preEI", "kappa_postEI"]
        
        for i in range(2):
            results.plot(x='epoch', y=param_list_E[i],  linestyle = '-', ax=axes[2,1], c=colors[i])
            results.plot(x='epoch', y=param_list_I[i],  linestyle = '--', ax=axes[2,1], c=colors[i])
    
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
            if epochs_plot==None:
                pass
            else:
                if np.isscalar(epochs_plot):
                    axes[2,0].axvline(x=epochs_plot, c = 'r')
                else:
                    axes[2,0].axvline(x=epochs_plot[0], c = 'r')
                    axes[2,0].axvline(x=epochs_plot[1], c='r')
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


        


def plot_losses_two_stage(training_losses, val_loss_per_epoch, epochs_plot = None, save=None, inset = None):
    
    fig, axs1 = plt.subplots()
    axs1.plot(np.linspace(1, len(training_losses.T), len(training_losses.T)), training_losses.T, label = ['Binary cross entropy', 'Avg_dx', 'R_max', 'w', 'b', 'Training total'] )
    axs1.plot(val_loss_per_epoch[:,1], val_loss_per_epoch[:,0], label='Validation')
    axs1.legend()
    axs1.set_xlabel('Epochs')
    axs1.set_ylabel('Loss')
    axs1.set_title('Training losses')
    
    
    
    if inset:    
        left, bottom, width, height = [0.2, 0.22, 0.35, 0.25]
        ax2 = fig.add_axes([left, bottom, width, height])

        ax2.plot(training_losses[0, :], label = 'Binary loss')
        ax2.legend()

    if epochs_plot==None:
                pass
    else:
        if np.isscalar(epochs_plot):
            axs1.axvline(x=epochs_plot, c = 'r')
            if inset:
                ax2.axvline(x=epochs_plot, c = 'r') 
        else:
            axs1.axvline(x=epochs_plot[0], c = 'r')
            axs1.axvline(x=epochs_plot[1], c='r')
            #axs1.axvline(x=epochs_plot[2], c='r')
            if inset:
                ax2.axvline(x=epochs_plot, c = 'r') 
                ax2.axvline(x=epochs_plot[1], c='r') 
                #as1.axvline(x=epochs_plot[2], c='r')

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

    

def plot_all_sig(all_sig_inputs, axis_title = None, save_fig = None):
    
    n_rows =  int(np.sqrt(len(all_sig_inputs)))
    n_cols = int(np.ceil(len(all_sig_inputs) / n_rows))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 20))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    count = 0


   #plot histograms
    for k in range(n_rows):
        for j in range (n_cols):
            axs[k,j].hist(all_sig_inputs[count])
            axs[k,j].set_xlabel(axis_title)
            axs[k,j].set_ylabel('Frequency')
            count+=1
            if count==len(all_sig_inputs):
                break
    
    if save_fig:
        fig.savefig(save_fig+'_'+axis_title+'.png')
        
    fig.show()
    plt.close()
    
    
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
            axs[k,j].set_title('noise = '+str(np.round(all_accuracies[count][1], 2))+ ' w_std = '+str(np.round(all_accuracies[count][0], 2)), fontsize=10)
            count+=1
            if count==len(all_accuracies):
                break
    
    if save_fig:
        fig.savefig(save_fig+'.png')
        
    fig.show()
    plt.close()
    
    

    

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
    

    if ssn.Ne ==162:
        
        fp_E = ssn.select_type(fp, map_number = 1).ravel()
        fp_I = ssn.select_type(fp, map_number = 2).ravel()
        titles = ['E', 'I']
        all_responses = [fp_E,  fp_I]
    
    if ssn.Ne>162:
        fp_E_on = ssn.select_type(fp, map_number = 1).ravel()
        fp_E_off = ssn.select_type(fp, map_number = 3).ravel()
        fp_I_on = ssn.select_type(fp, map_number = 2).ravel()
        fp_I_off = ssn.select_type(fp, map_number = 4).ravel()
        titles = ['E_on', 'I_on', 'E_off', 'I_off']
        all_responses = [fp_E_on,  fp_I_on, fp_E_off,  fp_I_off]
    
    rows = int(len(titles)/2)
    cols =int(len(titles)/rows)
    fig, axes = plt.subplots(2,2, figsize=(8,8))
    count = 0
    for row in range(0,rows):
        for col in range(0,cols):
            ax = axes[row, col]
            im = ax.imshow(all_responses[count].reshape(9,9), vmin = fp.min(), vmax = fp.max() )
            ax.set_title(titles[count])
            ax.set_xlabel('max '+str(all_responses[count].max())+' at index '+str(np.argmax(all_responses[count])))
            count+=1
        
    fig.colorbar(im, ax=axes.ravel().tolist())
    
    if save_fig:
        fig.savefig(save_fig+'.png')
    
    plt.close()
  



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


def label_neuron(index):
    
    labels = ['E_ON', 'I_ON', 'E_OFF', 'I_OFF']
    return  labels[int(np.floor(index/81))]


def pre_post_bar_plots(neuron_indices, pre_vec, post_vec, yaxis = None, saving_dir = None):
    
    pre_to_plot = np.abs(pre_vec[neuron_indices])
    post_to_plot = np.abs(post_vec[neuron_indices])

    X = np.arange(len(neuron_indices))
    fig = plt.figure()
    ax = fig.add_axes([0.15,0.15,0.75,0.75])
    ax.bar(X + 0.00, pre_to_plot, color = 'c', width = 0.25, label='pre')
    ax.bar(X + 0.25, post_to_plot, color = 'r', width = 0.25, label = 'post')
    plt.xticks(X, neuron_indices)
    plt.xlabel('Neuron index')
    plt.legend()
    plt.axis('on')
    if yaxis:
        plt.ylabel(yaxis)
        if saving_dir:
            fig.savefig(os.path.join(saving_dir, yaxis+'.jpg'))
    fig.show()
    
    
def full_width_half_max(vector, d_theta):
    
    #Remove baseline
    vector = vector-vector.min()
    half_height = vector.max()/2
    points_above = len(vector[vector>half_height])

    distance = d_theta * points_above
    
    return distance


def sort_neurons(ei_indices, close_far_indices):
    empty_list = []
    for i in ei_indices:
        if i in close_far_indices:
            empty_list.append(i)

    return np.asarray([empty_list])


def close_far_indices(train_ori, ssn):
    
    close_indices = []
    far_indices = []
    
    upper_range = train_ori+ 90/2
    print(upper_range)
    
    for i in range(len(ssn.ori_vec)):
        if 0 < ssn.ori_vec[i] <=upper_range:
            close_indices.append(i)
        else: 
            far_indices.append(i)
            
    return np.asarray([close_indices]), np.asarray([far_indices])

def sort_close_far_EI(ssn, train_ori):
    close, far = close_far_indices(55, ssn)
    close = close.squeeze()
    far = far.squeeze()
    e_indices = np.where(ssn.tau_vec == ssn.tauE)[0]
    i_indices= np.where(ssn.tau_vec == ssn.tauI)[0]
    
    e_close = sort_neurons(e_indices, close)
    e_far = sort_neurons(e_indices, far)
    i_close = sort_neurons(i_indices, close)
    i_far = sort_neurons(i_indices, far)
    
    return e_close, e_far, i_close, i_far



def plot_close_far(E_pre, E_post, I_pre, I_post, e_close, e_far, i_close, i_far, save = None, title=None):
    
    #EE
    E_E_pre_close = [E_pre[e_close].mean(), E_pre[e_close].std()]
    E_E_post_close = [E_post[e_close].mean(), E_post[e_close].std()]
    E_E_pre_far = [E_pre[e_far].mean(), E_pre[e_far].std()]
    E_E_post_far = [E_post[e_far].mean(), E_post[e_far].std()]
    
    
    #IE
    I_E_pre_close = [E_pre[i_close].mean(), E_pre[i_close].std()]
    I_E_post_close = [E_post[i_close].mean(), E_post[i_close].std()]
    I_E_pre_far = [E_pre[i_far].mean(), E_pre[i_far].std()]
    I_E_post_far = [E_post[i_far].mean(), E_post[i_far].std()]
    
    
    #EI
    E_I_pre_close = [np.abs(I_pre[e_close].mean()), np.abs(I_pre[e_close].std())]
    E_I_post_close = [np.abs(I_post[e_close].mean()), np.abs(I_post[e_close].std())]
    E_I_pre_far = [np.abs(I_pre[e_far].mean()), np.abs(I_pre[e_far].std())]
    E_I_post_far = [np.abs(I_post[e_far].mean()), np.abs(I_post[e_far].std())]
    
    #II
    I_I_pre_close = [np.abs(I_pre[i_close].mean()), np.abs(I_pre[i_close].std())]
    I_I_post_close = [np.abs(I_post[i_close].mean()), np.abs(I_post[i_close].std())]
    I_I_pre_far = [np.abs(I_pre[i_far].mean()), np.abs(I_pre[i_far].std())]
    I_I_post_far = [np.abs(I_post[i_far].mean()), np.abs(I_post[i_far].std())]
    
    pre_close_mean = [E_E_pre_close[0], I_E_pre_close[0], E_I_pre_close[0], I_I_pre_close[0]]
    post_close_mean = [E_E_post_close[0], I_E_post_close[0], E_I_post_close[0], I_I_post_close[0]]
    
    pre_far_mean = [E_E_pre_far[0], I_E_pre_far[0], E_I_pre_far[0], I_I_pre_far[0]]
    post_far_mean = [E_E_post_far[0], I_E_post_far[0], E_I_post_far[0], I_I_post_far[0]]
                     
    pre_close_error = [E_E_pre_close[1], I_E_pre_close[1], E_I_pre_close[1], I_I_pre_close[1]]
    post_close_error = [E_E_post_close[1], I_E_post_close[1], E_I_post_close[1], I_I_post_close[1]]
    
    pre_far_error = [E_E_pre_far[1], I_E_pre_far[1], E_I_pre_far[1], I_I_pre_far[1]]
    post_far_error= [E_E_post_far[1], I_E_post_far[1], E_I_post_far[1], I_I_post_far[1]]
    
    X = np.arange(4)
    labels = ['EE', 'IE', 'EI', 'II']
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(X + 0.00, pre_close_mean, color = 'c', width = 0.15, hatch='/', label='pre_close')
    ax.bar(X + 0.15, post_close_mean, color = 'c', width = 0.15, label = 'post_close')
    ax.bar(X + 0.30, pre_far_mean, color = 'r', width = 0.15, hatch='/', label = 'pre_far')
    ax.bar(X + 0.45, post_far_mean, color = 'r', width = 0.15, label = 'post_far')
    if title:
        plt.title(title)
    plt.xticks(X + 0.225, labels)
    plt.ylabel('Average input')
    plt.legend()
    plt.axis('on')
    if save:
            plt.savefig(os.path.join(save, title+'.png'))
    fig.show()

    
def plot_pre_post_scatter(x_axis, y_axis, orientations, indices_to_plot, title, save_dir = None):
    
    '''
    Create scatter plot for pre and post training responses. Colour represents preferred orientation according to Schoups et al bins
    '''
    
    #Create legend
    patches = []
    cmap = plt.get_cmap('rainbow')
    colors = numpy.flip(cmap(numpy.linspace(0,1, 8)), axis = 0)
    bins = ['0-4', '4-12', '12-20', '20-28', '28-36', '36-44', '44-50', '+50']
    for j in range(0,len(colors)):
        patches.append(mpatches.Patch(color=colors[j], label=bins[j]))
    
    #Iterate through required neurons
    for idx in indices_to_plot:
        #Select bin and colour
        if np.abs(orientations[idx]) <4:
            colour = colors[0]
            label = bins[0]
        elif np.abs(orientations[idx]) >50:
            colour = colors[-1]
            label = bins[-1]
        else:
            colour = colors[int(1+np.floor((np.abs(orientations[idx]) -4)/8) )]
            label = bins[int(1+np.floor((np.abs(orientations[idx]) -4)/8) )]
        plt.scatter(x = x_axis[idx], y =y_axis[idx], color =colour, label = label )
    
    #Plot x = y line
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='gold')
    plt.xlabel('Pre training')
    plt.ylabel('Post training')
    plt.title(title)
    plt.legend(handles = patches, loc = 'upper right', bbox_to_anchor=(1.3, 1.0), title = 'Pref ori - train ori')
    if save_dir:
        plt.savefig(os.path.join(save_dir, str(title)+'.png'), bbox_inches='tight')
    plt.show()
    
    
def avg_slope(vector, x_axis, x1, x2, normalised=False):
    '''
    Calculates average slope between points x1 and x2. x1 and x2  given in absolute values, then converted to indices in function 
    '''
    #Remove baseline if normalising
    if normalised == True:
        vector = (vector - vector.min())/vector.max()
    
    #Find indices corresponding to desired x values
    idx_1 = (np.abs(x_axis - x1)).argmin()
    idx_2 = (np.abs(x_axis - x2)).argmin()
    
    grad =(np.abs(vector[idx_2] - vector[idx_1]))/(x2-x1)
    
    return grad



def plot_w_sig(w_sig,  epochs_to_save , epochs_plot = None,save=None):
    
    plt.plot(w_sig)
    plt.xlabel('Epoch')
    plt.ylabel('Values of w')
    if epochs_plot:
        plt.axvline(x=epochs_plot, c='r', label='criterion')
    if save:
            plt.savefig(save+'.png')
    plt.show()
    plt.close()
    
    
def plot_max_rates(max_rates, epochs_plot = None, save=None):
    
    plt.plot(np.linspace(1, len(max_rates), len(max_rates)), max_rates, label = ['E_mid', 'I_mid', 'E_sup', 'I_sup'])
    plt.xlabel('Epoch')
    plt.ylabel('Maximum rates')
    plt.legend()
    
    if epochs_plot==None:
                pass
    else:
        if np.isscalar(epochs_plot):
            
            plt.axvline(x=epochs_plot, c = 'r')
        else:
            plt.axvline(x=epochs_plot[0], c = 'r')
            plt.axvline(x=epochs_plot[0]+epochs_plot[1], c='r')
            #plt.axvline(x=epochs_plot[2], c='r')
    
    if save:
            plt.savefig(save+'.png')
    plt.show()
    plt.close() 