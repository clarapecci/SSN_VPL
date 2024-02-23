import os
import jax.numpy as np
from model import two_layer_model, constant_to_vec
from util import init_set_func, load_param_from_csv, save_matrices, create_grating_single, smooth_data
from parameters import *
from SSN_classes_middle import SSN2DTopoV1_ONOFF_local
from SSN_classes_superficial import SSN2DTopoV1
from jax import vmap
import matplotlib.pyplot as plt
import scipy


#Initalise two layer model functiion
vmap_two_layer_model =  vmap(two_layer_model, in_axes = (None, None, 0, None, None, None, None, None))

#Load stimuli parameters
stimuli_pars = StimuliPars()
stimuli_pars.jitter_val = 0


#Specify parameters not trained
init_set_m ='C'
init_set_s=1
_, s_2x2, gE_s, gI_s, conn_pars_s  = init_set_func(init_set_s, conn_pars_s, ssn_pars)
_, _, gE_m, gI_m, conn_pars_m  = init_set_func(init_set_m, conn_pars_m, ssn_pars, middle = True)
gE = [gE_m, gE_s]
gI = [gI_m, gI_s]
sigma_oris = np.asarray([90.0, 90.0])
kappa_pre = np.asarray([ 0.0, 0.0])
kappa_post = np.asarray([ 0.0, 0.0])


#Results filename where parameters are stored
#seed_n = 20
#results_dir= os.path.join(os.getcwd(), 'results/11-12/', 'stair_results',  'stair_noise200.0gE0.3_'+str(seed_n))
#results_filename = os.path.join(results_dir, 'set_C_N_readout_125_results.csv')


#List of orientations and epochs to use
n_noisy_trials = 300 #number of noisy trials to do per epoch + ori
ori_list = np.asarray([55, 125, 0])
#Calculate response for first and last epochs
epoch_list = np.asarray([1, -1])


#Collect constant parameters into single class
class constant_pars:
    ssn_pars =ssn_pars
    s_2x2 = s_2x2
    sigma_oris = sigma_oris
    grid_pars = grid_pars
    conn_pars_m = conn_pars_m
    conn_pars_s = conn_pars_s
    gE = gE
    gI = gI
    filter_pars = filter_pars
    noise_type = 'poisson'
    ssn_ori_map = ssn_ori_map_loaded
    ref_ori = stimuli_pars.ref_ori
    conv_pars = conv_pars

    
##########Saving dir ######################
home_dir= os.path.join(os.getcwd(), 'results/11-12/', 'stair_results', )
#saving_dir = os.path.join(results_dir, 'noisy_responses_train_untrain_control') 

#if os.path.exists(saving_dir) == False:
#        os.makedirs(saving_dir)
#run_dir = os.path.join(saving_dir, 'noisy_response')

#################################################

#Iterate over seed folders
for seed_n in range(1, 3):
     
     if seed_n !=3:
        if  seed_n !=12:
           
            print('seed_n', seed_n)
            #Open seed folder
            results_dir = os.path.join(home_dir, 'stair_noise200.0gE0.3_'+str(seed_n))
            results_filename = os.path.join(results_dir, 'set_C_N_readout_125_results.csv')
            saving_dir =os.path.join(results_dir, 'noisy_respose_ori_map')

            #Load seed's orientation map
            constant_pars.ssn_ori_map = np.load(os.path.join(os.getcwd(), 'results/11-12/', 'maps', 'seed'+str(seed_n), 'ori_map.npy'))
            
            #Create saving directory
            if os.path.exists(saving_dir) == False:
                os.makedirs(saving_dir)
            run_dir = os.path.join(saving_dir, 'noisy_response')
            
            #Initialise empty lists
            epoch_mid = []
            epoch_sup = []
            labels = []

            #Iterate over orientations
            for epoch in epoch_list:
                
                all_mid = []
                all_sup = []
                
                #Load params from csv for given epoch
                [J_2x2_m, J_2x2_s, c_E, c_I, f_E, f_I] = load_param_from_csv(results_filename = results_filename, epoch = epoch)
                print('c_E', c_E)
                
                #Initialise SSN layers
                ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, conn_pars=constant_pars.conn_pars_m, filter_pars=constant_pars.filter_pars, J_2x2=J_2x2_m, gE = constant_pars.gE[0], gI=constant_pars.gI[0], ori_map = constant_pars.ssn_ori_map)
                ssn_sup=SSN2DTopoV1(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, conn_pars=constant_pars.conn_pars_s, J_2x2=J_2x2_s, s_2x2=constant_pars.s_2x2, sigma_oris = constant_pars.sigma_oris, ori_map = constant_pars.ssn_ori_map, train_ori = constant_pars.ref_ori, kappa_post = kappa_post, kappa_pre = kappa_pre)

                #Generate extra synaptic constant
                constant_vector_mid = constant_to_vec(c_E, c_I, ssn=ssn_mid)
                constant_vector_sup = constant_to_vec(c_E, c_I, ssn = ssn_sup, sup=True)
                
                for ori in ori_list:

                    #Select orientation from list
                    stimuli_pars.ref_ori = ori
                    print(stimuli_pars.ref_ori)

                    #Append orientation to label 
                    labels.append(np.repeat(ori, n_noisy_trials))

                    #Generate noisy data
                    test_grating = create_grating_single(stimuli_pars = stimuli_pars, n_trials = n_noisy_trials)

                    #Calculate fixed point for data    
                    _, _, _, _, [fp_mid, fp_sup] = vmap_two_layer_model(ssn_mid, ssn_sup, test_grating, constant_pars.conv_pars, constant_vector_mid, constant_vector_sup, f_E, f_I)

                    #Smooth data with Gaussian filter
                    smooth_mid= smooth_data(fp_mid, sigma = 1)      
                    smooth_sup= smooth_data(fp_sup, sigma = 1)  
                    
                    #Sum all contributions of E and I neurons
                    smooth_mid = smooth_mid.reshape(n_noisy_trials, 9,9, -1).sum(axis = 3)
                    smooth_sup = smooth_sup.reshape(n_noisy_trials, 9,9, -1).sum(axis = 3)
                    
                    #Concatenate all orientation responses
                    all_mid.append(smooth_mid.reshape(n_noisy_trials, -1))
                    all_sup.append(smooth_sup.reshape(n_noisy_trials, -1))
                
                #Concatenate all epoch responses
                epoch_mid.append(np.vstack(np.asarray(all_mid)))
                epoch_sup.append(np.vstack(np.asarray(all_sup)))

            #Save as matlab struct
            scipy.io.savemat(os.path.join(run_dir+'.mat') , dict(middle =np.stack(np.asarray(epoch_mid)) , superficial = np.stack(np.asarray(epoch_sup)) , labels = np.asarray(labels).ravel() ) )



    
    
    
        




