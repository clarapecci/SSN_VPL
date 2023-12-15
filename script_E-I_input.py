from importlib import reload
import two_layer_training
import os 
import jax
#from two_layer_training import create_data, constant_to_vec, middle_layer_fixed_point, obtain_fixed_point_centre_E
from SSN_classes_jax_jit import SSN2DTopoV1_ONOFF_local
from SSN_classes_jax_on_only import SSN2DTopoV1
import util
from util import take_log, init_set_func
import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as np
import numpy
import time

from SSN_classes_phases import SSN2DTopoV1_ONOFF_local
from two_layer_training_lateral_phases import evaluate_model_response, response_matrix, find_bins, bin_response,vmap_evaluate_response
from util import load_param_from_csv, create_stimuli
from pdb import set_trace
#PARAMETERS


#Stimuli parameters
ref_ori = 55
offset = 4

#Assemble parameters in dictionary

stimuli_pars = dict(outer_radius=3, inner_radius=2.5, grating_contrast=0.8, std = 0, jitter_val = 0, k=1, edge_deg=3.2,  degree_per_pixel=0.05)

#Network parameters
class ssn_pars():
    n = 2
    k = 0.04
    tauE = 20 # in ms
    tauI = 10 # in ms~
    psi = 0.774
    A=None
    A2 = None
    tau_s = np.array([5, 7, 100]) #in ms, AMPA, GABA, NMDA current decay time constants
    phases = 4
    
#Grid parameters
class grid_pars():
    gridsize_Nx = 9 # grid-points across each edge # gives rise to dx = 0.8 mm
    gridsize_deg = 2 * 1.6 # edge length in degrees
    magnif_factor = 2  # mm/deg
    hyper_col = 0.4 # mm   
    sigma_RF = 0.4 # deg (visual angle)

class conn_pars_m():
    PERIODIC = False
    p_local = None

    
class conn_pars_s():
    PERIODIC = False
    p_local = None

        
class filter_pars():
    sigma_g = np.array(0.27)
    conv_factor = numpy.array(2)
    k = numpy.array(1.0)
    edge_deg = numpy.array(3.2)
    degree_per_pixel = numpy.array(0.05)
    
    
class conv_pars:
    dt = 1
    xtol = 1e-03
    Tmax = 250
    verbose = False
    silent = True
    Rmax_E = None
    Rmax_I= None

class loss_pars:
    lambda_dx = 1
    lambda_r_max = 1
    lambda_w = 1
    lambda_b = 1
    


    
init_set_m ='C'
init_set_s=1
_, s_2x2_s, gE_s, gI_s, conn_pars_s  = init_set_func(init_set_s, conn_pars_s, ssn_pars)
_, _, gE_m, gI_m, conn_pars_m  = init_set_func(init_set_m, conn_pars_m, ssn_pars, middle = True)
gE = [gE_m, gE_s]
gI = [gI_m, gI_s]
sigma_oris = np.asarray([[90.0, 90.0], [90.0, 90.0]])



#epochs_to_analyse = np.asarray([0, 500, 1099]).astype(int)
epochs_to_analyse = np.asarray([0, 989]).astype(int)
oris_to_analyse = np.asarray([55])#, 125, 0])
radius_list = np.asarray([3.0])


#Results to READ filename
results_filename = '/mnt/d/ABG_Projects_Backup/ssn_modelling/ssn-simulator/results/23-10/phases_4k_1sigma_g0.27gE_0.3/set_C_sig_noise_2.0_batch50_lamw1_results.csv'
all_results = pd.read_csv(results_filename, header = 0)
ssn_ori_map_loaded = np.load(os.path.join(os.getcwd(), 'ssn_map_uniform_good.npy'))


#epochs_to_analyse = np.asarray([0, 500, 1099]).astype(int)
epochs_to_analyse = np.asarray([989]).astype(int)
oris_to_analyse = np.asarray([55])#, 125, 0])
radius_list = np.asarray([3.0])

#Results to WRITE filename
saving_dir = os.path.join(os.getcwd(), 'results', '30-10', 'analysis', 'EI_input_sigma_g0-27')
if os.path.exists(saving_dir) == False:
        os.makedirs(saving_dir)

#Specify results filename
run_dir = os.path.join(saving_dir, 'response_vec')


constant_ssn_pars = dict(ssn_pars = ssn_pars, grid_pars = grid_pars, conn_pars_m = conn_pars_m, conn_pars_s =conn_pars_s , gE =gE, gI = gI, filter_pars = filter_pars, conv_pars = conv_pars, loss_pars = loss_pars)
############## START #################




all_responses = []
all_Emid_input = []
all_Imid_input = []
all_Esup_input = []
all_Isup_input = []

for epoch in epochs_to_analyse:
    epoch_response = []
    
    [J_2x2_m, J_2x2_s, c_E, c_I, f_E, f_I, kappa_pre, kappa_post] = load_param_from_csv(results_filename, epoch = epoch)
    
    #Initialise SSN layers
    ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_m, filter_pars=filter_pars, J_2x2=J_2x2_m, gE = gE_m, gI=gI_m, ori_map = ssn_ori_map_loaded)
    ssn_sup=SSN2DTopoV1(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_s, filter_pars=filter_pars, J_2x2=J_2x2_s, s_2x2=s_2x2_s, gE = gE_s, gI=gI_s, sigma_oris = sigma_oris, kappa_pre = kappa_pre, kappa_post = kappa_post, ori_map = ssn_mid.ori_map, train_ori = 55)
    
    

    for ori in oris_to_analyse:
        
        #Create data for given ori
        general_pars = dict(k=np.pi/(6*0.5) , edge_deg=3.2,  degree_per_pixel=0.05)
        stimuli_pars = dict(outer_radius=3, inner_radius=2.5, grating_contrast=0.8, std = 0)
        stimuli_pars.update(general_pars)
        start_time = time.time()
        train_data = create_stimuli(stimuli_pars, ref_ori = ori, number = 100, jitter_val = 5)
        data_time = time.time() - start_time
        print('Data created', data_time)
        
        #Evaluate response for given stimulus
        start_time = time.time()
        response_vector, E_mid_input, E_sup_input, I_mid_input, I_sup_input = vmap_evaluate_response(ssn_mid, ssn_sup, c_E, c_I, f_E, f_I, conv_pars, train_data)
        data_time = time.time() - start_time
       
        print('epoch {} ori {}, E_mid input: {}, I_mid input {}, E_sup input {}, I_sup input{}'.format(epoch, ori, np.sum(E_mid_input), np.sum(I_mid_input), np.sum(E_sup_input), np.sum(I_sup_input)))
        
        #Concatenate responses
        epoch_response.append(response_vector)
        
        #start_time = time.time()
        #bin_response(response_vector = response_vector, bin_indices= bin_indices, epoch = epoch, ori = ori, save_dir = run_dir)
        #data_time = time.time() - start_time
        #print('responses binned', data_time)
        np.save(os.path.join(saving_dir, 'E_mid_'+str(epoch)+'_'+str(ori)+'.npy'), np.mean(E_mid_input, axis = 0))
        np.save(os.path.join(saving_dir, 'I_mid_'+str(epoch)+'_'+str(ori)+'.npy'), np.mean(I_mid_input, axis = 0))
        np.save(os.path.join(saving_dir, 'E_sup_'+str(epoch)+'_'+str(ori)+'.npy'), np.mean(E_sup_input, axis = 0))
        np.save(os.path.join(saving_dir, 'I_sup_'+str(epoch)+'_'+str(ori)+'.npy'), np.mean(I_sup_input, axis = 0))
        
        
    all_responses.append(np.vstack(np.asarray(epoch_response)))

all_responses = np.stack(np.asarray(all_responses))

    
    

