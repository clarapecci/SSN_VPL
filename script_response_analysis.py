from importlib import reload
import two_layer_training
import os 
import jax
from two_layer_training import create_data, constant_to_vec, middle_layer_fixed_point, obtain_fixed_point_centre_E
from SSN_classes_jax_jit import SSN2DTopoV1_ONOFF_local
from SSN_classes_jax_on_only import SSN2DTopoV1
import util
from util import take_log, init_set_func
import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as np
import numpy
import time

from SSN_classes_jax_jit import SSN2DTopoV1_ONOFF_local
from two_layer_training import vmap_evalute_response, response_matrix, find_bins, bin_response
from util import load_param_from_csv, create_stimuli
from pdb import set_trace
#PARAMETERS


#Gabor parameters 
sigma_g= 0.5
k = np.pi/(6*sigma_g)

#Stimuli parameters
ref_ori = 55
offset = 4

#Assemble parameters in dictionary
general_pars = dict(k=k , edge_deg=3.2,  degree_per_pixel=0.05)
stimuli_pars = dict(outer_radius=3, inner_radius=2.5, grating_contrast=0.8, std = 0)#, jitter_val = 5)
stimuli_pars.update(general_pars)

#Network parameters
class ssn_pars():
    n = 2
    k = 0.04
    tauE = 20 # in ms
    tauI = 10 # in ms~
    psi = 0.774
    A=None
    tau_s = np.array([5, 7, 100]) #in ms, AMPA, GABA, NMDA current decay time constants
    

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
    sigma_g = numpy.array(0.5)
    conv_factor = numpy.array(2)
    k = numpy.array(1.0471975511965976)
    edge_deg = numpy.array(3.2)
    degree_per_pixel = numpy.array(0.05)
    
class conv_pars:
    dt = 1
    xtol = 1e-03
    Tmax = 400
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


#epochs_to_analyse = np.linspace(1001, 2000, 11).astype(int)
epochs_to_analyse = np.linspace(215, 1215, 11).astype(int)
oris_to_analyse = np.asarray([55, 125, 0])



#Results to READ filename
results_filename = '/mnt/d/ABG_Projects_Backup/ssn_modelling/ssn-simulator/results/19-06/training/cos_reparam_f/set_C_sig_noise_1.0_batch50_lamw1eta_0.001_results.csv'
#results_filename = '/home/cp661/code/ABL/results/set_C_sig_noise_1.5_batch50_lamw1eta_0.01_results.csv'
all_results = pd.read_csv(results_filename, header = 0)
ssn_ori_map_loaded = np.load(os.path.join(os.getcwd(), 'ssn_map.npy'))

#Results to WRITE filename

saving_dir = os.path.join(os.getcwd(), 'results', '26-06', 'analysis_1000')

if os.path.exists(saving_dir) == False:
        os.makedirs(saving_dir)

#Specify results filename
run_dir = os.path.join(saving_dir, 'response_vec')



constant_ssn_pars = dict(ssn_pars = ssn_pars, grid_pars = grid_pars, conn_pars_m = conn_pars_m, conn_pars_s =conn_pars_s , gE =gE, gI = gI, filter_pars = filter_pars, conv_pars = conv_pars, loss_pars = loss_pars)
############## START #################

#Find tuning curves
ori_list = np.linspace(10, 167.5, 8)
radius_list = np.asarray([3])

#Get parameters from csv file

J_2x2_m, J_2x2_s, c_E, c_I, f_E, f_I, sigma_oris = load_param_from_csv(all_results, epoch = 0)

epoch_0_response_matrix_0 = response_matrix(J_2x2_m, J_2x2_s, s_2x2_s, sigma_oris, c_E, c_I, f_E, f_I, constant_ssn_pars, stimuli_pars, radius_list, ori_list, ssn_ori_map_loaded)

bin_indices = find_bins(epoch_0_response_matrix_0, ori_list)

general_pars = dict(k=k , edge_deg=3.2,  degree_per_pixel=0.05)
stimuli_pars = dict(outer_radius=3, inner_radius=2.5, grating_contrast=0.8, std = 0)#, jitter_val = 5)
stimuli_pars.update(general_pars)
start_time = time.time()
train_data = create_stimuli(stimuli_pars, ref_ori = 10, number = 10, jitter_val = 5)
data_time = time.time() - start_time
print('Data created outside loop', data_time)


for epoch in epochs_to_analyse:
    
    J_2x2_m, J_2x2_s, c_E, c_I, f_E, f_I, sigma_oris = load_param_from_csv(all_results, epoch = epoch)
    
    #Initialise SSN layers
    ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_m, filter_pars=filter_pars, J_2x2=J_2x2_m, gE = gE_m, gI=gI_m, ori_map = ssn_ori_map_loaded)
    ssn_sup=SSN2DTopoV1(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_s, filter_pars=filter_pars, J_2x2=J_2x2_s, s_2x2=s_2x2_s, gE = gE_s, gI=gI_s, sigma_oris = sigma_oris, ori_map = ssn_mid.ori_map)

    for ori in oris_to_analyse:
        print(epoch, ori)
        general_pars = dict(k=k , edge_deg=3.2,  degree_per_pixel=0.05)
        stimuli_pars = dict(outer_radius=3, inner_radius=2.5, grating_contrast=0.8, std = 0)#, jitter_val = 5)
        stimuli_pars.update(general_pars)
        start_time = time.time()
        train_data = create_stimuli(stimuli_pars, ref_ori = ori, number = 1000, jitter_val = 5)
        data_time = time.time() - start_time
        print('Data created', data_time)
        
        start_time = time.time()
        response_vector = vmap_evalute_response(ssn_mid, ssn_sup, c_E, c_I, f_E, f_I, conv_pars, train_data)
        data_time = time.time() - start_time
        print('vector created', data_time)
        
        start_time = time.time()
        bin_response(response_vector = response_vector, bin_indices= bin_indices, epoch = epoch, ori = ori, save_dir = run_dir)
        data_time = time.time() - start_time
        print('responses binned', data_time)
        

    
    
    

