from importlib import reload
import two_layer_training
import os 
import jax
from two_layer_training import create_data, model
from SSN_classes_jax_jit import SSN2DTopoV1_ONOFF_local
from SSN_classes_jax_on_only import SSN2DTopoV1
import util
from util import take_log, init_set_func
import matplotlib.pyplot as plt

import jax.numpy as np
import numpy

from SSN_classes_jax_jit import SSN2DTopoV1_ONOFF_local
from analysis import initial_acc, accuracies

############################## PARAMETERS ############################
#Gabor parameters 
sigma_g= 0.5
k = np.pi/(6*sigma_g)

#Stimuli parameters
ref_ori = 55
offset = 4

#Assemble parameters in dictionary
general_pars = dict(k=k , edge_deg=3.2,  degree_per_pixel=0.05)
stimuli_pars = dict(outer_radius=3, inner_radius=2.5, grating_contrast=0.8, std = 0, jitter_val = 5)
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
    J_2x2 = None
    w_sig = None
    b_sig = None
    
class conn_pars_s():
    PERIODIC = False
    p_local = None
    J_2x2 = None
    s_2x2 = None
    sigma_oris = None
    w_sig = None
    b_sig = None
        
class filter_pars():
    sigma_g = numpy.array(0.5)
    conv_factor = numpy.array(2)
    k = numpy.array(1.0471975511965976)
    edge_deg = numpy.array( 3.2)
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
    lambda_dx = 5
    lambda_r_max = 1
    lambda_w = 1
    lambda_b = 1
    

#Specify initialisation
init_set_m ='C'
init_set_s=1
J_2x2_s, s_2x2_s, gE_s, gI_s, conn_pars_s  = init_set_func(init_set_s, conn_pars_s, ssn_pars)
J_2x2_m, _, gE_m, gI_m, conn_pars_m  = init_set_func(init_set_m, conn_pars_m, ssn_pars, middle = True)

print(J_2x2_s, J_2x2_m)
sigma_oris = np.asarray([1000.0, 1000.0])

gE = [gE_m, gE_s]
gI = [gI_m, gI_s]
print(gE, gI)
sigma_oris_s = np.asarray([1000.0, 1000.0])

#Excitatory and inhibitory constants for extra synaptic GABA
c_E = 5.0
c_I = 5.0

#Feedforwards connections
f_E = 2.0
f_I = 1.0

b_sig = 0.0

ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_m, filter_pars=filter_pars, J_2x2=J_2x2_m, gE = gE_m, gI=gI_m)
ssn_pars.A = ssn_mid.A

ssn_ori_map = ssn_mid.ori_map
noise_type = 'poisson' 

###################### SAVING DIRECTORY ########################################

results_dir = os.path.join(os.getcwd(), 'results', '22-05', 'initial_hist')
if os.path.exists(results_dir) == False:
        os.makedirs(results_dir)

save_dir = os.path.join(results_dir, 'training_checkhistogram_'+str(init_set_m)+'_noise_type_'+str(noise_type)+'_Tmax'+str(conv_pars.Tmax)+'_dx'+str(loss_pars.lambda_dx))
################################################################################


logJ_2x2_s = take_log(J_2x2_s)
logs_2x2 = np.log(s_2x2_s)
logJ_2x2_m = take_log(J_2x2_m)
logJ_2x2 = [logJ_2x2_m, logJ_2x2_s]
sigma_oris = np.log(sigma_oris_s)
sig_noise = None



readout_pars = dict(w_sig = None, b_sig = b_sig)
ssn_layer_pars = dict(logJ_2x2 = logJ_2x2, c_E = c_E, c_I = c_I, f_E = f_E, f_I = f_I, sigma_oris = sigma_oris)
constant_ssn_pars = dict(ssn_pars = ssn_pars, grid_pars = grid_pars, conn_pars_m = conn_pars_m, conn_pars_s =conn_pars_s , gE =gE, gI = gI, filter_pars = filter_pars, conv_pars = conv_pars, loss_pars = loss_pars, sig_noise = sig_noise, noise_type = noise_type, ssn_mid_ori_map = ssn_ori_map, ssn_sup_ori_map = ssn_ori_map, logs_2x2 = logs_2x2)



all_accuracies, low_acc, percent_50, good_w_s = initial_acc(ssn_layer_pars, readout_pars, constant_ssn_pars, stimuli_pars, offset = 4, min_sig_noise = 0, max_sig_noise = 2.5, min_jitter = 3, max_jitter = 5, p = 0.9, len_noise=6, save_fig = save_dir)

print(accuracies(all_accuracies))
print(good_w_s[:10])