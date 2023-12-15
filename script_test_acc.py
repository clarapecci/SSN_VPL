import os
import matplotlib.pyplot as plt
import jax
import numpy
from jax import random
import jax.numpy as np
from pdb import set_trace
import numpy
from util import create_gratings
from two_layer_training_lateral_phases import take_log, test_accuracy
from SSN_classes_phases import SSN2DTopoV1_ONOFF_local
from SSN_classes_jax_on_only import SSN2DTopoV1
from util import init_set_func, load_param_from_csv, save_matrices

#Stimuli parameters
ref_ori = 55
offset = 4

#Assemble parameters in dictionary

stimuli_pars = dict(outer_radius=3, inner_radius=2.5, grating_contrast=0.8, std = 0, k=1 , edge_deg=3.2,  degree_per_pixel=0.05, jitter_val = 5)

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
    sigma_g = numpy.array(0.27)
    conv_factor = numpy.array(2)
    k = numpy.array(1)
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

noise_type = 'poisson'
w_sig = np.asarray([ 0.00221019, -0.0048709,  -0.00068029, -0.01807598, -0.03619977, -0.03086961, -0.02175927, -0.00787174,  0.04290454,  0.00211938,  0.03276497,  0.0048655,
  0.03715703,  0.00208713, -0.00563943, -0.03122774, -0.0251524,  -0.00955957,
 -0.03328597, -0.01977759, -0.02384707,  0.03579746,  0.01021243,  0.01081385,
 -0.01214695])
b_sig = -0.0050223344

#Specify initialisation
init_set_m ='C'
init_set_s=1
_, s_2x2_s, gE_s, gI_s, conn_pars_s  = init_set_func(init_set_s, conn_pars_s, ssn_pars)
_, _, gE_m, gI_m, conn_pars_m  = init_set_func(init_set_m, conn_pars_m, ssn_pars, middle = True)

gE = [gE_m, gE_s]
gI = [gI_m, gI_s]

sigma_oris = np.asarray([[90.0, 90.0], [90.0, 90.0]])

results_dir = os.path.join(os.getcwd(), 'results', '23-10', 'phases_4k_1sigma_g0.27gE_0.3')
results_filename = os.path.join(results_dir, 'set_C_sig_noise_2.0_batch50_lamw1_results.csv')
epoch = 989
[J_2x2_m, J_2x2_s, c_E, c_I, f_E, f_I, kappa_pre, kappa_post] = load_param_from_csv(results_filename = results_filename, epoch = epoch)


logJ_2x2_s = take_log(J_2x2_s)
logs_2x2 = np.log(s_2x2_s)
logJ_2x2_m = take_log(J_2x2_m)
logJ_2x2 = [logJ_2x2_m, logJ_2x2_s]
sigma_oris = np.log(sigma_oris)
sig_noise = 2.0

constant_ssn_pars = dict(ssn_pars = ssn_pars, grid_pars = grid_pars, conn_pars_m = conn_pars_m, conn_pars_s =conn_pars_s , gE =gE, gI = gI, filter_pars = filter_pars, conv_pars = conv_pars, loss_pars = loss_pars, noise_type = noise_type)
constant_ssn_pars['key'] = random.PRNGKey(numpy.random.randint(0,10000)) 
constant_ssn_pars['logs_2x2'] = logs_2x2
constant_ssn_pars['train_ori'] = ref_ori
constant_ssn_pars['sigma_oris']=sigma_oris
constant_ssn_pars['ssn_mid_ori_map'] = numpy.load(os.path.join(os.getcwd(), 'ssn_map_uniform_good.npy'))

readout_pars = dict(w_sig = w_sig, b_sig = b_sig)
ssn_layer_pars = dict(logJ_2x2 = logJ_2x2,f_E = f_E, f_I = f_I, c_E = c_E, c_I = c_I, kappa_pre = kappa_pre, kappa_post = kappa_post)

test_accuracy(ssn_layer_pars, readout_pars, constant_ssn_pars, stimuli_pars, offset, ref_ori, sig_noise, save=os.path.join(results_dir, 'acc_histogram'), number_trials = 200, batch_size = 100)