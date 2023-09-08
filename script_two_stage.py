import os
import matplotlib.pyplot as plt
import time, os, json
import pandas as pd
from scipy import stats 
import scipy
from tqdm import tqdm
import seaborn as sns
import jax

from jax import random#
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

from jax.lib import xla_bridge
print("jax backend {}".format(xla_bridge.get_backend().platform))
from SSN_classes_jax_jit import SSN2DTopoV1_ONOFF_local
from SSN_classes_jax_on_only import SSN2DTopoV1

import util
from util import take_log, init_set_func

from analysis import findRmax, plot_losses, plot_losses_two_stage,  plot_results_two_layers, param_ratios_two_layer, plot_sigmoid_outputs, plot_training_accs
import two_layer_training
import numpy
jax.config.update("jax_enable_x64", True)

#jax.config.update("jax_debug_nans", True)

############################# NETWORK PARAMETERS #########################

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

    
class conn_pars_s():
    PERIODIC = False
    p_local = None

        
class filter_pars():
    sigma_g = numpy.array(0.5)
    conv_factor = numpy.array(2)
    k = numpy.array(1.0471975511965976)
    edge_deg = numpy.array( 3.2)
    degree_per_pixel = numpy.array(0.05)
    
class conv_pars:
    dt = 1
    xtol = 1e-04
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
    

#Specify initialisation
init_set_m ='C'
init_set_s=1
J_2x2_s, s_2x2_s, gE_s, gI_s, conn_pars_s  = init_set_func(init_set_s, conn_pars_s, ssn_pars)
J_2x2_m, _, gE_m, gI_m, conn_pars_m  = init_set_func(init_set_m, conn_pars_m, ssn_pars, middle = True)


gE = [gE_m, gE_s]
gI = [gI_m, gI_s]
print('g s', gE, gI)
sigma_oris = np.asarray([45.0, 45.0])

#Excitatory and inhibitory constants for extra synaptic GABA
c_E = 5.0
c_I = 5.0

#Feedforwards connections
#f_E = 2.0
#f_I = 1.0
param_f_E = 0.693
param_f_I = 0.0
#Sigmoid parameters
N_neurons = 25

print(J_2x2_s, J_2x2_m)

#Readout later

'''
w_sig = np.asarray([ 1.65255964e-02, -2.02851743e-02,  1.47125358e-03,  4.32006381e-02,
  2.59337798e-02, -4.07500396e-04,  2.88240220e-02,  1.46174524e-03,
 -1.32988971e-02, -3.61239421e-03,  6.57914279e-05, -1.17886653e-02,
 -6.87494315e-03,  1.67297688e-03, -2.15619290e-03, -3.67136369e-03,
 -5.64075832e-04,  3.58234299e-03, -1.59664568e-03,  1.00693129e-01,
 -8.73062983e-02, -1.87561601e-01,  1.21363625e-01,  2.78673577e-03,
  1.95321068e-03])


w_sig = np.asarray([-0.04467659,  0.0311383 ,  0.03057568,  0.00045344,
              0.01791583,  0.09311003, -0.10766725,  0.04019434,
              0.04202819, -0.03946787, -0.02907836,  0.0293383 ,
             -0.00775917, -0.01200267, -0.02961951,  0.02872184,
             -0.05024109, -0.01084556,  0.02031561,  0.02439974,
              0.00960739, -0.048269  ,  0.09633252,  0.0830293 ,
              0.05182353])
'''
w_sig= numpy.random.normal(scale = 0.25, size = (N_neurons)) / np.sqrt(N_neurons)
print(w_sig)


#w_sig= numpy.random.normal(size = (N_neurons,)) / np.sqrt(N_neurons)

b_sig =0.0


ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_m, filter_pars=filter_pars, J_2x2=J_2x2_m, gE = gE_m, gI=gE_s)
ssn_pars.A = ssn_mid.A
print('mid W ', ssn_mid.W[:2, :2])


#Load orientation map
ssn_ori_map = np.load(os.path.join(os.getcwd(), 'ssn_map.npy'))
#ssn_ori_map=None

#######################TRAINING PARAMETERS #############################

epochs = 5
num_epochs_to_save =3

epochs_to_save =  np.insert((np.unique(np.linspace(1 , epochs, num_epochs_to_save).astype(int))), 0 , 0)

noise_type = 'poisson'

eta=10e-4
sig_noise =2.0 if noise_type!= 'no_noise' else 0.0
batch_size = 50

constant_ssn_pars = dict(ssn_pars = ssn_pars, grid_pars = grid_pars, conn_pars_m = conn_pars_m, conn_pars_s =conn_pars_s , gE =gE, gI = gI, filter_pars = filter_pars, conv_pars = conv_pars, loss_pars = loss_pars, noise_type = noise_type)


#####################SAVE RESULTS ############################### 

#Name of results csv
home_dir = os.getcwd()

#Specify folder to save results
results_dir = os.path.join(home_dir, 'results', '04-09', 'test_notlateral')

if os.path.exists(results_dir) == False:
        os.makedirs(results_dir)

#Specify results filename
run_dir = os.path.join(results_dir, 'set_'+str(init_set_m)+'_sig_noise_'+str(sig_noise)+'_batch'+str(batch_size)+'_lamw'+str(loss_pars.lambda_w)+'eta_'+str(eta))

#results_filename = None
results_filename = os.path.join(run_dir+'_results.csv')

########### TRAINING LOOP ########################################

[ssn_layer_pars, readout_pars], val_loss_per_epoch, training_losses, training_accs, train_sig_inputs, train_sig_outputs, val_sig_inputs, val_sig_outputs, epoch_c, save_w_sigs= two_layer_training.new_two_stage_training(J_2x2_m, J_2x2_s, s_2x2_s, sigma_oris, c_E, c_I, param_f_E, param_f_I, w_sig, b_sig, constant_ssn_pars, stimuli_pars, epochs_to_save, results_filename = results_filename, batch_size=batch_size, ref_ori = ref_ori, offset = offset, epochs=epochs, eta=eta, sig_noise = sig_noise, noise_type=noise_type, results_dir = run_dir, extra_stop = 2, ssn_ori_map = ssn_ori_map)

print('new_pars ', ssn_layer_pars, readout_pars )

#Save training and validation losses
np.save(os.path.join(run_dir+'_training_losses.npy'), training_losses)
np.save(os.path.join(run_dir+'_validation_losses.npy'), val_loss_per_epoch)

#Plot losses
losses_dir = os.path.join(run_dir+'_losses')
plot_losses_two_stage(training_losses, val_loss_per_epoch, epoch_c = epoch_c, save = losses_dir, inset=False)

#Plot results
results_plot_dir =  os.path.join(run_dir+'_results')
plot_results_two_layers(results_filename, bernoulli = False, epoch_c = epoch_c, save= results_plot_dir)

#Plot sigmoid
sig_dir = os.path.join(run_dir+'_sigmoid')
plot_sigmoid_outputs( train_sig_input= train_sig_inputs, val_sig_input =  val_sig_inputs, train_sig_output = train_sig_outputs, val_sig_output = val_sig_outputs, epoch_c = epoch_c, save=sig_dir)

    
#Plot training_accs
training_accs_dir = os.path.join(run_dir+'_training_accs')
plot_training_accs(training_accs, epoch_c = epoch_c, save = training_accs_dir)

    
#histogram_dir =os.path.join(run_dir+'_histogram')    
#two_layer_training.test_accuracy(ssn_layer_pars, readout_pars, constant_ssn_pars, stimuli_pars, offset, ref_ori, save=histogram_dir, number_trials = 20, batch_size = 500)

#b two_layer_training.py:440, debug_flag==True
#b two_layer_training.py:349, debug_flag==True
#b two_layer_training.py:403, debug_flag==True
#util.save_h5(os.path.join(os.getcwd(), 'results', '24-04', 'dict_epoch_c'), readout_pars)

