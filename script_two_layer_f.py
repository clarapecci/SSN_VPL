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
jax.config.update("jax_enable_x64", True)



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
    tauE = 30 # in ms
    tauI = 10 # in ms~
    psi = 0.774
    A=None
    tau_s = np.array([5, 7, 100]) #in ms, AMPA, GABA, NMDA current decay time constants
    

#Grid parameters
class grid_pars():
    gridsize_Nx = 9 # grid-points across each edge # gives rise to dx = 0.8 mm
    gridsize_deg = 2 * 1.6 # edge length in degrees
    magnif_factor = 2  # mm/deg
    hyper_col = 0.8 # mm   
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
    xtol = 1e-05
    Tmax = 2000
    verbose = False
    silent = True
    Rmax_E = None
    Rmax_I= None

class loss_pars:
    lambda_1 = 5
    lambda_2 = 1
    lambda_w = 1.5
    lambda_b = 1
    
    
init_set =1
J_2x2_s, s_2x2_s, gE, gI, conn_pars_s  = init_set_func(init_set, conn_pars_s, ssn_pars)
J_2x2_m, _, gE_m, gI_m, conn_pars_m  = init_set_func(init_set, conn_pars_m, ssn_pars, middle = True)

sigma_oris = (np.asarray([1000.0, 1000.0]))

#Excitatory and inhibitory constants for extra synaptic GABA
c_E = 5.0
c_I = 5.0

#Feedforwards connections
f=1.0

#Sigmoid parameters
N_neurons = 25


w_sig = np.asarray([ 1.65255964e-02, -2.02851743e-02,  1.47125358e-03,  4.32006381e-02,
  2.59337798e-02, -4.07500396e-04,  2.88240220e-02,  1.46174524e-03,
 -1.32988971e-02, -3.61239421e-03,  6.57914279e-05, -1.17886653e-02,
 -6.87494315e-03,  1.67297688e-03, -2.15619290e-03, -3.67136369e-03,
 -5.64075832e-04,  3.58234299e-03, -1.59664568e-03,  1.00693129e-01,
 -8.73062983e-02, -1.87561601e-01,  1.21363625e-01,  2.78673577e-03,
  1.95321068e-03])

b_sig =0.0

ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_m, filter_pars=filter_pars, J_2x2=J_2x2_m, gE = gE, gI=gI)
ssn_pars.A = ssn_mid.A




    
#######################TRAINING PARAMETERS #############################

epochs = 1000
num_epochs_to_save =101

epochs_to_save =  np.insert((np.unique(np.linspace(1 , epochs, num_epochs_to_save).astype(int))), 0 , 0)
noise_type = 'poisson'
model_type = 4

eta=10e-5
sig_noise = 10
batch_size = 50



#####################SAVE RESULTS ###############################

#Name of results csv
home_dir = os.getcwd()

#Specify folder to save results
results_dir = os.path.join(home_dir, 'results', '12-04')
if os.path.exists(results_dir) == False:
        os.makedirs(results_dir)

#Specify results filename
run_dir = os.path.join(results_dir, 'set_'+str(init_set)+'_sig_noise_'+str(sig_noise)+'_Tmax'+str(conv_pars.Tmax)+'_model_type_'+str(model_type))

#results_filename = None
results_filename = os.path.join(run_dir+'_results.csv')

########### TRAINING LOOP ########################################

new_opt_pars, val_loss_per_epoch, training_losses, training_accs, train_sig_inputs, train_sig_outputs, val_sig_inputs, val_sig_outputs, epoch_c, save_w_sigs= two_layer_training.two_stage_training(J_2x2_m, J_2x2_s, s_2x2_s, sigma_oris, c_E, c_I, f, w_sig, b_sig, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, gE, gI, stimuli_pars, filter_pars, conv_pars, loss_pars, epochs_to_save, results_filename = results_filename, batch_size=batch_size, ref_ori = ref_ori, offset = offset, epochs=epochs, eta=eta, sig_noise = sig_noise, noise_type=noise_type, results_dir = run_dir)


#Plot losses
losses_dir = os.path.join(run_dir+'_losses')
plot_losses_two_stage(training_losses, val_loss_per_epoch, epoch_c = epoch_c, save = losses_dir)

#Plot results
results_plot_dir =  os.path.join(run_dir+'_results')
#plot_results_two_layers(results_filename, bernoulli = False, epoch_c = epoch_c, save= results_plot_dir)

#Plot sigmoid
sig_dir = os.path.join(run_dir+'_sigmoid')
#plot_sigmoid_outputs(train_sig_inputs, val_sig_inputs, train_sig_outputs, val_sig_outputs, epochs_to_save[:len(val_sig_outputs)], epoch_c = epoch_c, save=sig_dir)

    
#Plot training_accs
training_accs_dir = os.path.join(run_dir+'_training_accs')
#plot_training_accs(training_accs, epoch_c = epoch_c, save = training_accs_dir)


if model_type ==4:
    w_sig = new_opt_pars['w_sig']
    b_sig = new_opt_pars['b_sig']

if model_type ==5:
    
    J_2x2 = new_opt_pars['logJ_2x2']
    s_2x2 = new_opt_pars['logs_2x2']
    c_E = new_opt_pars['c_E']
    c_I = new_opt_pars['c_I']
    f= new_opt_pars['f']
    
if model_type ==1:
    
    J_2x2 = new_opt_pars['logJ_2x2']
    s_2x2 = new_opt_pars['logs_2x2']
    c_E = new_opt_pars['c_E']
    c_I = new_opt_pars['c_I']
    w_sig = new_opt_pars['w_sig']
    b_sig = new_opt_pars['b_sig']
    f= new_opt_pars['f']
    
histogram_dir =os.path.join(run_dir+'_histogram')    
#two_layer_training.test_accuracy(stimuli_pars, offset, ref_ori, J_2x2_m, J_2x2_s, s_2x2_s, c_E, c_I, f, w_sig, b_sig, sigma_oris, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, gE, gI, filter_pars, conv_pars, loss_pars, sig_noise, noise_type,  save = histogram_dir, number_trials = 20, batch_size = 500)
