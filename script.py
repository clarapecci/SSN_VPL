import os
import matplotlib.pyplot as plt
import time, os, json
import jax
from jax import random
from jax.config import config 
import jax.numpy as np
from jax import vmap
import pdb
import csv
import time

import numpy
from importlib import reload
from jax.lib import xla_bridge
print("jax backend {}".format(xla_bridge.get_backend().platform))

#Import functions from other modules
from analysis import findRmax, plot_losses, plot_results, plot_sigmoid_outputs, param_ratios
import training
from training import train_SSN_vmap
import util
from util import take_log, init_set_func
from SSN_classes_jax_jit import SSN2DTopoV1_ONOFF



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

class conn_pars():
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
    Tmax = 1000
    verbose = False
    silent = True
    Rmax_E = None
    Rmax_I= None

class loss_pars:
    lambda_1 = 1
    lambda_2 = 1
    lambda_w = 1.5
    lambda_b = 1
    

init_set = 1
J_2x2, s_2x2, gE, gI, conn_pars  = util.init_set_func(init_set, conn_pars, ssn_pars)
c_E =5.0
c_I =5.0
w_sig = np.asarray([-1.02713061e-02, -6.87258071e-05,  2.43387539e-02, -8.20847750e-02,
  4.62155826e-02,  1.72400951e-01,  5.59922159e-02, -8.82579833e-02,
  2.25766958e-03, -4.16562445e-02,  5.43892905e-02, -2.95555107e-02,
  1.09360315e-01,  2.50574071e-02,  2.82999277e-01,  5.35705686e-02,
 -1.92006286e-02,  1.53160781e-01, -3.99518805e-03, -6.24436513e-02,
 -1.11802489e-01, -7.97314197e-02,  8.48980471e-02,  5.15499227e-02,
  1.36474982e-01])

b_sig = -0.01777091

sigma_oris = np.asarray([1000.0, 1000.0]) #np.asarray([45.0, 45.0])
ssn=SSN2DTopoV1_ONOFF(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars, filter_pars=filter_pars, J_2x2=J_2x2, s_2x2=s_2x2, gE = gE, gI=gI, sigma_oris = sigma_oris)
ssn_pars.A = ssn.A
    
    
#######################TRAINING PARAMETERS #############################

epochs = 500
num_epochs_to_save =51

epochs_to_save =  np.insert((np.unique(np.linspace(1 , epochs, num_epochs_to_save).astype(int))), 0 , 0)
noise_type = 'multiplicative'
model_type = 4

eta=10e-4
sig_noise = 2.5
batch_size = 50


#####################SAVE RESULTS ###############################

#Name of results csv
home_dir = os.getcwd()

#Create directory for results
results_dir = os.path.join(home_dir, 'results', '27-03', 'script_results')
if os.path.exists(results_dir) == False:
        os.makedirs(results_dir)

results_name = 'testing_script.csv'

#results_name = 'set_'+str(init_set)+'_model_type_'+str(model_type)+'_noise_type_'+str(noise_type)+'_10.csv' #SPECIFY NAME OF RESULTS FILE
if results_name == None:
    results_name = 'results.csv'

results_filename = os.path.join(results_dir, results_name)


########### TRAINING LOOP ########################################

#Train network
new_opt_pars, val_loss_per_epoch, training_losses, training_accs, train_sig_inputs, train_sig_outputs, val_sig_inputs, val_sig_outputs = training.train_SSN_vmap(J_2x2, s_2x2, sigma_oris, c_E, c_I, w_sig, b_sig, ssn_pars, grid_pars, conn_pars, gE, gI, stimuli_pars, filter_pars, conv_pars, loss_pars, epochs_to_save, results_filename = results_filename, batch_size=batch_size, ref_ori = ref_ori, offset = offset, epochs=epochs, eta=eta, sig_noise = sig_noise, noise_type=noise_type, model_type=model_type, early_stop = 0.7)


#Plot losses
losses_dir = os.path.join(results_dir,'set_'+str(init_set)+'_model_type_'+str(model_type)+'_noise_type_'+str(noise_type)+'_losses')
plot_losses(training_losses, val_loss_per_epoch, epochs_to_save[:len(val_loss_per_epoch)], save = losses_dir)

#Plot results
results_plot_dir = os.path.join(results_dir,'set_'+str(init_set)+'_model_type_'+str(model_type)+'_noise_type_'+str(noise_type)+'results')
plot_results(results_filename, bernoulli = False, save= results_plot_dir)

#Plot sigmoid
sig_dir = os.path.join(results_dir,'set_'+str(init_set)+'_model_type_'+str(model_type)+'_noise_type_'+str(noise_type)+'_sigmoid')
plot_sigmoid_outputs(train_sig_inputs, val_sig_inputs, train_sig_outputs, val_sig_outputs, epochs_to_save[:len(val_sig_outputs)], save=sig_dir)




