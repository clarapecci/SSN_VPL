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
from analysis import findRmax, plot_losses, plot_results, plot_sigmoid_outputs, param_ratios, plot_training_accs
import training
from training import train_SSN_vmap, test_accuracy
import util
from util import take_log, init_set_func
from SSN_classes_jax_jit import SSN2DTopoV1_ONOFF
#jax.config.update("jax_enable_x64", True)



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
    lambda_w = 1
    lambda_b = 1
    

init_set = 1
J_2x2, s_2x2, gE, gI, conn_pars  = util.init_set_func(init_set, conn_pars, ssn_pars)
c_E =5.0
c_I =5.0

w_sig = np.asarray([-1.6419834e-01, -1.2039653e-01,  4.9190052e-02,
                7.1532793e-02,  1.9918610e-01, -7.0686176e-02,
               -3.3405674e-01, -2.6064694e-01,  2.6405019e-01,
                3.8113732e-02, -1.1205500e-06,  2.4953848e-02,
                3.4481710e-01, -7.9764109e-03,  2.4903497e-01,
                2.1424881e-01,  1.0315515e-01, -5.3877108e-02,
                1.1453278e-01,  1.8711287e-01,  1.4289799e-01,
               -2.1493974e-01, -1.5911350e-01,  1.5544648e-01,
                2.1656343e-01])

b_sig = 0.0

sigma_oris = np.asarray([1000.0, 1000.0])
ssn=SSN2DTopoV1_ONOFF(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars, filter_pars=filter_pars, J_2x2=J_2x2, s_2x2=s_2x2, gE = gE, gI=gI, sigma_oris = sigma_oris)
ssn_pars.A = ssn.A
    
    
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
run_dir = os.path.join(results_dir, 'set_'+str(init_set)+'_sig_noise_'+str(sig_noise)+'_model_type_'+str(model_type))

#results_filename = None
results_filename = os.path.join(run_dir+'_results.csv')  #SPECIFY NAME OF RESULTS FILE

########### TRAINING LOOP ########################################

#Train network
new_opt_pars, val_loss_per_epoch, training_losses, training_accs, train_sig_inputs, train_sig_outputs, val_sig_inputs, val_sig_outputs, epoch_c, _ =\
training.train_SSN_vmap_both_lossandgrad(J_2x2, s_2x2, sigma_oris, c_E, c_I, w_sig, b_sig, ssn_pars, grid_pars, conn_pars, gE, gI, stimuli_pars, filter_pars, conv_pars, loss_pars, epochs_to_save, results_filename = results_filename, batch_size=batch_size, ref_ori = ref_ori, offset = offset, epochs=epochs, eta=eta, sig_noise = sig_noise, noise_type=noise_type, model_type=model_type, early_stop = 0.7, results_dir = results_dir)


if model_type !=4:
    epoch_c = None
    
    
#Plot losses
losses_dir = os.path.join(run_dir+'_losses')
plot_losses(training_losses, val_loss_per_epoch, epochs_to_save[:len(val_loss_per_epoch)], epoch_c = epoch_c, save = losses_dir)

#Plot results
results_plot_dir =  os.path.join(run_dir+'_results')
plot_results(results_filename, bernoulli = False, epoch_c = epoch_c, save= results_plot_dir)

#Plot sigmoid
sig_dir =  os.path.join(run_dir+'_sigmoid')
plot_sigmoid_outputs(train_sig_inputs, val_sig_inputs, train_sig_outputs, val_sig_outputs, epochs_to_save[:len(val_sig_outputs)], epoch_c = epoch_c, save=sig_dir)

#Additional plots for first step of training
if model_type ==4:
    #Plot training_accs
    training_accs_dir = os.path.join(run_dir+'_training_accs')
    plot_training_accs(training_accs, epoch_c = epoch_c, save = training_accs_dir)

histogram_dir =os.path.join(run_dir+'_histogram')


if model_type ==4:
    w_sig = new_opt_pars['w_sig']
    b_sig = new_opt_pars['b_sig']

if model_type ==5:
    
    J_2x2 = new_opt_pars['logJ_2x2']
    s_2x2 = new_opt_pars['logs_2x2']
    c_E = new_opt_pars['c_E']
    c_I = new_opt_pars['c_I']
    
if model_type ==1:
    
    J_2x2 = new_opt_pars['logJ_2x2']
    s_2x2 = new_opt_pars['logs_2x2']
    c_E = new_opt_pars['c_E']
    c_I = new_opt_pars['c_I']
    w_sig = new_opt_pars['w_sig']
    b_sig = new_opt_pars['b_sig']
    
    
test_accuracy(stimuli_pars, offset, ref_ori, J_2x2, s_2x2, c_E, c_I, w_sig, b_sig, sigma_oris, ssn_pars, grid_pars, conn_pars, gE, gI, filter_pars,  conv_pars, loss_pars, sig_noise, noise_type, save = histogram_dir)



