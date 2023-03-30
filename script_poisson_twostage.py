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
from analysis import findRmax, plot_losses, plot_results, plot_sigmoid_outputs, param_ratios, test_accuracy, plot_training_accs
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

w_sig = np.asarray([ 4.6691943e-02,  3.4450240e-02,  2.6463202e-01, -3.6834985e-01,
  6.1189532e-02,  3.1668279e-02, -1.9055042e-02,  5.1375467e-01,
  2.4316998e-02,  2.9258179e-03,  1.5156704e-01,  4.2141914e-01,
 -1.0102301e-01,  1.2537725e-01, -3.4159210e-01, -1.4907039e-06,
  1.0735187e-01,  1.4633578e-01, -1.0555874e-01, -1.7405464e-01,
 -3.6909419e-01,  3.0254589e-02,  1.2516534e-01,  1.6338615e-02,
  3.1285040e-02])

b_sig = 0.015322882

sigma_oris = np.asarray([1000.0, 1000.0])
ssn=SSN2DTopoV1_ONOFF(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars, filter_pars=filter_pars, J_2x2=J_2x2, s_2x2=s_2x2, gE = gE, gI=gI, sigma_oris = sigma_oris)
ssn_pars.A = ssn.A
    
    
#######################TRAINING PARAMETERS #############################

epochs = 750
num_epochs_to_save =51

epochs_to_save =  np.insert((np.unique(np.linspace(1 , epochs, num_epochs_to_save).astype(int))), 0 , 0)
noise_type = 'poisson'
model_type = 5

eta=10e-5
sig_noise = 10
batch_size = 50


#####################SAVE RESULTS ###############################

#Name of results csv
home_dir = os.getcwd()

#Create directory for results
results_dir = os.path.join(home_dir, 'results', '27-03', 'script_results')
if os.path.exists(results_dir) == False:
        os.makedirs(results_dir)

results_name = 'set_'+str(init_set)+'_model_type_'+str(model_type)+'_noise_type_'+str(noise_type)+'.csv' #SPECIFY NAME OF RESULTS FILE
if results_name == None:
    results_name = 'results.csv'

results_filename = os.path.join(results_dir, results_name)


########### TRAINING LOOP ########################################

#Train network
new_opt_pars, val_loss_per_epoch, training_losses, training_accs, train_sig_inputs, train_sig_outputs, val_sig_inputs, val_sig_outputs, epoch_c = training.train_SSN_vmap_both_lossandgrad(J_2x2, s_2x2, sigma_oris, c_E, c_I, w_sig, b_sig, ssn_pars, grid_pars, conn_pars, gE, gI, stimuli_pars, filter_pars, conv_pars, loss_pars, epochs_to_save, results_filename = results_filename, batch_size=batch_size, ref_ori = ref_ori, offset = offset, epochs=epochs, eta=eta, sig_noise = sig_noise, noise_type=noise_type, model_type=model_type, early_stop = 0.7)

#Plot training_accs
training_accs_dir = os.path.join(results_dir,'set_'+str(init_set)+'_model_type_'+str(model_type)+'_noise_type_'+str(noise_type)+'_training_accs_loss_grad.png') 
plot_training_accs(training_accs, epoch_c = epoch_c, save = training_accs_dir)

#Test accuracy with obtained w
histogram_dir = os.path.join(results_dir,'set_'+str(init_set)+'_model_type_'+str(model_type)+'_noise_type_'+str(noise_type)+'_accuracy_histogram_loss_grad')

trained_w_sig = new_opt_pars['w_sig']
trained_b_sig = new_opt_pars['b_sig']

test_accuracy(stimuli_pars, offset, ref_ori, J_2x2, s_2x2, c_E, c_I, trained_w_sig, trained_b_sig, sigma_oris, ssn_pars, grid_pars, conn_pars, gE, gI, filter_pars,  conv_pars, loss_pars, sig_noise, noise_type, save = histogram_dir, number_trials =20, batch_size = 500)

#Plot losses
losses_dir = os.path.join(results_dir,'set_'+str(init_set)+'_model_type_'+str(model_type)+'_noise_type_'+str(noise_type)+'_losses_loss_grad')
plot_losses(training_losses, val_loss_per_epoch, epochs_to_save[:len(val_loss_per_epoch)], epoch_c = epoch_c, save = losses_dir)

#Plot results
results_plot_dir = os.path.join(results_dir,'set_'+str(init_set)+'_model_type_'+str(model_type)+'_noise_type_'+str(noise_type)+'results_loss_grad')
plot_results(results_filename, bernoulli = False, epoch_c = epoch_c, save= results_plot_dir)

#Plot sigmoid
sig_dir = os.path.join(results_dir,'set_'+str(init_set)+'_model_type_'+str(model_type)+'_noise_type_'+str(noise_type)+'_sigmoid_loss_grad')
plot_sigmoid_outputs(train_sig_inputs, val_sig_inputs, train_sig_outputs, val_sig_outputs, epochs_to_save[:len(val_sig_outputs)], epoch_c = epoch_c, save=sig_dir)




