import os 
import jax


import util
from util import init_set_func, load_param_from_csv
import matplotlib.pyplot as plt
from pdb import set_trace
import jax.numpy as np
import numpy

from model import train_model
from parameters import *
import analysis
from SSN_classes_middle import SSN2DTopoV1_ONOFF_local



################### PARAMETER SPECIFICATION #################
#SSN layer parameter initialisation
init_set_m ='C'
init_set_s=1
J_2x2_s, s_2x2, gE_s, gI_s, conn_pars_s  = init_set_func(init_set_s, conn_pars_s, ssn_pars)
J_2x2_m, _, gE_m, gI_m, conn_pars_m  = init_set_func(init_set_m, conn_pars_m, ssn_pars, middle = True)

#Excitatory and inhibitory constants for extra synaptic GABA
c_E = 5.0
c_I = 5.0

#Superficial layer W parameters
sigma_oris = np.asarray([90.0, 90.0])
kappa_pre = np.asarray([ 0.0, 0.0])
kappa_post = np.asarray([ 0.0, 0.0])

#Feedforwards connections
f_E =  np.log(1.11)
f_I = np.log(0.7)

#Constants for Gabor filters
gE = [gE_m, gE_s]
gI = [gI_m, gI_s]

#Sigmoid layer parameters
N_neurons = 25
w_sig = numpy.random.normal(scale = 0.25, size = (N_neurons,)) / np.sqrt(N_neurons)
b_sig = 0.0

#Load orientation map
ssn_ori_map_loaded = np.load(os.path.join(os.getcwd(), 'orientation_maps', 'ssn_map_uniform_good.npy'))


#Collect training terms into corresponding dictionaries
readout_pars = dict(w_sig = w_sig, b_sig = b_sig)
ssn_layer_pars = dict(J_2x2_m = J_2x2_m, J_2x2_s = J_2x2_s, kappa_pre = kappa_pre, kappa_post = kappa_post, c_E = c_E, c_I = c_I, f_E = f_E, f_I = f_I )
 
#Find normalization constant of Gabor filters
ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_m, filter_pars=filter_pars, J_2x2=J_2x2_m, gE = gE[0], gI=gI[0], ori_map = ssn_ori_map_loaded)
ssn_pars.A = ssn_mid.A
ssn_pars.A2 = ssn_mid.A2

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
    
    
################### RESULTS DIRECTORY #################
#Name of results csv
home_dir = os.getcwd()

#Specify folder to save results
results_dir = os.path.join(home_dir, 'results', '20-11', 'testing_script')
if os.path.exists(results_dir) == False:
        os.makedirs(results_dir)
        
run_dir = os.path.join(results_dir,'set_'+str(init_set_m)+'_sig_noise_'+str(training_pars.sig_noise))
results_filename = os.path.join(run_dir+'_results.csv')

    
##################### TRAcINING ############

[ssn_layer_pars, readout_pars], val_loss_per_epoch, training_losses, training_accs, train_sig_inputs, train_sig_outputs, val_sig_inputs, val_sig_outputs, epochs_plot, save_w_sigs = train_model(ssn_layer_pars, readout_pars, constant_pars, conv_pars, loss_pars, training_pars, stimuli_pars, results_filename = results_filename, results_dir = run_dir, ssn_ori_map=ssn_ori_map_loaded)

print(ssn_layer_pars)
print(readout_pars)

#Save training and validation losses
#np.save(os.path.join(run_dir+'_training_losses.npy'), training_losses)
#np.save(os.path.join(run_dir+'_validation_losses.npy'), val_loss_per_epoch)

#Plot losses
losses_dir = os.path.join(run_dir+'_losses')
analysis.plot_losses_two_stage(training_losses, val_loss_per_epoch, epochs_plot = epochs_plot, save = losses_dir, inset=False)

#Plot results
results_plot_dir =  os.path.join(run_dir+'_results')
analysis.plot_results_two_layers(results_filename, bernoulli = False, epochs_plot = epochs_plot, save= results_plot_dir)

#Plot sigmoid
sig_dir = os.path.join(run_dir+'_sigmoid')
analysis.plot_sigmoid_outputs( train_sig_input= train_sig_inputs, val_sig_input =  val_sig_inputs, train_sig_output = train_sig_outputs, val_sig_output = val_sig_outputs, epochs_plot = epochs_plot, save=sig_dir)

    
#Plot training_accs
training_accs_dir = os.path.join(run_dir+'_training_accs')
analysis.plot_training_accs(training_accs, epochs_plot = epochs_plot, save = training_accs_dir)