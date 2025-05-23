import os 
import jax
import util
from util import init_set_func, load_param_from_csv, constant_to_vec, select_neurons, create_grating_single
import matplotlib.pyplot as plt
from pdb import set_trace
import jax.numpy as np
import numpy

from training import train_model
from model_Ro import two_layer_model
#from training_staircase import train_model_staircase
#from training_homeostatic import train_model_homeo
from parameters import *
import analysis
from SSN_classes_middle import SSN2DTopoV1_ONOFF_local
from SSN_classes_superficial import SSN2DTopoV1
numpy.random.seed(0)


conv_pars.Rmax_E_sup = 0
conv_pars.Rmax_E_mid = 0
conv_pars.Rmax_I_sup = 0
conv_pars.Rmax_I_mid = 0
conv_pars.Rmean_E_sup = 0
conv_pars.Rmean_E_mid = 0
conv_pars.Rmean_I_sup = 0
conv_pars.Rmean_I_mid = 0

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
f_E = 1.25
f_I = 1.0

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
ssn_layer_pars = dict(J_2x2_m = J_2x2_m, J_2x2_s = J_2x2_s, c_E = c_E, c_I = c_I, f_E = f_E, f_I = f_I)
 
#Find normalization constant of Gabor filters
ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_m, filter_pars=filter_pars, J_2x2=J_2x2_m, gE = gE[0], gI=gI[0], ori_map = ssn_ori_map_loaded)
ssn_pars.A = ssn_mid.A
ssn_pars.A2 = ssn_mid.A2



#Find rates before training
ssn_sup=SSN2DTopoV1(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_s, J_2x2=J_2x2_s, s_2x2=s_2x2, sigma_oris = sigma_oris, ori_map = ssn_ori_map_loaded, train_ori = stimuli_pars.ref_ori, kappa_post = kappa_post, kappa_pre = kappa_pre)

stimuli = create_grating_single(stimuli_pars = stimuli_pars, n_trials = 1)
constant_vector_mid = constant_to_vec(c_E, c_I, ssn=ssn_mid)
constant_vector_sup = constant_to_vec(c_E, c_I, ssn = ssn_sup, sup=True)

_, _, _, [max_E_mid, max_I_mid, max_E_sup, max_I_sup], [fp_mid, fp_sup] = two_layer_model(ssn_mid, ssn_sup, stimuli.squeeze(), conv_pars, constant_vector_mid, constant_vector_sup, f_E, f_I)

E_mid_responses, I_mid_responses = util.select_neurons(fp_mid, layer='mid')
E_sup_responses, I_sup_responses = util.select_neurons(fp_sup, layer='sup')


conv_pars.Rmax_E_sup = max_E_sup
conv_pars.Rmax_E_mid = max_E_mid
conv_pars.Rmax_I_sup = max_I_sup
conv_pars.Rmax_I_mid = max_I_mid
conv_pars.Rmean_E_sup = np.mean(E_sup_responses)
conv_pars.Rmean_E_mid = np.mean(E_mid_responses)
conv_pars.Rmean_I_sup = np.mean(I_sup_responses)
conv_pars.Rmean_I_mid = np.mean(I_mid_responses)


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
    conv_pars = conv_pars
    loss_pars= loss_pars
    kappa_pre = kappa_pre 
    kappa_post = kappa_post

################### RESULTS DIRECTORY #################
#Name of results csv
home_dir = os.getcwd()

#Specify folder to save results
results_dir = os.path.join(home_dir, 'results', '11-12', 'new_loss_noise_end_stimuli_noise'+str(stimuli_pars.std)+'gE'+str(gE_m)+'lamda'+str(loss_pars.lambda_r_max), 'lambda_mean'+str(conv_pars.lambda_rmean)+'lambda_max'+str(conv_pars.lambda_rmean))
if os.path.exists(results_dir) == False:
        os.makedirs(results_dir)
        
run_dir = os.path.join(results_dir,'set_'+str(init_set_m)+'_N_readout_'+str(training_pars.N_readout))
results_filename = os.path.join(run_dir+'_results.csv')
    
##################### TRAINING ############


[ssn_layer_pars, readout_pars], val_loss_per_epoch, training_losses, training_accs, train_sig_inputs, train_sig_outputs, val_sig_inputs, val_sig_outputs, epochs_plot, save_w_sigs = train_model(ssn_layer_pars, readout_pars, constant_pars, training_pars, stimuli_pars, results_filename = results_filename, results_dir = run_dir)


#Homeostatic training
#single_stimuli_pars = StimuliPars()
#[ssn_layer_pars, readout_pars], val_loss_per_epoch, training_losses, training_accs, train_sig_inputs, train_sig_outputs, val_sig_inputs, val_sig_outputs, epochs_plot, save_w_sigs = train_model_homeo(ssn_layer_pars, readout_pars, constant_pars, training_pars, stimuli_pars, single_stimuli_pars, results_filename = results_filename, results_dir = run_dir)

#Staircase training
#performance_pars = StimuliPars()
#[ssn_layer_pars, readout_pars], val_loss_per_epoch, training_losses, training_accs, train_sig_inputs, train_sig_outputs, val_sig_inputs, val_sig_outputs, epochs_plot, save_w_sigs, saved_offsets = train_model_staircase(ssn_layer_pars, readout_pars, constant_pars, training_pars, performance_pars, results_filename = results_filename, results_dir = run_dir)

#Plot offsets
#threshold_dir = os.path.join(run_dir+'_threshold')
#analysis.plot_offset(saved_offsets, epochs_plot = epochs_plot, save = threshold_dir)

print(ssn_layer_pars)
print(readout_pars)


#Plot losses
losses_dir = os.path.join(run_dir+'_losses')
analysis.plot_losses_two_stage(training_losses, val_loss_per_epoch, epochs_plot = epochs_plot, save = losses_dir, inset=False)

#Plot results
results_plot_dir =  os.path.join(run_dir+'_results')
analysis.plot_results_two_layers(results_filename = results_filename, epochs_plot = epochs_plot, save= results_plot_dir)

#Plot sigmoid
sig_dir = os.path.join(run_dir+'_sigmoid')
analysis.plot_sigmoid_outputs( train_sig_input= train_sig_inputs, val_sig_input =  val_sig_inputs, train_sig_output = train_sig_outputs, val_sig_output = val_sig_outputs, epochs_plot = epochs_plot, save=sig_dir)

    
#Plot training_accs
training_accs_dir = os.path.join(run_dir+'_training_accs')
analysis.plot_training_accs(training_accs, epochs_plot = epochs_plot, save = training_accs_dir)

