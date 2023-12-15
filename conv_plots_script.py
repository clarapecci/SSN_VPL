import os 
import jax
import matplotlib.pyplot as plt
from pdb import set_trace
import jax.numpy as np
import numpy


from analysis import plot_mutiple_gabor_filters, obtain_min_max_indices, plot_tuning_curves, plot_vec2map
from util import take_log, init_set_func, load_param_from_csv, create_grating_pairs, constant_to_vec
from model import middle_layer_fixed_point, obtain_fixed_point_centre_E, two_layer_model
from SSN_classes_middle import SSN2DTopoV1_ONOFF_local
from SSN_classes_superficial import SSN2DTopoV1

from parameters import *
numpy.random.seed(1)
############################## PARAMETERS ############################
#Gabor parameters 



#Stimuli parameters
trained_ori = 55

#Specify initialisation
init_set_m ='C'
init_set_s=1

J_2x2_s, s_2x2, gE_s, gI_s, conn_pars_s  = init_set_func(init_set_s, conn_pars_s, ssn_pars)
J_2x2_m, _, gE_m, gI_m, conn_pars_m  = init_set_func(init_set_m, conn_pars_m, ssn_pars, middle = True)

sigma_oris = np.asarray([90.0, 90.0])

kappa_pre = np.asarray([ 0.0, 0.0])
kappa_post = np.asarray([ 0.0, 0.0])

#Feedforwards connections
f_E = 1.25
f_I = 1.0

#Excitatory and inhibitory constants for extra synaptic GABA
c_E = 5.0
c_I = 5.0


#results_filename = '/mnt/d/ABG_Projects_Backup/ssn_modelling/ssn-simulator/results/11-12/stimuli_noise50.0gE0.15lamda0/set_C_N_readout_125_results.csv'
#epoch = 51
#[J_2x2_m, J_2x2_s, c_E, c_I, f_E, f_I] = load_param_from_csv(results_filename = results_filename, epoch = epoch)

gE = [gE_m, gE_s]
gI = [gI_m, gI_s]

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



#############################################################################

#Results dir
results_dir = os.path.join(os.getcwd(), 'results', '11-12', 'conv', 'stimuli_noise_at_end')

if os.path.exists(results_dir) == False:
        os.makedirs(results_dir)

run_dir = os.path.join(results_dir, 'set_'+str(init_set_m)+'_gEI_'+str(gE[0])+'-'+str(gI[0])+'_noise_'+str(stimuli_pars.std)+'fE_'+str(f_E))

if os.path.exists(run_dir) == False:
        os.makedirs(run_dir)

mid_ref = os.path.join(run_dir, 'mid_ref')
mid_target = os.path.join(run_dir, 'mid_target')
sup_ref = os.path.join(run_dir, 'sup_ref')
sup_target = os.path.join(run_dir,'sup_target')
responses_dir = os.path.join(run_dir, 'responses_dir')


############################################################################

inds = [30, 31, 32, 39, 40, 41, 48, 49, 50]
#inds = [0, 81, 162, 243]
#Create stimuli

train_data = create_grating_pairs(stimuli_pars = stimuli_pars, n_trials = 1)
np.save('stimuli_example'+str(stimuli_pars.std)+'.npy', train_data['ref'])
ssn_ori_map_loaded = np.load(os.path.join(os.getcwd(), 'orientation_maps', 'ssn_map_uniform_good.npy'))


#Intialise SSNs 
ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, conn_pars=constant_pars.conn_pars_m, filter_pars=constant_pars.filter_pars, J_2x2=J_2x2_m, gE = constant_pars.gE[0], gI=constant_pars.gI[0], ori_map = constant_pars.ssn_ori_map)

ssn_sup=SSN2DTopoV1(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, conn_pars=constant_pars.conn_pars_s, J_2x2=J_2x2_s, s_2x2=constant_pars.s_2x2, sigma_oris = constant_pars.sigma_oris, ori_map = constant_pars.ssn_ori_map, train_ori = constant_pars.ref_ori, kappa_post = kappa_post, kappa_pre = kappa_pre)

constant_vector_mid = constant_to_vec(c_E, c_I, ssn=ssn_mid)
constant_vector_sup = constant_to_vec(c_E, c_I, ssn = ssn_sup, sup=True)

output_ref=np.matmul(ssn_mid.gabor_filters, train_data['ref'].squeeze()) 
output_target=np.matmul(ssn_mid.gabor_filters, train_data['target'].squeeze())
set_trace()

#Rectify output
SSN_input_ref=np.maximum(0, output_ref) +  constant_vector_mid
SSN_input_target=np.maximum(0, output_target) + constant_vector_mid
r_init = np.zeros(SSN_input_ref.shape[0])


#PLOT SSN INPUT
plot_vec2map(ssn_mid, output_ref, save_fig=os.path.join(run_dir, 'output_ref'))
plot_vec2map(ssn_mid, SSN_input_ref, save_fig=os.path.join(run_dir, 'SSN_input_ref'))

PLOT=True
#Find fixed point for middle layer
r_ref_mid, r_max_ref_mid, avg_dx_ref_mid, fp, max_E_mid, max_I_mid = middle_layer_fixed_point(ssn_mid, SSN_input_ref, conv_pars, PLOT=PLOT, inds = inds,  save=mid_ref, return_fp = True, print_dt = True)
r_target_mid, r_max_target_mid, avg_dx_target_mid, _, _, _ = middle_layer_fixed_point(ssn_mid, SSN_input_target, conv_pars, PLOT=PLOT, inds = inds, save=mid_target, return_fp = True)


#Input to superficial layer
sup_input_ref = np.hstack([r_ref_mid*f_E, r_ref_mid*f_I]) + constant_vector_sup
sup_input_target = np.hstack([r_target_mid*f_E, r_target_mid*f_I]) + constant_vector_sup

#plot_vec2map(ssn_mid, fp, save_fig=os.path.join(run_dir, 'mid_response'))

#Find fixed point for superficial layer
r_ref, r_max_ref_sup, avg_dx_ref_sup, fp_sup, max_E_sup, max_I_sup = obtain_fixed_point_centre_E(ssn_sup, sup_input_ref, conv_pars, PLOT=PLOT, inds = inds, save = sup_ref, return_fp = True)

r_target, r_max_target_sup, avg_dx_target_sup= obtain_fixed_point_centre_E(ssn_sup, sup_input_target, conv_pars, PLOT=PLOT, inds = inds, save = sup_target)

print('r_max_ref_mid ', r_max_ref_mid, max_E_mid, max_I_mid)
print('r_max_target_mid ', r_max_target_mid)
print('r_max_ref_sup ', r_max_ref_sup, max_E_sup, max_I_sup)
print('r_max_target_sup ', r_max_target_sup)


#Test same results
r_sup_mode_function, _, _, _, _ = two_layer_model(ssn_mid, ssn_sup, train_data['ref'].squeeze(), conv_pars, constant_vector_mid, constant_vector_sup, f_E, f_I)
