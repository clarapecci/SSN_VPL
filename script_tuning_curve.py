import os
import jax
import jax.numpy as np
from model import response_matrix
from util import init_set_func, load_param_from_csv, save_matrices
from parameters import *

#Reference orientation during training
trained_ori = 55

#Load stimuli parameters
tuning_pars = StimuliPars()
tuning_pars.jitter_val = 0
print(tuning_pars.std)

#Specify parameters not trained
init_set_m ='C'
init_set_s=1
_, s_2x2, gE_s, gI_s, conn_pars_s  = init_set_func(init_set_s, conn_pars_s, ssn_pars)
_, _, gE_m, gI_m, conn_pars_m  = init_set_func(init_set_m, conn_pars_m, ssn_pars, middle = True)
gE = [gE_m, gE_s]
gI = [gI_m, gI_s]


#Superficial layer W parameters
sigma_oris = np.asarray([90.0, 90.0])
kappa_pre = np.asarray([ 0.0, 0.0])
kappa_post = np.asarray([ 0.0, 0.0])


#Load params from csv
#Results filename where parameters are stored
results_dir= os.path.join(os.getcwd(), 'results/11-12/noise200.0gE0.3_5')
results_filename = os.path.join(results_dir, 'set_C_N_readout_125_results.csv')

#Select epoch to load parameters from
epoch = 685
[J_2x2_m, J_2x2_s, c_E, c_I, f_E, f_I] = load_param_from_csv(results_filename = results_filename, epoch = epoch)


#List of orientations and stimuli  radii
ori_list = np.linspace(-35, 145, 61).astype(int)
radius_list = np.asarray([3.0])
ssn_ori_map_loaded = np.load(os.path.join(os.getcwd(), 'orientation_maps', 'ssn_map_uniform_good.npy'))

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


#####################SAVE RESULTS ############################### 

#Name of results csv
home_dir = os.getcwd()

#Specify folder to save results
saving_dir =os.path.join(results_dir, 'response_matrices')

if os.path.exists(saving_dir) == False:
        os.makedirs(saving_dir)

run_dir = os.path.join(saving_dir, 'response_epoch'+str(epoch))
##################################################################

response_matrix_contrast_sup, response_matrix_contrast_mid = response_matrix(J_2x2_m, J_2x2_s, kappa_pre, kappa_post, c_E, c_I, f_E, f_I, constant_pars, tuning_pars, radius_list, ori_list, trained_ori = trained_ori)
save_matrices(run_dir, matrix_sup = response_matrix_contrast_sup, matrix_mid = response_matrix_contrast_mid)
