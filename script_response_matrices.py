import os
import matplotlib.pyplot as plt
import jax

from jax import random
import jax.numpy as np
from pdb import set_trace
import numpy
from util import create_gratings
from two_layer_training_lateral import response_matrix
from SSN_classes_jax_jit import SSN2DTopoV1_ONOFF_local
from SSN_classes_jax_on_only import SSN2DTopoV1
from util import init_set_func, load_param_from_csv, save_matrices

#Grid parameters
sigma_g= 0.5
k = np.pi/(6*sigma_g)

#Stimuli parameters
ref_ori = 55
offset = 4

#Assemble parameters in dictionary
stimuli_pars = dict(k=k , edge_deg=3.2,  degree_per_pixel=0.05, outer_radius=3, inner_radius=2.5, grating_contrast=0.8, std = 0)
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
    k = 0.6875 #numpy.array(1.0471975511965976)
    edge_deg = numpy.array( 3.2)
    degree_per_pixel = numpy.array(0.05)
    
class conv_pars:
    dt = 1
    xtol = 1e-04
    Tmax = 250
    verbose = False
    silent = True
    Rmax_E = None
    Rmax_I= None

#Specify initialisation
init_set_m ='C'
init_set_s=1
_, s_2x2_s, gE_s, gI_s, conn_pars_s  = init_set_func(init_set_s, conn_pars_s, ssn_pars)
_, _, gE_m, gI_m, conn_pars_m  = init_set_func(init_set_m, conn_pars_m, ssn_pars, middle = True)

gE = [gE_m, gE_s]
gI = [gI_m, gI_s]
print('g s', gE, gI)


results_filename =  '/mnt/d/ABG_Projects_Backup/ssn_modelling/ssn-simulator/results/04-09/lateral/new_w/set_C_sig_noise_2.0_batch50_lamw1opt_1_results.csv'
epoch = 0
J_2x2_m, J_2x2_s, c_E, c_I, f_E, f_I, sigma_oris, kappa_pre, kappa_post = load_param_from_csv(results_filename = results_filename, lateral =True, epoch = epoch)

ori_list = np.linspace(-15, 125, 31)
#radius_list = np.linspace(0, 3.6, 13)
radius_list = np.asarray([3.0])



constant_ssn_pars = dict(ssn_pars = ssn_pars, grid_pars = grid_pars, conn_pars_m = conn_pars_m, conn_pars_s =conn_pars_s , gE =gE, gI = gI, filter_pars = filter_pars, conv_pars = conv_pars)
ssn_ori_map = ssn_ori_map = np.load(os.path.join(os.getcwd(), 'ssn_map.npy'))


#####################SAVE RESULTS ############################### 

#Name of results csv
home_dir = os.getcwd()

#Specify folder to save results
results_dir = os.path.join(home_dir, 'results', '04-09', 'analysis', 'k_and_c')

if os.path.exists(results_dir) == False:
        os.makedirs(results_dir)

run_dir = os.path.join(results_dir, 'response_matrix_')
##################################################################


#stimuli_pars['grating_contrast'] = 0.2
#response_matrix_contrast_sup, response_matrix_contrast_mid = response_matrix(J_2x2_m, J_2x2_s, s_2x2_s, sigma_oris, kappa_pre, kappa_post, c_E, c_I, f_E, f_I, constant_ssn_pars, stimuli_pars, radius_list, ori_list, ssn_ori_map, trained_ori = ref_ori)
#save_matrices(run_dir, stimuli_pars['grating_contrast'], response_matrix_contrast_sup, response_matrix_contrast_mid)


#stimuli_pars['grating_contrast'] = 0.4
#response_matrix_contrast_sup, response_matrix_contrast_mid = response_matrix(J_2x2_m, J_2x2_s, s_2x2_s, sigma_oris, kappa_pre, kappa_post, c_E, c_I, f_E, f_I, constant_ssn_pars, stimuli_pars, radius_list, ori_list, ssn_ori_map, trained_ori = ref_ori)
#save_matrices(run_dir, stimuli_pars['grating_contrast'], response_matrix_contrast_sup, response_matrix_contrast_mid)

#stimuli_pars['grating_contrast'] = 0.6
#response_matrix_contrast_sup, response_matrix_contrast_mid = response_matrix(J_2x2_m, J_2x2_s, s_2x2_s, sigma_oris, kappa_pre, kappa_post, c_E, c_I, f_E, f_I, constant_ssn_pars, stimuli_pars, radius_list, ori_list, ssn_ori_map, trained_ori = ref_ori)
#save_matrices(run_dir, stimuli_pars['grating_contrast'], response_matrix_contrast_sup, response_matrix_contrast_mid)

#stimuli_pars['grating_contrast'] = 0.8
#response_matrix_contrast_sup, response_matrix_contrast_mid = response_matrix(J_2x2_m, J_2x2_s, s_2x2_s, sigma_oris, kappa_pre, kappa_post, c_E, c_I, f_E, f_I, constant_ssn_pars, stimuli_pars, radius_list, #ori_list, ssn_ori_map, trained_ori = ref_ori)
#save_matrices(run_dir, stimuli_pars['grating_contrast'], response_matrix_contrast_sup, response_matrix_contrast_mid)


#stimuli_pars['grating_contrast'] = 0.99
#response_matrix_contrast_sup, response_matrix_contrast_mid = response_matrix(J_2x2_m, J_2x2_s, s_2x2_s, sigma_oris, kappa_pre, kappa_post, c_E, c_I, f_E, f_I, constant_ssn_pars, stimuli_pars, radius_list, ori_list, ssn_ori_map, trained_ori = ref_ori)
#save_matrices(run_dir, stimuli_pars['grating_contrast'], response_matrix_contrast_sup, response_matrix_contrast_mid)


k_s = np.linspace(0.6, 1.0, 5)
c_s = np.linspace(12, 12*1.5, 5)

for new_k in k_s:
    filter_pars.k = new_k
    
    for new_c in c_s:
        
        print('k {}'.format(new_k))
        filter_pars.sigma_g = (2*np.pi)/(new_c*new_k)
        _, _, SSN_inputs, ssn_mid = response_matrix(J_2x2_m, J_2x2_s, s_2x2_s, sigma_oris, kappa_pre, kappa_post, c_E, c_I, f_E, f_I, constant_ssn_pars, stimuli_pars, radius_list, ori_list, ssn_ori_map, trained_ori = ref_ori)

        np.save(os.path.join(results_dir, 'SSN_inputs'+str(new_k)+'_'+str(new_k)+'_'+str(new_c)+'.npy'), SSN_inputs)
        fig, ax = plt.subplots(1,1, figsize=(8,8))
        ax.imshow(ssn_mid.gabor_filters[40].reshape(129, 129), cmap = 'Greys')
        fig.savefig(os.path.join(results_dir, 'centre_gabor'+str(new_k)+'_'+str(new_c)+'.png'))
        plt.close()
