import os
import matplotlib.pyplot as plt
import jax

from jax import random
import jax.numpy as np
from pdb import set_trace
import numpy
from util import create_gratings
from two_layer_training import response_matrix
from SSN_classes_jax_jit import SSN2DTopoV1_ONOFF_local
from SSN_classes_jax_on_only import SSN2DTopoV1
from util import init_set_func

#Grid parameters
sigma_g= 0.5
k = np.pi/(6*sigma_g)

#Stimuli parameters
ref_ori = 55
offset = 4

#Assemble parameters in dictionary
general_pars = dict(k=k , edge_deg=3.2,  degree_per_pixel=0.05)
stimuli_pars = dict(outer_radius=3, inner_radius=2.5, grating_contrast=0.8, std = 0, jitter_val = 5)
stimuli_pars.update(general_pars)
class ssn_pars():
    n = 2
    k = 0.04
    tauE = 20 # in ms
    tauI = 10 # in ms~
    psi = 0.774
    A=None
    tau_s = np.array([5, 7, 100]) #in ms, AMPA, GABA, NMDA current decay time constants
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

#Specify initialisation
init_set_m ='C'
init_set_s=1
J_2x2_s, s_2x2_s, gE_s, gI_s, conn_pars_s  = init_set_func(init_set_s, conn_pars_s, ssn_pars)
J_2x2_m, _, gE_m, gI_m, conn_pars_m  = init_set_func(init_set_m, conn_pars_m, ssn_pars, middle = True)
c_E = 5.0
c_I = 5.0
f_E = 1.0
f_I = 1.0

gE = [gE_m, gE_s]
gI = [gI_m, gI_s]
print('g s', gE, gI)
sigma_oris_s = np.asarray([1000.0, 1000.0])

ori_list = np.linspace(10, 167.5, 8)
radius_list = np.linspace(0, 3.6, 13)

constant_ssn_pars = dict(ssn_pars = ssn_pars, grid_pars = grid_pars, conn_pars_m = conn_pars_m, conn_pars_s =conn_pars_s , gE =gE, gI = gI, filter_pars = filter_pars, conv_pars = conv_pars)
ssn_ori_map = ssn_ori_map = np.load(os.path.join(os.getcwd(), 'ssn_map.npy'))


#####################SAVE RESULTS ############################### 

#Name of results csv
home_dir = os.getcwd()

#Specify folder to save results
results_dir = os.path.join(home_dir, 'results', '19-06', 'response_matrices_new')

if os.path.exists(results_dir) == False:
        os.makedirs(results_dir)

run_dir = os.path.join(results_dir, 'response_matrix_')
##################################################################


stimuli_pars['grating_contrast'] = 0.2
response_matrix_contrast_02 = response_matrix(J_2x2_m, J_2x2_s, s_2x2_s, sigma_oris_s, c_E, c_I, f_E, f_I, constant_ssn_pars, stimuli_pars, radius_list, ori_list, ssn_ori_map)
np.save(os.path.join(run_dir+str(stimuli_pars['grating_contrast'])+'.npy'), response_matrix_contrast_02) 

stimuli_pars['grating_contrast'] = 0.4
response_matrix_contrast_04 = response_matrix(J_2x2_m, J_2x2_s, s_2x2_s, sigma_oris_s, c_E, c_I, f_E, f_I, constant_ssn_pars, stimuli_pars, radius_list, ori_list, ssn_ori_map)
np.save(os.path.join(run_dir+str(stimuli_pars['grating_contrast'])+'.npy'), response_matrix_contrast_04)

stimuli_pars['grating_contrast'] = 0.6
response_matrix_contrast_06 = response_matrix(J_2x2_m, J_2x2_s, s_2x2_s, sigma_oris_s, c_E, c_I, f_E, f_I, constant_ssn_pars, stimuli_pars, radius_list, ori_list, ssn_ori_map)
np.save(os.path.join(run_dir+str(stimuli_pars['grating_contrast'])+'.npy'), response_matrix_contrast_06)

stimuli_pars['grating_contrast'] = 0.8
response_matrix_contrast_08 = response_matrix(J_2x2_m, J_2x2_s, s_2x2_s, sigma_oris_s, c_E, c_I, f_E, f_I, constant_ssn_pars, stimuli_pars, radius_list, ori_list, ssn_ori_map)
np.save(os.path.join(run_dir+str(stimuli_pars['grating_contrast'])+'.npy'), response_matrix_contrast_08)

stimuli_pars['grating_contrast'] = 0.99
response_matrix_contrast_099 = response_matrix(J_2x2_m, J_2x2_s, s_2x2_s, sigma_oris_s, c_E, c_I, f_E, f_I, constant_ssn_pars, stimuli_pars, radius_list, ori_list, ssn_ori_map)
np.save(os.path.join(run_dir+str(stimuli_pars['grating_contrast'])+'.npy'), response_matrix_contrast_099)
