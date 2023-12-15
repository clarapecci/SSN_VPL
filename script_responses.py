from importlib import reload
import two_layer_training
import os 
import jax
from two_layer_training import create_data, constant_to_vec, middle_layer_fixed_point, obtain_fixed_point_centre_E
from SSN_classes_jax_jit import SSN2DTopoV1_ONOFF_local
from SSN_classes_jax_on_only import SSN2DTopoV1
import util
from util import take_log, init_set_func
import matplotlib.pyplot as plt

import jax.numpy as np
import numpy

from SSN_classes_jax_jit import SSN2DTopoV1_ONOFF_local
from analysis import plot_vec2map, plot_mutiple_gabor_filters, obtain_min_max_indices, plot_tuning_curves


############################## PARAMETERS ############################
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
    k = numpy.array(1.0471975511965976)
    edge_deg = numpy.array(3.2)
    degree_per_pixel = numpy.array(0.05)
    
class conv_pars:
    dt = 1
    xtol = 1e-03
    Tmax = 400
    verbose = False
    silent = True
    Rmax_E = None
    Rmax_I= None

class loss_pars:
    lambda_dx = 1
    lambda_r_max = 1
    lambda_w = 1
    lambda_b = 1
    

#Specify initialisation - STARTING VALUES

init_set_m ='C'
init_set_s=1
J_2x2_s, s_2x2_s, gE_s, gI_s, conn_pars_s  = init_set_func(init_set_s, conn_pars_s, ssn_pars)
J_2x2_m, _, gE_m, gI_m, conn_pars_m  = init_set_func(init_set_m, conn_pars_m, ssn_pars, middle = True)

#Excitatory and inhibitory constants for extra synaptic GABA
c_E = 5.0
c_I = 5.0

#Feedforwards connections
f_E = 1.0
f_I = 1.0

sigma_oris = np.asarray([1000.0, 1000.0])


###############INPUT OWN VALUES##################

#make_J2x2 = lambda Jee, Jei, Jie, Jii: np.array([[Jee, -Jei], [Jie,  -Jii]])
#J_2x2_m_o = [2.49819601411255, 1.2535758449939, 3.13895880914003, 0.965496242148542]
#J_2x2_s_o = [4.09009393479124, 1.80011104021068, 4.41112119675624, 1.38315625629152]
#c_E, c_I = 4.57316649102221, 5.19979533161814
#f_E, f_I =  1.38904923387682, 0.445106642142078
#J_2x2_m = make_J2x2(*J_2x2_m_o)
#J_2x2_s = make_J2x2(*J_2x2_s_o)
#sigma_oris = np.asarray([1000.0, 1000.0])
#################################################



gE = [gE_m, gE_s]
gI = [gI_m, gI_s]

print('Js ', J_2x2_s, J_2x2_m)
print('gE, gI ', gE, gI)
print('cs and fs ', c_E, c_I, f_E, f_I)


#############################################################################

#Results dir

results_dir = os.path.join(os.getcwd(), 'results', '19-06', 'test_convg', 'new_make_dist')
if os.path.exists(results_dir) == False:
        os.makedirs(results_dir)



############################################################################

#inds = [30, 31, 32, 39, 40, 41, 48, 49, 50]
inds = [0, 81, 162, 243]

#Create stimuli
train_data = create_data(stimuli_pars, number = 1, offset = offset, ref_ori = ref_ori)
#Load orientation map
ssn_ori_map_loaded = np.load(os.path.join(os.getcwd(), 'ssn_map.npy'))

#Intialise SSNs 
ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_m, filter_pars=filter_pars, J_2x2=J_2x2_m, gE = gE_m, gI=gI_m, ori_map = ssn_ori_map_loaded)
ssn_pars.A = ssn_mid.A
ssn_sup=SSN2DTopoV1(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_s, filter_pars=filter_pars, J_2x2=J_2x2_s, s_2x2=s_2x2_s, gE = gE_s, gI=gI_s, sigma_oris = sigma_oris, ori_map = ssn_mid.ori_map)

#Vectorise extra synaptic constant
constant_vector = constant_to_vec(c_E, c_I, ssn=ssn_mid)
constant_vector_sup = constant_to_vec(c_E, c_I, ssn = ssn_sup, sup=True)

#Multiply gabor filters by stimuli
output_ref=np.matmul(ssn_mid.gabor_filters, train_data['ref'].squeeze()) 

#Rectify + add extra synapic constant
SSN_input_ref=np.maximum(0, output_ref) +  constant_vector


#Find fixed point for middle layer
r_ref_mid, r_max_ref_mid, avg_dx_ref_mid, fp, max_E_mid, max_I_mid = middle_layer_fixed_point(ssn_mid, SSN_input_ref, conv_pars, PLOT=PLOT, inds = inds,  save=mid_ref, return_fp = True)

#Input to superficial layer
sup_input_ref = np.hstack([r_ref_mid*f_E, r_ref_mid*f_I]) + constant_vector_sup

#Find fixed point for superficial layer
r_ref, r_max_ref_sup, avg_dx_ref_sup, _, max_E_sup, max_I_sup = obtain_fixed_point_centre_E(ssn_sup, sup_input_ref, conv_pars, PLOT=PLOT, inds = inds, save = sup_ref, return_fp = True)





        