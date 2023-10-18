from importlib import reload

import os 
import jax
from two_layer_training_lateral_phases import create_data, constant_to_vec, middle_layer_fixed_point, obtain_fixed_point_centre_E
from SSN_classes_phases import SSN2DTopoV1_ONOFF_local
from SSN_classes_jax_on_only import SSN2DTopoV1
import util
from util import take_log, init_set_func, load_param_from_csv
import matplotlib.pyplot as plt
from pdb import set_trace

import jax.numpy as np
import numpy
from analysis import full_width_half_max


############################## PARAMETERS ############################
#Gabor parameters 
sigma_g= 0.5
k = np.pi/(6*sigma_g)
#k = 0.5

#Stimuli parameters
ref_ori = 55
offset = 4

#Assemble parameters in dictionary
general_pars = dict(k=k , edge_deg=3.2,  degree_per_pixel=0.05)
stimuli_pars = dict(outer_radius=3, inner_radius=2.5, grating_contrast=0.99, std = 0, jitter_val = 0)
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
    phases = 4
    A2 = None
    

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
    #sigma_g = numpy.array(0.39)
    sigma_g = 0.39*0.5/1.04
    conv_factor = numpy.array(2)
    k = numpy.array(1.0471975511965976)
    #k =numpy.array(0.5)
    edge_deg = numpy.array(3.2)
    degree_per_pixel = numpy.array(0.05)
    
    
class conv_pars:
    dt = 1
    xtol = 1e-03
    Tmax = 250
    verbose = False
    silent = True
    Rmax_E = None
    Rmax_I= None

class loss_pars:
    lambda_dx = 1
    lambda_r_max = 1
    lambda_w = 1
    lambda_b = 1
    

#Specify initialisation
init_set_m ='C'
init_set_s=1
J_2x2_s, s_2x2_s, gE_s, gI_s, conn_pars_s  = init_set_func(init_set_s, conn_pars_s, ssn_pars)
J_2x2_m, _, gE_m, gI_m, conn_pars_m  = init_set_func(init_set_m, conn_pars_m, ssn_pars, middle = True)

#Excitatory and inhibitory constants for extra synaptic GABA
c_E = 5.0
c_I = 5.0

#Feedforwards connections
f_E = 1.5
f_I = 1.0
sigma_oris = np.asarray([90.0, 90.0])
kappa_pre = np.asarray([0.0, 0.0])
kappa_post = np.asarray([0.0, 0.0])


ori_list = np.linspace(10, 180, 31)
all_data = []
for ori in ori_list:
    train_data = create_data(stimuli_pars, number = 1, offset = offset, ref_ori = ori)
    all_data.append(train_data['ref'])
all_data = np.vstack([all_data])
all_data = all_data.squeeze()

ssn_ori_map_loaded = np.load(os.path.join(os.getcwd(), 'ssn_map_uniform_good.npy'))

#Intialise SSNs 

ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_m, filter_pars=filter_pars, J_2x2=J_2x2_m, gE = gE_m, gI=gI_m, ori_map = ssn_ori_map_loaded)
ssn_pars.A = ssn_mid.A
ssn_sup=SSN2DTopoV1(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_s, filter_pars=filter_pars, J_2x2=J_2x2_s, s_2x2=s_2x2_s, gE = gE_s, gI=gI_s, sigma_oris = sigma_oris, kappa_pre = kappa_pre, kappa_post = kappa_post, ori_map = ssn_mid.ori_map, train_ori = ref_ori)
constant_vector = constant_to_vec(c_E, c_I, ssn=ssn_mid)
constant_vector_sup = constant_to_vec(c_E, c_I, ssn = ssn_sup, sup=True)
constant_vector_new = np.tile(constant_vector, (31,1)).T


output_ref = np.matmul(ssn_mid.gabor_filters, all_data.T)

SSN_input_ref = np.maximum(0, output_ref) + constant_vector_new
reshape_SSN = SSN_input_ref.reshape( 8, 81, -1)

E_SSN = np.stack([np.asarray(reshape_SSN[0, :, :]), np.asarray(reshape_SSN[2, :, :]), np.asarray(reshape_SSN[4, :, :]), np.asarray(reshape_SSN[6, :, :])])
ori_list = numpy.array(ori_list)
ori_map_vec = numpy.array(ssn_mid.ori_map.ravel())

inp = np.sqrt(np.sum(E_SSN**2, axis=0) )
print(inp.shape)
for i in range(0, 81):
    #normalised_ori = ori_list - ori_map_vec[i]
    normalised_ori = np.angle(np.exp( (ori_list - ori_map_vec[i] ) * 1j * 2 * np.pi / 180) )   * 180 / (2 * np.pi)
    sorted_inp = inp[i,:][np.argsort(normalised_ori)]
    #plt.plot(normalised_ori, inp[i, :])
    plt.plot(np.sort(normalised_ori), sorted_inp)
    plt.xlabel('Normalised stimulus orientation')
    plt.ylabel('Sum of input to E neurons')
    print(full_width_half_max(inp[i, :], normalised_ori[1] - normalised_ori[0]))

#plt.show()
plt.savefig('/mnt/d/ABG_Projects_Backup/ssn_modelling/ssn-simulator/results/16-10/tuning_curves'+str(filter_pars.k)+'_'+str(filter_pars.sigma_g)+'phases_'+str(ssn_pars.phases)+'.png')
plt.close()