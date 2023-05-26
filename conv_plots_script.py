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
    tauE = 60 # in ms
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
    J_2x2 = None
    w_sig = None
    b_sig = None
    
class conn_pars_s():
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
    

#Specify initialisation
init_set_m ='C'
init_set_s=1
J_2x2_s, s_2x2_s, gE_s, gI_s, conn_pars_s  = init_set_func(init_set_s, conn_pars_s, ssn_pars)
J_2x2_m, _, gE_m, gI_m, conn_pars_m  = init_set_func(init_set_m, conn_pars_m, ssn_pars, middle = True)

print(J_2x2_s, J_2x2_m)
sigma_oris = np.asarray([1000.0, 1000.0])
gE_m, gI_m = 0.4, 0.4
gE = [gE_m, gE_s]
gI = [gI_m, gI_s]

print(gE, gI)

#Excitatory and inhibitory constants for extra synaptic GABA
c_E = 5.0
c_I = 5.0

#Feedforwards connections
f_E = 1.0
f_I = 1.0


#############################################################################

#Results dir

results_dir = os.path.join(os.getcwd(), 'results', '22-05', 'convergence', 'set_'+str(init_set_m)+'gE_'+str(gE_m)+'tau_'+str(ssn_pars.tauE)+'-'+str(ssn_pars.tauI)+'fE_'+str(f_E))
if os.path.exists(results_dir) == False:
        os.makedirs(results_dir)

run_dir = os.path.join(results_dir, 'set'+str(init_set_m)+'_gEI_'+str(gE[0])+'-'+str(gI[0]))
mid_ref = os.path.join(run_dir+'mid_ref')
mid_target = os.path.join(run_dir+'mid_target')
sup_ref = os.path.join(run_dir+'sup_ref')
sup_target = os.path.join(run_dir+'sup_target')
responses_dir = os.path.join(run_dir+'responses_dir')


############################################################################

inds = [30, 31, 32, 39, 40, 41, 48, 49, 50]

#Create stimuli
#train_data = create_data(stimuli_pars, number = 1, offset = offset, ref_ori = ref_ori)
train_data = util.load(os.path.join(os.getcwd(), 'results', '15-05', 'Gabor_plots', 'setC_gE_[0.25, 0.37328625]training_data'))
ssn_ori_map_loaded = np.load(os.path.join(os.getcwd(), 'results', '15-05', 'Gabor_plots', 'setC_gE_[0.25, 0.37328625]ori_map.npy'), allow_pickle = True)


#Intialise SSNs 

ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_m, filter_pars=filter_pars, J_2x2=J_2x2_m, gE = gE_m, gI=gI_m, ori_map = ssn_ori_map_loaded)
ssn_pars.A = ssn_mid.A
ssn_sup=SSN2DTopoV1(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_s, filter_pars=filter_pars, J_2x2=J_2x2_s, s_2x2=s_2x2_s, gE = gE_s, gI=gI_s, sigma_oris = sigma_oris, ori_map = ssn_mid.ori_map)


constant_vector = constant_to_vec(c_E, c_I, ssn=ssn_mid)
constant_vector_sup = constant_to_vec(c_E, c_I, ssn = ssn_sup, sup=True)


output_ref=np.matmul(ssn_mid.gabor_filters, train_data['ref'].squeeze()) 
output_target=np.matmul(ssn_mid.gabor_filters, train_data['target'].squeeze())

#Rectify output
SSN_input_ref=np.maximum(0, output_ref) +  constant_vector
SSN_input_target=np.maximum(0, output_target) + constant_vector
r_init = np.zeros(SSN_input_ref.shape[0])


plt.imshow(ssn_mid.ori_map.reshape(9,9), vmin = ssn_mid.ori_map.min(), vmax = ssn_mid.ori_map.max(), cmap='hsv')
plt.colorbar()
plt.savefig(os.path.join(run_dir+'_ori_map.png'))
plt.close()

plot_vec2map(ssn_mid, output_ref, save_fig=os.path.join(run_dir+'output_ref'))
plot_vec2map(ssn_mid, SSN_input_ref, save_fig=os.path.join(run_dir+'SSN_input_ref'))

PLOT=True
#Find fixed point for middle layer
r_ref_mid, r_max_ref_mid, avg_dx_ref_mid, fp = middle_layer_fixed_point(ssn_mid, SSN_input_ref, conv_pars, PLOT=PLOT, inds = inds,  save=mid_ref, return_fp = True)
r_target_mid, r_max_target_mid, avg_dx_target_mid, _ = middle_layer_fixed_point(ssn_mid, SSN_input_target, conv_pars, PLOT=PLOT, inds = inds, save=mid_target, return_fp = True)

#Input to superficial layer
sup_input_ref = np.hstack([r_ref_mid*f_E, r_ref_mid*f_I]) + constant_vector_sup
sup_input_target = np.hstack([r_target_mid*f_E, r_target_mid*f_I]) + constant_vector_sup

plot_vec2map(ssn_mid, fp, save_fig=os.path.join(run_dir+'mid_response'))

#Find fixed point for superficial layer
r_ref, r_max_ref_sup, avg_dx_ref_sup= obtain_fixed_point_centre_E(ssn_sup, sup_input_ref, conv_pars, PLOT=PLOT, inds = inds, save = sup_ref)

r_target, r_max_target_sup, avg_dx_target_sup= obtain_fixed_point_centre_E(ssn_sup, sup_input_target, conv_pars, PLOT=PLOT, inds = inds, save = sup_target)



j_s = obtain_min_max_indices(ssn_mid, fp)





#Plot tuning curves for minimum and maximum

for j in j_s:
    print('Fixed point response '+ str(fp[j])+ ' index '+str(j))
    tuning_dir = os.path.join(results_dir, 'tuning_curves'+str(j))
    plot_tuning_curves(ssn = ssn_mid, index = j, conv_pars = conv_pars, stimuli_pars =stimuli_pars, save_fig =tuning_dir)

#Plot gabor filters of neurons
gabor_dir = os.path.join(results_dir, 'gabor_plots')
plot_mutiple_gabor_filters(ssn_mid, fp, save_fig=gabor_dir, indices=j_s)

fig, axes = plt.subplots(1,2, figsize=(8,8))
all_responses = [r_ref, r_target]
count = 0
for ax in axes.flat:

    im = ax.imshow(all_responses[count].reshape(5,5), vmin = r_ref.min(), vmax = r_ref.max())
    ax.set_xlabel(all_responses[count].max())

    count+=1

plt.colorbar(im, ax=axes.ravel().tolist())

plt.savefig(responses_dir+'sig_input.png')
plt.close()




        