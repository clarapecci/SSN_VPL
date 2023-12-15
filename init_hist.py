import os 
from SSN_classes_middle import SSN2DTopoV1_ONOFF_local
from util import init_set_func

import jax.numpy as np
import numpy
import jax.random as random

from init_hist_func import initial_acc
from pdb import set_trace
from parameters import *
from util import take_log
############################## PARAMETERS ############################


#Specify initialisation
init_set_m ='C'
init_set_s=1
J_2x2_s, s_2x2, gE_s, gI_s, conn_pars_s  = init_set_func(init_set_s, conn_pars_s, ssn_pars)
J_2x2_m, _, gE_m, gI_m, conn_pars_m  = init_set_func(init_set_m, conn_pars_m, ssn_pars, middle = True)


gE = [gE_m, gE_s]
gI = [gI_m, gI_s]
print('g s', gE, gI)



sigma_oris = np.asarray([90.0, 90.0])
kappa_pre = np.asarray([ 0.0, 0.0])
kappa_post = np.asarray([ 0.0, 0.0])


#Excitatory and inhibitory constants for extra synaptic GABA
c_E = 5.0
c_I = 5.0

#Feedforwards connections
f_E = np.log(1.1)
f_I = np.log(0.7)

b_sig = 0.0

ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_m, filter_pars=filter_pars, J_2x2=J_2x2_m, gE = gE[0], gI=gI[0], ori_map = ssn_ori_map_loaded)
ssn_pars.A = ssn_mid.A
ssn_pars.A2 =ssn_mid.A2 

logJ_2x2_s = take_log(J_2x2_s)
logJ_2x2_m = take_log(J_2x2_m)


readout_pars = dict(w_sig = None, b_sig = b_sig)
ssn_layer_pars = dict(J_2x2_m = logJ_2x2_m, J_2x2_s = logJ_2x2_s, c_E = c_E, c_I = c_I,  f_E = f_E, f_I = f_I, kappa_pre = kappa_pre, kappa_post = kappa_post)

init_stimuli_pars = StimuliPars()
class constant_pars:
    ssn_pars =ssn_pars
    s_2x2 = s_2x2
    sigma_oris = sigma_oris
    grid_pars = grid_pars
    conn_pars_m = conn_pars_m
    conn_pars_s = conn_pars_s
    conv_pars = conv_pars
    loss_pars = loss_pars
    gE = gE
    gI = gI
    filter_pars = filter_pars
    noise_type = 'poisson'
    ssn_ori_map = ssn_ori_map_loaded
    ref_ori = stimuli_pars.ref_ori
    

###################### SAVING DIRECTORY ########################################

results_dir = os.path.join(os.getcwd(), 'results', '11-12', 'init_hist', 'sig_noise')
if os.path.exists(results_dir) == False:
        os.makedirs(results_dir)

save_dir = os.path.join(results_dir, 'histogram_')
################################################################################

list_noise = np.linspace(10, 200, 8)
list_noise = np.asarray([0.0])
list_sig_noise = np.linspace(0, 1, 5)
print('here')
all_accuracies, low_acc = initial_acc(ssn_layer_pars = ssn_layer_pars, readout_pars = readout_pars, constant_pars =constant_pars, stimuli_pars = init_stimuli_pars, list_noise = list_noise, list_w_std = list_sig_noise, save_fig = save_dir, trials = 200)

for i in range(0, len(low_acc)):
    print(low_acc[i])
