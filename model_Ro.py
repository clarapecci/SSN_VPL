import jax
import matplotlib.pyplot as plt
import jax.numpy as np
from jax import vmap
import numpy
from IPython.core.debugger import set_trace
from SSN_classes_middle import SSN2DTopoV1_ONOFF_local
from SSN_classes_superficial import SSN2DTopoV1
from util import create_grating_pairs, create_grating_single, select_neurons

from util import take_log, sep_exponentiate, constant_to_vec, sigmoid, binary_loss, save_params_dict_two_stage, leaky_relu, homeo_loss, rates_loss


#rng_noise = numpy.random.default_rng(10)
def generate_noise(batch_size, length, N_readout = 125,  dt_readout = 0.2):
    '''
    Creates vectors of neural noise. Function creates N vectors, where N = batch_size, each vector of length = length. 
    '''
    
    sig_noise = 1/np.sqrt(dt_readout*N_readout)
    assert(sig_noise == 0.2)
    return sig_noise*numpy.random.randn(batch_size, length)


def obtain_fixed_point(ssn, ssn_input, conv_pars, PLOT=False, save=None, inds=None, print_dt = False):
    
    r_init = np.zeros(ssn_input.shape[0])
    dt = conv_pars.dt
    xtol = conv_pars.xtol
    Tmax = conv_pars.Tmax
    #Find fixed point
    if PLOT==True:
        fp, avg_dx = ssn.fixed_point_r_plot(ssn_input, r_init=r_init, dt=dt, xtol=xtol, Tmax=Tmax, PLOT=PLOT, save=save, inds=inds, print_dt = print_dt)
    else:
        fp, avg_dx = ssn.fixed_point_r(ssn_input, r_init=r_init, dt=dt, xtol=xtol, Tmax=Tmax, PLOT=PLOT, save=save)

    avg_dx = np.maximum(0, (avg_dx -1))
    return fp, avg_dx



def obtain_fixed_point_centre_E(ssn, ssn_input, conv_pars, inhibition = False, PLOT=False, save=None, inds=None, return_fp = False):
    
    #Obtain fixed point
    fp, avg_dx = obtain_fixed_point(ssn=ssn, ssn_input = ssn_input, conv_pars = conv_pars, PLOT = PLOT, save = save, inds = inds)

    #Apply bounding box to data
    r_box = (ssn.apply_bounding_box(fp, size=3.2)).ravel()
    
    #Obtain inhibitory response 
    if inhibition ==True:
        r_box_i = ssn.apply_bounding_box(fp, size=3.2, select='I_ON').ravel()
        r_box = [r_box, r_box_i]
 

    max_E = np.max(fp[:ssn.Ne])
    max_I = np.max(fp[ssn.Ne:-1])
    mean_E = np.mean(fp[:ssn.Ne])
    mean_I= np.mean(fp[ssn.Ne:-1])
    
    #Loss for high rates
    #r_max = np.maximum(0, (max_E/conv_pars.Rmax_E - 1)) + np.maximum(0, (max_I/conv_pars.Rmax_I - 1))
    #r_max = leaky_relu(r = max_E, R_thresh = conv_pars.Rmax_E_sup, slope_2 = 1/conv_pars.Rmax_E) + leaky_relu(max_I, R_thresh = conv_pars.Rmax_I, slope_2 = 1/conv_pars.Rmax_I)
    #r_max = homeo_loss(mean_E, max_E, R_mean_const = 12, R_max_const = 50) + homeo_loss(mean_I, max_I, R_mean_const = 41.5, R_max_const = 100)
    
    r_max = (rates_loss(max_E, conv_pars.Rmax_E_sup) +  rates_loss(max_I, conv_pars.Rmax_I_sup))*conv_pars.lambda_rmax +  (rates_loss(mean_E, conv_pars.Rmean_E_sup) +  rates_loss(mean_I, conv_pars.Rmean_I_sup))*conv_pars.lambda_rmean
  
    if return_fp ==True:
        return r_box, r_max, avg_dx, fp, max_E, max_I
    else:
        return r_box, r_max, avg_dx



def middle_layer_fixed_point(ssn, ssn_input, conv_pars, inhibition = False, PLOT=False, save=None, inds=None, return_fp = False, print_dt = False):
    
    fp, avg_dx = obtain_fixed_point(ssn=ssn, ssn_input = ssn_input, conv_pars = conv_pars, PLOT = PLOT, save = save, inds = inds, print_dt = print_dt)
    
    #Add responses from E and I neurons
    fp_E_on = ssn.select_type(fp, map_number = 1)
    fp_E_off = ssn.select_type(fp, map_number = (ssn.phases+1))

    layer_output = fp_E_on + fp_E_off
    E_responses, I_responses = select_neurons(fp, layer='mid')
    
    #Find maximum and mean rate
    max_E = np.max(E_responses)
    max_I = np.max(I_responses)
    mean_E = np.mean(E_responses)
    mean_I = np.mean(I_responses)
   
    if ssn.phases==4:
        fp_E_on_pi2 = ssn.select_type(fp, map_number = 3)
        fp_E_off_pi2 = ssn.select_type(fp, map_number = 7)

        #Changes
        layer_output = layer_output + fp_E_on_pi2 + fp_E_off_pi2    
        
    #Loss for high rates
    #r_max = homeo_loss(mean_E, max_E, R_mean_const = 6.1, R_max_const = 50) + homeo_loss(mean_I, max_I, R_mean_const = 10.3, R_max_const = 100)
    #r_max = np.maximum(0, (max_E/conv_pars.Rmax_E - 1)) + np.maximum(0, (max_I/conv_pars.Rmax_I - 1))
    #r_max = leaky_relu(r = max_E, R_thresh = conv_pars.Rmax_E, slope_2 = 1/conv_pars.Rmax_E) + leaky_relu(r = max_I, R_thresh = conv_pars.Rmax_I, slope_2 = 1/conv_pars.Rmax_I)
    
    r_max = rates_loss(max_E, conv_pars.Rmax_E_mid) +  rates_loss(max_I, conv_pars.Rmax_I_mid) +  rates_loss(mean_E, conv_pars.Rmean_E_mid) +  rates_loss(mean_I, conv_pars.Rmean_I_mid)
   
    if return_fp ==True:
            return layer_output, r_max, avg_dx, fp, max_E, max_I
    else:
        return layer_output, r_max, avg_dx


def two_layer_model(ssn_m, ssn_s, stimuli, conv_pars, constant_vector_mid, constant_vector_sup, f_E, f_I):
    
    '''
    Run individual stimulus through two layer model. 
    
    Inputs:
     ssn_mid, ssn_sup: middle and superficial layer classes
     stimuli: stimuli to pass through network
     conv_pars: convergence parameters for ssn 
     constant_vector_mid, constant_vector_sup: extra synaptic constants for middle and superficial layer
     f_E, f_I: feedforward connections between layers
    
    Outputs:
     r_sup - fixed point of centre neurons (5x5) in superficial layer
     loss related terms (wrt to middle and superficial layer) :
         - r_max_": loss minimising maximum rates
         - avg_dx_": loss minimising number of steps taken during convergence 
     max_(E/I)_(mid/sup): maximum rate for each type of neuron in each layer 
     
    '''
    
    #Find input of middle layer
    stimuli_gabor=np.matmul(ssn_m.gabor_filters, stimuli)
 
    #Rectify input
    SSN_mid_input = np.maximum(0, stimuli_gabor) + constant_vector_mid
    
    #Calculate steady state response of middle layer
    r_mid, r_max_mid, avg_dx_mid, fp_mid, max_E_mid, max_I_mid = middle_layer_fixed_point(ssn_m, SSN_mid_input, conv_pars, return_fp = True)

    #Concatenate input to superficial layer
    sup_input_ref = np.hstack([r_mid*f_E, r_mid*f_I]) + constant_vector_sup
    
    #Calculate steady state response of superficial layer
    r_sup, r_max_sup, avg_dx_sup, fp_sup, max_E_sup, max_I_sup= obtain_fixed_point_centre_E(ssn_s, sup_input_ref, conv_pars, return_fp= True)
    
    return r_sup, [r_max_mid, r_max_sup], [avg_dx_mid, avg_dx_sup], [max_E_mid, max_I_mid, max_E_sup, max_I_sup], [fp_mid, fp_sup]


                    
def ori_discrimination(ssn_layer_pars, readout_pars, constant_pars, train_data, noise_ref, noise_target):
    
    '''
    Orientation discrimanation task ran using SSN two-layer model.The reference and target are run through the two layer model individually. 
    Inputs:
        individual parameters - having taken logs of differentiable parameters
        noise_type: select different noise models
    Outputs:
        losses to take gradient with respect to
        sig_input, x: I/O values for sigmoid layer
    '''
    
    logJ_2x2_m = ssn_layer_pars['J_2x2_m']
    logJ_2x2_s = ssn_layer_pars['J_2x2_s']
    
    
    if 'kappa_pre' in ssn_layer_pars.keys():
        kappa_pre = np.tanh(ssn_layer_pars['kappa_pre'])
        kappa_post = np.tanh(ssn_layer_pars['kappa_post'])
    else:
        kappa_pre =  constant_pars.kappa_pre 
        kappa_post = constant_pars.kappa_post 
    if 'c_E' in ssn_layer_pars.keys():
        c_E = ssn_layer_pars['c_E']
        c_I = ssn_layer_pars['c_I']
    else:
        c_E = constant_pars.c_E
        c_I  = constant_pars.c_I
        
    if 'f_E' in ssn_layer_pars.keys():
        f_E = np.exp(ssn_layer_pars['f_E'])
        f_I = np.exp(ssn_layer_pars['f_I'])
    else:
        f_E = constant_pars.f_E
        f_I  = constant_pars.f_I

    w_sig = readout_pars['w_sig']
    b_sig = readout_pars['b_sig']
    loss_pars = constant_pars.loss_pars
    conv_pars = constant_pars.conv_pars
    J_2x2_m = sep_exponentiate(logJ_2x2_m)
    J_2x2_s = sep_exponentiate(logJ_2x2_s)
    ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, conn_pars=constant_pars.conn_pars_m, filter_pars=constant_pars.filter_pars, J_2x2=J_2x2_m, gE = constant_pars.gE[0], gI=constant_pars.gI[0], ori_map = constant_pars.ssn_ori_map)
    ssn_sup=SSN2DTopoV1(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, conn_pars=constant_pars.conn_pars_s, J_2x2=J_2x2_s, s_2x2=constant_pars.s_2x2, sigma_oris = constant_pars.sigma_oris, ori_map = constant_pars.ssn_ori_map, train_ori = constant_pars.ref_ori, kappa_post = kappa_post, kappa_pre = kappa_pre)
    
    #Recalculate new connectivity matrix
    #ssn_mid.make_local_W(J_2x2_m)
    #ssn_sup.make_W(J_2x2_s, kappa_pre, kappa_post)
    
    #Create vector of extrasynaptic constants
    constant_vector_mid = constant_to_vec(c_E = c_E, c_I = c_I, ssn= ssn_mid)
    constant_vector_sup = constant_to_vec(c_E = c_E, c_I = c_I, ssn = ssn_sup, sup=True)
    
    #Run reference through two layer model
    r_ref, [r_max_ref_mid, r_max_ref_sup], [avg_dx_ref_mid, avg_dx_ref_sup],[max_E_mid, max_I_mid, max_E_sup, max_I_sup], _ = two_layer_model(ssn_mid, ssn_sup, train_data['ref'], conv_pars, constant_vector_mid, constant_vector_sup, f_E, f_I)
    
    #Run target through two layer model
    r_target, [r_max_target_mid, r_max_target_sup], [avg_dx_target_mid, avg_dx_target_sup], _, _= two_layer_model(ssn_mid, ssn_sup, train_data['target'], conv_pars, constant_vector_mid, constant_vector_sup, f_E, f_I)
    
    noise_type = constant_pars.noise_type
    #Add noise
    if noise_type =='additive':
        r_ref = r_ref + noise_ref
        r_target = r_target + noise_target
        
    elif noise_type == 'multiplicative':
        r_ref = r_ref*(1 + noise_ref)
        r_target = r_target*(1 + noise_target)
        
    elif noise_type =='poisson':
        r_ref = r_ref + noise_ref*np.sqrt(jax.nn.softplus(r_ref))
        r_target = r_target + noise_target*np.sqrt(jax.nn.softplus(r_target))
    
    #Find difference between reference and target
    delta_x = r_ref - r_target
    
    #Multiply delta by sigmoid lyer weights and add bias
    sig_input = np.dot(w_sig, (delta_x)) + b_sig 
    
    #Apply sigmoid function - combine ref and target
    x = sigmoid(sig_input)
    
    #Calculate losses
    loss_binary=binary_loss(train_data['label'], x)
    loss_avg_dx = loss_pars.lambda_dx*(avg_dx_ref_mid + avg_dx_target_mid + avg_dx_ref_sup + avg_dx_target_sup )/4
    loss_r_max =  loss_pars.lambda_r_max*(r_max_ref_mid + r_max_target_mid + r_max_ref_sup + r_max_target_sup )/4
    loss_w = loss_pars.lambda_w*(np.linalg.norm(w_sig)**2)
    loss_b = loss_pars.lambda_b*(b_sig**2)
    
    #Combine all losses
    loss = loss_binary + loss_w + loss_b +  loss_avg_dx + loss_r_max
    all_losses = np.vstack((loss_binary, loss_avg_dx, loss_r_max, loss_w, loss_b, loss))
    pred_label = np.round(x) 
    
    return loss, all_losses, pred_label, sig_input, x,  [max_E_mid, max_I_mid, max_E_sup, max_I_sup]


                       
#Parallelize orientation discrimination task
vmap_ori_discrimination = vmap(ori_discrimination, in_axes = ({'J_2x2_m': None, 'J_2x2_s':None, 'c_E':None, 'c_I':None, 'f_E':None, 'f_I':None, 'kappa_pre':None, 'kappa_post':None}, {'w_sig':None, 'b_sig':None}, None, {'ref':0, 'target':0, 'label':0}, 0, 0) )
jit_ori_discrimination = jax.jit(vmap_ori_discrimination, static_argnums = [2])

#Parallelize task - kappa frozen
vmap_no_kappa = vmap(ori_discrimination, in_axes = ({'J_2x2_m': None, 'J_2x2_s':None, 'c_E':None, 'c_I':None, 'f_E':None, 'f_I':None}, {'w_sig':None, 'b_sig':None}, None, {'ref':0, 'target':0, 'label':0}, 0, 0) )
jit_no_kappa= jax.jit(vmap_no_kappa, static_argnums = [2])

#Parallelize orientation discrimination task - frozen parameters
vmap_ori_discrimination_frozen_pars = vmap(ori_discrimination, in_axes = ({'J_2x2_m': None, 'J_2x2_s':None}, {'w_sig':None, 'b_sig':None}, None, {'ref':0, 'target':0, 'label':0}, 0, 0) )
jit_ori_discrimination_frozen = jax.jit(vmap_ori_discrimination_frozen_pars, static_argnums = [2])








def response_matrix(J_2x2_m, J_2x2_s, kappa_pre, kappa_post, c_E, c_I, f_E, f_I, constant_pars, tuning_pars, radius_list, ori_list, trained_ori):
    '''
    Construct a response matrix of sizze n_orientations x n_neurons x n_radii
    '''
    #Initialize ssn
    ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, conn_pars=constant_pars.conn_pars_m, filter_pars=constant_pars.filter_pars, J_2x2=J_2x2_m, gE = constant_pars.gE[0], gI=constant_pars.gI[0], ori_map = constant_pars.ssn_ori_map)
    ssn_sup=SSN2DTopoV1(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, conn_pars=constant_pars.conn_pars_s, J_2x2=J_2x2_s, s_2x2=constant_pars.s_2x2, sigma_oris = constant_pars.sigma_oris, ori_map = constant_pars.ssn_ori_map, train_ori = trained_ori, kappa_post = kappa_post, kappa_pre = kappa_pre)

    responses_sup = []
    responses_mid = []
    inputs = []
    conv_pars = constant_pars.conv_pars
    constant_vector_mid = constant_to_vec(c_E = c_E, c_I = c_I, ssn= ssn_mid)
    constant_vector_sup = constant_to_vec(c_E = c_E, c_I = c_I, ssn = ssn_sup, sup=True)
    
    for i in range(len(ori_list)):
        
        #Find responses at different stimuli radii
        x_response_sup, x_response_mid = surround_suppression(ssn_mid, ssn_sup, tuning_pars, conv_pars, radius_list, constant_vector_mid, constant_vector_sup, f_E, f_I, ref_ori = ori_list[i])
        print(x_response_sup.shape)
        #inputs.append(SSN_input)
        responses_sup.append(x_response_sup)
        responses_mid.append(x_response_mid)
        
    return np.stack(responses_sup, axis = 2), np.stack(responses_mid, axis = 2)




def surround_suppression(ssn_mid, ssn_sup, tuning_pars, conv_pars, radius_list, constant_vector_mid, constant_vector_sup, f_E, f_I, ref_ori, title= None):    
    
    '''
    Produce matrix response for given two layer ssn network given a list of varying stimuli radii
    '''
    
    all_responses_sup = []
    all_responses_mid = []
    
    tuning_pars.ref_ori = ref_ori
   
    print(ref_ori) #create stimuli in the function just input radii)
    for radii in radius_list:
        
        tuning_pars.outer_radius = radii
        tuning_pars.inner_radius = radii*(2.5/3)
        
        stimuli = create_grating_single(n_trials = 1, stimuli_pars = tuning_pars)
        stimuli = stimuli.squeeze()
        
        #stimuli = np.load('/mnt/d/ABG_Projects_Backup/ssn_modelling/ssn-simulator/debugging/new_stimuli.npy')
        
        r_sup, _, _, [max_E_mid, max_I_mid, max_E_sup, max_I_sup], [fp_mid, fp_sup] = two_layer_model(ssn_mid, ssn_sup, stimuli, conv_pars, constant_vector_mid, constant_vector_sup, f_E, f_I)
         
        all_responses_sup.append(fp_sup.ravel())
        all_responses_mid.append(fp_mid.ravel())
        print('Mean population response {} (max in population {}), centre neurons {}'.format(fp_sup.mean(), fp_sup.max(), fp_sup.mean()))
    
    if title:
        plt.plot(radius_list, np.asarray(all_responses))
        plt.xlabel('Radius')
        plt.ylabel('Response of centre neuron')
        if title:
            plt.title(title)
        plt.show()
    
    return np.vstack(all_responses_sup), np.vstack(all_responses_mid)



    
