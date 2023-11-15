import os
import jax
from jax import random
import matplotlib.pyplot as plt
import jax.numpy as np
from jax import vmap
import optax
from functools import partial
import time
import numpy
from util import create_gratings, create_stimuli
from IPython.core.debugger import set_trace
from SSN_classes_phases import SSN2DTopoV1_ONOFF_local
from SSN_classes_jax_on_only import SSN2DTopoV1

from two_layer_training_lateral_phases import take_log, create_data, sep_exponentiate, middle_layer_fixed_point, obtain_fixed_point_centre_E, constant_to_vec, sigmoid, binary_loss, save_params_dict_two_stage



def generate_noise(sig_noise,  batch_size, length):
    '''
    Creates vectors of neural noise. Function creates N vectors, where N = batch_size, each vector of length = length. 
    '''
    return sig_noise*numpy.random.randn(batch_size, length)



def two_layer_model(ssn_mid, ssn_sup, stimuli, conv_pars, constant_vector_mid, constant_vector_sup, f_E, f_I):
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
    SSN_mid_input=np.matmul(ssn_mid.gabor_filters, stimuli)
 
    #Rectify input
    SSN_mid_input = np.maximum(0, SSN_mid_input) + constant_vector_mid
    
    #Calculate steady state response of middle layer
    r_mid, r_max_mid, avg_dx_mid, _, max_E_mid, max_I_mid = middle_layer_fixed_point(ssn_mid, SSN_mid_input, conv_pars, return_fp = True)
    
    #Concatenate input to superficial layer
    sup_input_ref = np.hstack([r_mid*f_E, r_mid*f_I]) + constant_vector_sup

    #Calculate steady state response of superficial layer
    r_sup, r_max_sup, avg_dx_sup, _, max_E_sup, max_I_sup= obtain_fixed_point_centre_E(ssn_sup, sup_input_ref, conv_pars, return_fp= True)
    
    return r_sup, [r_max_mid, r_max_sup], [avg_dx_mid, avg_dx_sup], [max_E_mid, max_I_mid, max_E_sup, max_I_sup]




                            
def ori_discrimination(ssn_layer_pars, readout_pars, conv_pars, loss_pars, ssn_mid, ssn_sup, train_data, noise_ref, noise_target, noise_type = 'poisson'):
    
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
    c_E = ssn_layer_pars['c_E']
    c_I = ssn_layer_pars['c_I']
    f_E = ssn_layer_pars['f_E']
    f_I = ssn_layer_pars['f_I']
    kappa_pre = ssn_layer_pars['kappa_pre']
    kappa_post = ssn_layer_pars['kappa_post']
    
    w_sig = readout_pars['w_sig']
    b_sig = readout_pars['b_sig']
    
    J_2x2_m = sep_exponentiate(logJ_2x2_m)
    J_2x2_s = sep_exponentiate(logJ_2x2_s)
    
    #Recalculate new connectivity matrix
    ssn_mid.make_local_W(J_2x2_m)
    ssn_sup.make_W(J_2x2_s, kappa_pre, kappa_post)
    
    #Create vector of extrasynaptic constants
    constant_vector_mid = constant_to_vec(c_E = c_E, c_I = c_I, ssn= ssn_mid)
    constant_vector_sup = constant_to_vec(c_E = c_E, c_I = c_I, ssn = ssn_sup, sup=True)
    
    #Run reference through two layer model
    r_ref, [r_max_ref_mid, r_max_ref_sup], [avg_dx_ref_mid, avg_dx_ref_sup],[max_E_mid, max_I_mid, max_E_sup, max_I_sup] = two_layer_model(ssn_mid, ssn_sup, train_data['ref'], conv_pars, constant_vector_mid, constant_vector_sup, f_E, f_I)
    
    #Run target through two layer model
    r_target, [r_max_target_mid, r_max_target_sup], [avg_dx_target_mid, avg_dx_target_sup], _= two_layer_model(ssn_mid, ssn_sup, train_data['target'], conv_pars, constant_vector_mid, constant_vector_sup, f_E, f_I)
    
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

    delta_x = r_ref - r_target
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
vmap_ori_discrimination = vmap(ori_discrimination, in_axes = ({'J_2x2_m': None, 'J_2x2_s':None, 'c_E':None, 'c_I':None, 'f_E':None, 'f_I':None, 'kappa_pre':None, 'kappa_post':None}, {'w_sig':None, 'b_sig':None}, None, None, None, None, {'ref':0, 'target':0, 'label':0}, 0, 0, None) )

#Jit vmap function
jit_ori_discrimination = jax.jit(vmap_ori_discrimination, static_argnums = [2, 3, 4, 5, 9])




def training_loss(ssn_layer_pars, readout_pars, conv_pars, loss_pars, ssn_mid, ssn_sup, train_data, noise_ref, noise_target, noise_type):
    
    '''
    Run orientation discrimination task on given batch of data. Returns losses averaged over the trials within the batch. Function over which the gradient is taken.
    '''
    
    #Run orientation discrimination task
    total_loss, all_losses, pred_label, sig_input, x, max_rates = jit_ori_discrimination(ssn_layer_pars, readout_pars, conv_pars, loss_pars, ssn_mid, ssn_sup, train_data, noise_ref, noise_target, noise_type)
    
    #Total loss to take gradient with respect to 
    loss= np.mean(total_loss)
    
    #Find mean of different kind of loss
    all_losses = np.mean(all_losses, axis = 0)
    
    #Find maximum rates across batches
    max_rates = [item.max() for item in max_rates]
    
    #Calculate accuracy 
    true_accuracy = np.sum(train_data['label'] == pred_label)/len(train_data['label'])  
    
    return loss, [all_losses, true_accuracy, sig_input, x, max_rates]






def train_model(ssn_layer_pars, readout_pars, constant_pars, conv_pars, loss_pars, training_pars, stimuli_pars, results_filename = None, ref_ori = 55, offset = 4, results_dir = None, ssn_ori_map=None):
    
    '''
    Training function for two layer model in two stages: first train readout layer up until early_acc ( default 70% accuracy), in second stage SSN layer parameters are trained.
    '''
    
    #Initialize loss
    val_loss_per_epoch = []
    training_losses=[]
    train_accs = []
    train_sig_input = []
    train_sig_output = []
    val_sig_input = []
    val_sig_output = []
    val_accs=[]
    r_refs = []
    save_w_sigs = []
    first_stage_final_epoch = training_pars.epochs
    #save_w_sigs.append(w_sig[:5])
  
    
    #Initialize networks
    
    ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=constant_pars['ssn_pars'], grid_pars=constant_pars['grid_pars'], conn_pars=constant_pars['conn_pars_m'], filter_pars=constant_pars['filter_pars'], J_2x2=ssn_layer_pars['J_2x2_m'], gE = constant_pars['gE'][0], gI=constant_pars['gI'][0], ori_map = ssn_ori_map)
    
    ssn_sup=SSN2DTopoV1(ssn_pars=constant_pars['ssn_pars'], grid_pars=constant_pars['grid_pars'], conn_pars=constant_pars['conn_pars_s'], J_2x2=ssn_layer_pars['J_2x2_s'], s_2x2=constant_pars['s_2x2'], sigma_oris = constant_pars['sigma_oris'], ori_map = ssn_ori_map, train_ori = ref_ori, kappa_post = ssn_layer_pars['kappa_post'], kappa_pre = ssn_layer_pars['kappa_pre'])
    
    batch_size = training_pars.batch_size
    ref_ori = training_pars.ref_ori
    offset = training_pars.offset
    
    #Take logs of parameters
    ssn_layer_pars['J_2x2_m'] = take_log(ssn_layer_pars['J_2x2_m'])
    ssn_layer_pars['J_2x2_s'] = take_log(ssn_layer_pars['J_2x2_s'])

    #Validation test size equals batch size
    test_size = training_pars.batch_size
    
    #Find epochs to save
    epochs_to_save =  np.insert((np.unique(np.linspace(1 , training_pars.epochs, training_pars.num_epochs_to_save).astype(int))), 0 , 0)
        
    #Initialise optimizer
    optimizer = optax.adam(training_pars.eta)
    readout_state = optimizer.init(readout_pars)
    
    print('Training model for {} epochs  with learning rate {}, sig_noise {} at offset {}, lam_w {}, batch size {}, noise_type {}'.format(training_pars.epochs, training_pars.eta, training_pars.sig_noise, offset, loss_pars.lambda_w, batch_size, training_pars.noise_type))


    #Initialise csv file
    if results_filename:
        print('Saving results to csv ', results_filename)
    else:
        print('#### NOT SAVING! ####')
    
    #Gradient descent function
    loss_and_grad_readout = jax.value_and_grad(training_loss, argnums=1, has_aux =True)
    loss_and_grad_ssn = jax.value_and_grad(training_loss, argnums=0, has_aux = True)

   
    ######## FIRST STAGE OF TRAINING #############
    for epoch in range(0, training_pars.epochs+1):
        start_time = time.time()
           
        #Load next batch of data and convert
        train_data = create_data(stimuli_pars, n_trials = batch_size, offset = offset, ref_ori = ref_ori)
            
        #Generate noise
        noise_ref = generate_noise(training_pars.sig_noise, batch_size, readout_pars['w_sig'].shape[0])
        noise_target = generate_noise(training_pars.sig_noise, batch_size, readout_pars['w_sig'].shape[0])

        #Compute loss and gradient
        [epoch_loss, [epoch_all_losses, train_true_acc, train_delta_x, train_x, train_r_ref]], grad =loss_and_grad_readout(ssn_layer_pars, readout_pars, conv_pars, loss_pars, ssn_mid, ssn_sup, train_data, noise_ref, noise_target, training_pars.noise_type)
        
        training_losses.append(epoch_loss)
        if epoch==0:
            all_losses = epoch_all_losses
        else:
            all_losses = np.hstack((all_losses, epoch_all_losses)) 
        
        train_accs.append(train_true_acc)
        train_sig_input.append(train_delta_x)
        train_sig_output.append(train_x)
        r_refs.append(train_r_ref)
 
        epoch_time = time.time() - start_time
        

        #Save the parameters given a number of epochs
        if epoch in epochs_to_save:
            
            #Evaluate model 
            test_data = create_data(stimuli_pars, n_trials = test_size, offset = offset, ref_ori = ref_ori)
            #Generate noise
            noise_ref = generate_noise(training_pars.sig_noise, batch_size, readout_pars['w_sig'].shape[0])
            noise_target = generate_noise(training_pars.sig_noise, batch_size, readout_pars['w_sig'].shape[0])
            
            start_time = time.time()
            
            [val_loss, [val_all_losses, true_acc, val_delta_x, val_x, _ ]], _= loss_and_grad_readout(ssn_layer_pars, readout_pars, conv_pars, loss_pars, ssn_mid, ssn_sup, test_data, noise_ref, noise_target, training_pars.noise_type)
            val_time = time.time() - start_time
            
            print('Training loss: {} ¦ Validation -- loss: {}, true accuracy: {}, at epoch {}, (time {}, {}), '.format(epoch_loss, val_loss, true_acc, epoch, epoch_time, val_time))

            if epoch%50 ==0:
                    print('Training accuracy: {}, all losses{}'.format(np.mean(np.asarray(train_accs[-20:])), epoch_all_losses))
            
            val_loss_per_epoch.append([val_loss, int(epoch)])
            val_sig_input.append([val_delta_x, epoch])
            val_sig_output.append(val_x)
            val_accs.append(true_acc)
            
            #Save parameters
            if results_filename:
                save_params = save_params_dict_two_stage(ssn_layer_pars, readout_pars, true_acc, epoch)
                
                #Initialise results file
                if epoch==0:
                        results_handle = open(results_filename, 'w')
                        results_writer = csv.DictWriter(results_handle, fieldnames=save_params.keys(), delimiter=',')
                        results_writer.writeheader()
                        
                results_writer.writerow(save_params)

            
        #Early stop in first stage of training
        if epoch>20 and  np.mean(np.asarray(train_accs[-20:]))>early_stop:
            print('Early stop: {} accuracy achieved at epoch {}'.format(early_stop, epoch))
            print('Entering second stage at epoch {}'.format(epoch))   
            first_stage_final_epoch = epoch
            
            #Save final parameters
            if results_filename:
                save_params = save_params_dict_two_stage(ssn_layer_pars, readout_pars, true_acc, epoch)
                results_writer.writerow(save_params)
            
            #Exit first training loop
            break

        #Update readout parameters
        updates, readout_state = optimizer.update(grad, readout_state)
        readout_pars = optax.apply_updates(readout_pars, updates)
        #save_w_sigs.append(readout_pars['w_sig'][:5])
                   
#############START TRAINING NEW STAGE ##################################    
    #Restart number of epochs
    epoch = 0
    #Reinitialize optimizer for second stage
    ssn_layer_state = optimizer.init(ssn_layer_pars)
    
    for epoch in range(0, training_pars.epochs +1):
                
        #Generate next batch of data
        train_data = create_data(stimuli_pars, n_trials = batch_size, offset = offset, ref_ori = ref_ori)
        
        #Generate noise
        noise_ref = generate_noise(training_pars.sig_noise, batch_size, readout_pars['w_sig'].shape[0])
        noise_target = generate_noise(training_pars.sig_noise, batch_size, readout_pars['w_sig'].shape[0])
            

        [epoch_loss, [epoch_all_losses, train_true_acc, train_delta_x, train_x, train_r_ref]], grad =loss_and_grad_ssn(ssn_layer_pars, readout_pars, conv_pars, loss_pars, ssn_mid, ssn_sup, test_data, noise_ref, noise_target, training_pars.noise_type)
        
        #Save training losses
        all_losses = np.hstack((all_losses, epoch_all_losses))
        training_losses.append(epoch_loss)
        train_accs.append(train_true_acc)
        train_sig_input.append(train_delta_x)
        train_sig_output.append(train_x)
        r_refs.append(train_r_ref)

        #Save the parameters given a number of epochs
        if epoch in epochs_to_save:

            #Evaluate model 
            test_data = create_data(stimuli_pars, n_trials = test_size, offset = offset, ref_ori = ref_ori)
            #Generate noise
            noise_ref = generate_noise(training_pars.sig_noise, batch_size, readout_pars['w_sig'].shape[0])
            noise_target = generate_noise(training_pars.sig_noise, batch_size, readout_pars['w_sig'].shape[0])

            start_time = time.time()

            [val_loss, [val_all_losses, true_acc, val_delta_x, val_x, _]], _= loss_and_grad_ssn(ssn_layer_pars, readout_pars, conv_pars, loss_pars, ssn_mid, ssn_sup, test_data, noise_ref, noise_target, training_pars.noise_type)
            val_time = time.time() - start_time
            print('Training loss: {} ¦ Validation -- loss: {}, true accuracy: {}, at epoch {}, (time {}, {})'.format(epoch_loss, val_loss, true_acc, epoch, epoch_time, val_time))

            if epoch%50 ==0 or epoch==1:
                print('Training accuracy: {}, all losses{}'.format(train_true_acc, epoch_all_losses))

            val_loss_per_epoch.append([val_loss, epoch+first_stage_final_epoch])
            val_sig_input.append([val_delta_x, epoch+first_stage_final_epoch])
            val_sig_output.append(val_x)

            if results_filename:
                    save_params = save_params_dict_two_stage(ssn_layer_pars, readout_pars, true_acc, epoch = epoch+final_epoch)
                    results_writer.writerow(save_params)

        #Update parameters
        updates, ssn_layer_state = optimizer.update(grad, ssn_layer_state)
        ssn_layer_pars = optax.apply_updates(ssn_layer_pars, updates)

    #Plot evolution of w_sig values
    #save_w_sigs = np.asarray(np.vstack(save_w_sigs))
    #plot_w_sig(save_w_sigs, epochs_to_save[:len(save_w_sigs)], first_stage_final_epoch, save = os.path.join(results_dir+'_w_sig_evolution') )
    
    #Save transition epochs to plot losses and accuracy
    epochs_plot = [first_stage_final_epoch, epoch+first_stage_final_epoch]

    #Plot maximum rates achieved during training
    #r_refs = np.vstack(np.asarray(r_refs))
    #plot_max_rates(r_refs, epoch_c = epochs_plot, save= os.path.join(results_dir+'_max_rates'))

    
    return [ssn_layer_pars, readout_pars], np.vstack([val_loss_per_epoch]), all_losses, train_accs, train_sig_input, train_sig_output, val_sig_input, val_sig_output, epochs_plot, save_w_sigs


