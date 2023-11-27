import os
import jax
import jax.numpy as np
import optax
import time
import csv
from IPython.core.debugger import set_trace
from SSN_classes_middle import SSN2DTopoV1_ONOFF_local
from SSN_classes_superficial import SSN2DTopoV1
from util import create_grating_pairs, create_grating_single, take_log, save_params_dict_two_stage

from model import jit_ori_discrimination, generate_noise

from analysis import plot_max_rates, plot_w_sig






def training_loss_staircase(ssn_layer_pars, readout_pars, constant_pars, train_data, noise_ref, noise_target, threshold=0.79):
    
    '''
    Run orientation discrimination task on given batch of data. Returns losses averaged over the trials within the batch. Function over which the gradient is taken.
    '''
    
    #Run orientation discrimination task
    total_loss, all_losses, pred_label, sig_input, x, max_rates = jit_ori_discrimination(ssn_layer_pars, readout_pars, constant_pars, train_data, noise_ref, noise_target)
    
    #Total loss to take gradient with respect to 
    loss= np.mean(total_loss)
    
    #Find mean of different losses
    all_losses = np.mean(all_losses, axis = 0)
    
    #Find maximum rates across trials
    max_rates = [item.max() for item in max_rates]
    
    #Calculate accuracy 
    true_accuracy = np.sum(train_data['label'] == pred_label)/len(train_data['label'])  
    
    if true_accuracy< threshold:
        thresh_variable = 0
    else:
        thresh_variable = 1
        
    
    return loss, [all_losses, true_accuracy, sig_input, x, max_rates, thresh_variable]




def train_model_staircase(ssn_layer_pars, readout_pars, constant_pars, training_pars, stimuli_pars, results_filename = None, results_dir = None, step = 0.25):
    
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
    save_w_sigs.append(readout_pars['w_sig'][:5])
    all_offsets = []
  
    #Check for orientation map
    if constant_pars.ssn_ori_map ==None:
        ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, conn_pars=constant_pars.conn_pars_m, filter_pars=constant_pars.filter_pars, J_2x2=ssn_layer_pars['J_2x2_m'], gE = constant_pars.gE[0], gI=constant_pars.gI[0])
        constant_pars.ssn_ori_map  = ssn_mid.ori_map
        
                         
    batch_size = training_pars.batch_size

    #Take logs of parameters
    ssn_layer_pars['J_2x2_m'] = take_log(ssn_layer_pars['J_2x2_m'])
    ssn_layer_pars['J_2x2_s'] = take_log(ssn_layer_pars['J_2x2_s'])

    #Validation test size equals batch size
    test_size = training_pars.batch_size
    
    #Find epochs to save
    epochs_to_save =  np.linspace(0, training_pars.epochs, training_pars.num_epochs_to_save).astype(int)
    epochs_to_save = epochs_to_save.at[0].set(1)

        
    #Initialise optimizer
    optimizer = optax.adam(training_pars.eta)
    readout_state = optimizer.init(readout_pars)
    
    print('Training model for {} epochs  with learning rate {}, sig_noise {} at offset {}, lam_w {}, batch size {}, noise_type {}'.format(training_pars.epochs, training_pars.eta, training_pars.sig_noise, stimuli_pars.offset, constant_pars.loss_pars.lambda_w, batch_size, constant_pars.noise_type))


    #Initialise csv file
    if results_filename:
        print('Saving results to csv ', results_filename)
    else:
        print('#### NOT SAVING! ####')
    
    #Gradient descent function
    loss_and_grad_readout = jax.value_and_grad(training_loss_staircase, argnums=1, has_aux =True)
    loss_and_grad_ssn = jax.value_and_grad(training_loss_staircase, argnums=0, has_aux = True)

   
    ######## FIRST STAGE OF TRAINING #############
    for epoch in range(1, training_pars.epochs+1):
        start_time = time.time()
           
        #Load next batch of data and convert
        train_data = create_grating_pairs(stimuli_pars = stimuli_pars, n_trials = batch_size)
            
        #Generate noise
        noise_ref = generate_noise(training_pars.sig_noise, batch_size, readout_pars['w_sig'].shape[0])
        noise_target = generate_noise(training_pars.sig_noise, batch_size, readout_pars['w_sig'].shape[0])

        #Compute loss and gradient
        [epoch_loss, [epoch_all_losses, train_true_acc, train_delta_x, train_x, train_r_ref,_]], grad =loss_and_grad_readout(ssn_layer_pars, readout_pars, constant_pars, train_data, noise_ref, noise_target)
        
        
        if epoch==1:
            all_losses = epoch_all_losses
        else:
            all_losses = np.hstack((all_losses, epoch_all_losses)) 
        
        training_losses.append(epoch_loss)
        train_accs.append(train_true_acc)
        train_sig_input.append(train_delta_x)
        train_sig_output.append(train_x)
        r_refs.append(train_r_ref)
        all_offsets.append(stimuli_pars.offset)
 
        epoch_time = time.time() - start_time
        

        #Save the parameters given a number of epochs
        if epoch in epochs_to_save:
            
            #Evaluate model 
            test_data = create_grating_pairs(stimuli_pars = stimuli_pars, n_trials = test_size)
            
            #Generate noise
            noise_ref = generate_noise(training_pars.sig_noise, batch_size, readout_pars['w_sig'].shape[0])
            noise_target = generate_noise(training_pars.sig_noise, batch_size, readout_pars['w_sig'].shape[0])
            
            start_time = time.time()
            
            #Calculate loss for testing data
            [val_loss, [val_all_losses, true_acc, val_delta_x, val_x, _ , _]], _= loss_and_grad_readout(ssn_layer_pars, readout_pars, constant_pars, test_data, noise_ref, noise_target)
            val_time = time.time() - start_time
            
            print('Training loss: {} ¦ Validation -- loss: {}, true accuracy: {}, at epoch {}, (time {}, {}), '.format(epoch_loss, val_loss, true_acc, epoch, epoch_time, val_time))
            
            #Every 50 epochs print individual values of the loss
            if epoch%50 ==0:
                    print('Training accuracy: {}, all losses{}'.format(np.mean(np.asarray(train_accs[-20:])), epoch_all_losses))
            
            #Save validation loss
            val_loss_per_epoch.append([val_loss, int(epoch)])
            val_sig_input.append([val_delta_x, epoch])
            val_sig_output.append(val_x)
            val_accs.append(true_acc)
            
            #Save parameters
            if results_filename:
                save_params = save_params_dict_two_stage(ssn_layer_pars, readout_pars, true_acc, epoch)
                
                #Initialise results file
                if epoch==1:
                        results_handle = open(results_filename, 'w')
                        results_writer = csv.DictWriter(results_handle, fieldnames=save_params.keys(), delimiter=',')
                        results_writer.writeheader()
                        
                results_writer.writerow(save_params)

            
        #Early stop in first stage of training
        if epoch>20 and  np.mean(np.asarray(train_accs[-20:]))>training_pars.first_stage_acc:
            print('Early stop: {} accuracy achieved at epoch {}'.format(training_pars.first_stage_acc, epoch))
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
        save_w_sigs.append(readout_pars['w_sig'][:5])
                   
#############START TRAINING NEW STAGE ##################################    
    
    print('Entering second stage at epoch {}'.format(epoch))  
    #Restart number of epochs
    
    #Reinitialize optimizer for second stage
    ssn_layer_state = optimizer.init(ssn_layer_pars)
    threshold_variables = []
    
    for epoch in range(1, training_pars.epochs +1):
                
        #Generate next batch of data
        train_data = create_grating_pairs(stimuli_pars = stimuli_pars, n_trials = batch_size)
        
        #Generate noise
        noise_ref = generate_noise(training_pars.sig_noise, batch_size, readout_pars['w_sig'].shape[0])
        noise_target = generate_noise(training_pars.sig_noise, batch_size, readout_pars['w_sig'].shape[0])
         
        #Run model and calculate gradient    
        [epoch_loss, [epoch_all_losses, train_true_acc, train_delta_x, train_x, train_r_ref, thresh_variable]], grad =loss_and_grad_ssn(ssn_layer_pars, readout_pars, constant_pars, train_data, noise_ref, noise_target)
        
        #Save training losses
        all_losses = np.hstack((all_losses, epoch_all_losses))
        training_losses.append(epoch_loss)
        train_accs.append(train_true_acc)
        train_sig_input.append(train_delta_x)
        train_sig_output.append(train_x)
        r_refs.append(train_r_ref)
        threshold_variables.append(thresh_variable)
        all_offsets.append(stimuli_pars.offset)
        
        #Implementation of staircase
        if thresh_variable ==0:
            stimuli_pars.offset =  stimuli_pars.offset + step
        
        if epoch>3 and np.sum(np.asarray(threshold_variables[-3:])) ==3:
            stimuli_pars.offset =  stimuli_pars.offset - step
            threshold_variables = []
            
        print(stimuli_pars.offset)
        #Save the parameters given a number of epochs
        if epoch in epochs_to_save:

            #Evaluate model 
            test_data = create_grating_pairs(stimuli_pars = stimuli_pars, n_trials = test_size)
            #Generate noise
            noise_ref = generate_noise(training_pars.sig_noise, batch_size, readout_pars['w_sig'].shape[0])
            noise_target = generate_noise(training_pars.sig_noise, batch_size, readout_pars['w_sig'].shape[0])

            start_time = time.time()

            [val_loss, [val_all_losses, true_acc, val_delta_x, val_x, _, _]], _= loss_and_grad_ssn(ssn_layer_pars, readout_pars, constant_pars, test_data, noise_ref, noise_target)
            val_time = time.time() - start_time
            print('Training loss: {} ¦ Validation -- loss: {}, true accuracy: {}, at epoch {}, (time {}, {})'.format(epoch_loss, val_loss, true_acc, epoch, epoch_time, val_time))

            if epoch%50 ==0 or epoch==1:
                print('Training accuracy: {}, all losses{}'.format(train_true_acc, epoch_all_losses))

            val_loss_per_epoch.append([val_loss, epoch+first_stage_final_epoch])
            val_sig_input.append([val_delta_x, epoch+first_stage_final_epoch])
            val_sig_output.append(val_x)

            if results_filename:
                    save_params = save_params_dict_two_stage(ssn_layer_pars, readout_pars, true_acc, epoch = epoch+first_stage_final_epoch)
                    results_writer.writerow(save_params)
        
        
        #Update parameters
        updates, ssn_layer_state = optimizer.update(grad, ssn_layer_state)
        ssn_layer_pars = optax.apply_updates(ssn_layer_pars, updates)

    
    #Plot evolution of w_sig values
    save_w_sigs = np.asarray(np.vstack(save_w_sigs))
    plot_w_sig(save_w_sigs, epochs_to_save[:len(save_w_sigs)], first_stage_final_epoch, save = os.path.join(results_dir+'_w_sig_evolution') )
    
    #Save transition epochs to plot losses and accuracy
    epochs_plot = first_stage_final_epoch

    #Plot maximum rates achieved during training
    r_refs = np.vstack(np.asarray(r_refs))
    plot_max_rates(r_refs, epochs_plot = epochs_plot, save= os.path.join(results_dir+'_max_rates'))

    return [ssn_layer_pars, readout_pars], np.vstack([val_loss_per_epoch]), all_losses, train_accs, train_sig_input, train_sig_output, val_sig_input, val_sig_output, epochs_plot, save_w_sigs, all_offsets
