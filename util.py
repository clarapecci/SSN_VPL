import jax
from jax import random
import math
from PIL import Image
from random import random
from scipy.stats import norm
import jax.numpy as np
import scipy
from jax import random
from jax import vmap
import pandas as pd
import matplotlib.pyplot as plt
import numpy 
from numpy.random import binomial
from pdb import set_trace
import os
from parameters import *
#numpy.random.seed(0)

 
#####  ORIGINAL UTIL ####

def Euler2fixedpt(dxdt, x_initial, Tmax, dt, xtol=1e-5, xmin=1e-0, Tmin=200, PLOT=True, save= None, inds=None, verbose=True, silent=False, print_dt = False):
    """
    Finds the fixed point of the D-dim ODE set dx/dt = dxdt(x), using the
    Euler update with sufficiently large dt (to gain in computational time).
    Checks for convergence to stop the updates early.

    IN:
    dxdt = a function handle giving the right hand side function of dynamical system
    x_initial = initial condition for state variables (a column vector)
    Tmax = maximum time to which it would run the Euler (same units as dt, e.g. ms)
    dt = time step of Euler
    xtol = tolerance in relative change in x for determining convergence
    xmin = for x(i)<xmin, it checks convergenece based on absolute change, which must be smaller than xtol*xmin
        Note that one can effectively make the convergence-check purely based on absolute,
        as opposed to relative, change in x, by setting xmin to some very large
        value and inputting a value for 'xtol' equal to xtol_desired/xmin.
    PLOT: if True, plot the convergence of some component
    inds: indices of x (state-vector) to plot

    OUT:
    xvec = found fixed point solution
    CONVG = True if determined converged, False if not
    """

    
    if PLOT==True:
        if inds is None:
            N = x_initial.shape[0] # x_initial.size
            inds = [int(N/4), int(3*N/4)]
            
        #xplot = x_initial[inds][:,None]
        xplot = x_initial[np.array(inds)][:,None]
        xplot_all = np.sum(x_initial)
        xplot_max=[]
        xplot_max.append(x_initial.max())
    
    Nmax = np.round(Tmax/dt).astype(int)
    Nmin = np.round(Tmin/dt) if Tmax > Tmin else (Nmax/2)
    xvec = x_initial 
    CONVG = False
    
    for n in range(Nmax):
        
        dx = dxdt(xvec) * dt
        
        xvec = xvec + dx
        
        if PLOT:
            #xplot = np.asarray([xplot, xvvec[inds]])
            xplot = np.hstack((xplot, xvec[np.asarray(inds)][:,None]))
            xplot_all=np.hstack((xplot_all, np.sum(xvec)))
            xplot_max.append(xvec.max())
            
        
        if n > Nmin:
            if np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max() < xtol: # y
                if verbose:
                    print("      converged to fixed point at iter={},      as max(abs(dx./max(xvec,{}))) < {} ".format(n, xmin, xtol))
                CONVG = True
                break

    if not CONVG and not silent: # n == Nmax:
        print("\n Warning 1: reached Tmax={}, before convergence to fixed point.".format(Tmax))
        print("       max(abs(dx./max(abs(xvec), {}))) = {},   xtol={}.\n".format(xmin, np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max(), xtol))
        #mybeep(.2,350)
        #beep

    if PLOT==True:
        print('plotting')

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20,5))
        
        axes[0].plot(np.arange(n+2)*dt, xplot.T, 'o-', label=inds)
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Neural responses')
        axes[0].legend()
        
        
        axes[1].plot(np.arange(n+2)*dt, xplot_all)
        axes[1].set_ylabel('Sum of response')
        axes[1].set_xlabel('Steps')
        axes[1].set_ylim([0, 1.2*np.max(np.asarray(xplot_all[-100:]))])
        axes[1].set_title('Final sum: '+str(np.sum(xvec))+', converged '+str(CONVG))
        
        axes[2].plot(np.arange(n+2)*dt, np.asarray(xplot_max))
        axes[2].set_ylabel('Maximum response')
        axes[2].set_title('final maximum: '+str(xvec.max())+'at index '+str(np.argmax(xvec)))
        axes[2].set_xlabel('Steps')
        axes[2].set_ylim([0, 1.2*np.max(np.asarray(xplot_max[-100:]))])
        
        if save:
            fig.savefig(save+'.png')
        
        
        fig.show()
        plt.close()
        
                                                      
    print(xvec.max(), np.argmax(xvec))
    return xvec, CONVG

####### TRAINING SUPPLMENTARY FUNCTIONS ######

def take_log(J_2x2):
    
    signs=np.array([[1, -1], [1, -1]])
    logJ_2x2 =np.log(J_2x2*signs)
    
    return logJ_2x2

def sep_exponentiate(J_s):
    signs=np.array([[1, -1], [1, -1]]) 
    new_J =np.exp(J_s)*signs

    return new_J


def homeo_loss(r_mean, r_max, R_mean_const, R_max_const, lambda_mean = 1):
    
    return np.maximum(0, (r_max/R_max_const) -1) + lambda_mean*(((r_mean / R_mean_const) -1)**2)    


def leaky_relu(r, R_thresh, slope_1 = 0.15, slope_2 = 1/50):
    
    height = slope_1/2
    constant = height/(R_thresh**2)
    return jax.lax.cond((r<R_thresh), r_less_than, r_greater_than, r, constant, slope_2, height)
 
def r_greater_than(r, constant, slope_2, height):
    return r*slope_2 - (1-height)

def r_less_than(r, constant, slope_2, height):
    return constant*(r**2) 


def rates_loss(numerator, denominator):
    
    return ((numerator / denominator) -1)**2


def constant_to_vec(c_E, c_I, ssn, sup = False):
    
    edge_length = ssn.grid_pars.gridsize_Nx

    matrix_E = np.ones((edge_length, edge_length)) * c_E
    vec_E = np.ravel(matrix_E)
    
    matrix_I = np.ones((edge_length, edge_length))* c_I
    vec_I = np.ravel(matrix_I)
    
    constant_vec = np.hstack((vec_E, vec_I, vec_E, vec_I))
  
    if sup==False and ssn.phases ==4:
        constant_vec = np.kron(np.asarray([1,1]), constant_vec)

    if sup:
        constant_vec = np.hstack((vec_E, vec_I))
        
    return constant_vec


def sigmoid(x, epsilon = 0.01):
    '''
    Introduction of epsilon stops asymptote from reaching 1 (avoids NaN)
    '''
    return (1 - 2*epsilon)*sig(x) + epsilon


def sig(x):
    return 1/(1+np.exp(-x))


def f_sigmoid(x, a = 0.75):
    return (1.25-a) + 2*a*sig(x)

def binary_loss(n, x):
    return - (n*np.log(x) + (1-n)*np.log(1-x))


def save_params_dict_two_stage(ssn_layer_pars, readout_pars, true_acc, epoch ):
    
    '''
    Assemble trained parameters and epoch information into single dictionary for saving
    Inputs:
        dictionaries containing trained parameters
        other epoch parameters (accuracy, epoch number)
    Outputs:
        single dictionary concatenating all information to be saved
    '''
    
    
    save_params = {}
    save_params= dict(epoch = epoch, val_accuracy= true_acc)
    
    if 'J_2x2_m' in ssn_layer_pars.keys():
        J_2x2_m = sep_exponentiate(ssn_layer_pars['J_2x2_m'])
        Jm = dict(J_EE_m= J_2x2_m[0,0], J_EI_m = J_2x2_m[0,1], 
                                J_IE_m = J_2x2_m[1,0], J_II_m = J_2x2_m[1,1])
        save_params.update(Jm)
            
    
    if 'J_2x2_s' in ssn_layer_pars.keys():
        J_2x2_s = sep_exponentiate(ssn_layer_pars['J_2x2_s'])
        Js = dict(J_EE_s= J_2x2_s[0,0], J_EI_s = J_2x2_s[0,1], 
                              J_IE_s = J_2x2_s[1,0], J_II_s = J_2x2_s[1,1])
            
    
        save_params.update(Js)
    
    if 'c_E' in ssn_layer_pars.keys():
        save_params['c_E'] = ssn_layer_pars['c_E']
        save_params['c_I'] = ssn_layer_pars['c_I']

   
    if 'sigma_oris' in ssn_layer_pars.keys():

        if len(ssn_layer_pars['sigma_oris']) ==1:
            save_params['sigma_oris'] = np.exp(ssn_layer_pars['sigma_oris'])
        elif np.shape(ssn_layer_pars['sigma_oris'])==(2,2):
            save_params['sigma_orisEE'] = np.exp(ssn_layer_pars['sigma_oris'][0,0])
            save_params['sigma_orisEI'] = np.exp(ssn_layer_pars['sigma_oris'][0,1])
        else:
            sigma_oris = dict(sigma_orisE = np.exp(ssn_layer_pars['sigma_oris'][0]), sigma_orisI = np.exp(ssn_layer_pars['sigma_oris'][1]))
            save_params.update(sigma_oris)
      
    if 'kappa_pre' in ssn_layer_pars.keys():
        if np.shape(ssn_layer_pars['kappa_pre']) == (2,2):
            save_params['kappa_preEE'] = np.tanh(ssn_layer_pars['kappa_pre'][0,0])
            save_params['kappa_preEI'] = np.tanh(ssn_layer_pars['kappa_pre'][0,1])
            save_params['kappa_postEE'] = np.tanh(ssn_layer_pars['kappa_post'][0,0])
            save_params['kappa_postEI'] = np.tanh(ssn_layer_pars['kappa_post'][0,1])


        else:
            save_params['kappa_preE'] = np.tanh(ssn_layer_pars['kappa_pre'][0])
            save_params['kappa_preI'] = np.tanh(ssn_layer_pars['kappa_pre'][1])
            save_params['kappa_postE'] = np.tanh(ssn_layer_pars['kappa_post'][0])
            save_params['kappa_postI'] = np.tanh(ssn_layer_pars['kappa_post'][1])
    
    if 'f_E' in ssn_layer_pars.keys():

        save_params['f_E'] = np.exp(ssn_layer_pars['f_E'])#*f_sigmoid(ssn_layer_pars['f_E'])
        save_params['f_I'] = np.exp(ssn_layer_pars['f_I'])
        
    #Add readout parameters
    save_params.update(readout_pars)

    return save_params







make_J2x2_o = lambda Jee, Jei, Jie, Jii: np.array([[Jee, -Jei], [Jie,  -Jii]])

def init_set_func(init_set, conn_pars, ssn_pars, middle=False):
    
    
    #ORIGINAL TRAINING!!
    if init_set ==0:
        Js0 = [1.82650658, 0.68194475, 2.06815311, 0.5106321]
        gE, gI = 0.57328625, 0.26144141
        sigEE, sigIE = 0.2, 0.40
        sigEI, sigII = .09, .09
        conn_pars.p_local = [0.4, 0.7]

    if init_set ==1:
        Js0 = [1.82650658, 0.68194475, 2.06815311, 0.5106321]
        gE, gI = 0.37328625*1.5, 0.26144141*1.5
        sigEE, sigIE = 0.2, 0.40
        sigEI, sigII = .09, .09
        conn_pars.p_local = [0.4, 0.7]

    if init_set==2:
        Js0 = [1.72881688, 1.29887564, 1.48514091, 0.76417991]
        gE, gI = 0.5821754, 0.22660373
        sigEE, sigIE = 0.225, 0.242
        sigEI, sigII = .09, .09
        conn_pars.p_local = [0.0, 0.0]
    
    if init_set ==3:
        Js0 = [1.82650658, 0.68194475, 2.06815311, 0.5106321]
        gE, gI = 1,1
        sigEE, sigIE = 0.2, 0.40
        sigEI, sigII = .09, .09
        conn_pars.p_local = [0.4, 0.7]
        
    if init_set=='A':
        Js0 = [2.5, 1.3, 2.4, 1.0]
        gE, gI =  0.4, 0.4
        print(gE, gI)
        sigEE, sigIE = 0.2, 0.40
        sigEI, sigII = .09, .09
        conn_pars.p_local = [0.4, 0.7]
        
    if init_set=='C':
        Js0 = [2.5, 1.3, 4.7, 2.2]
        gE, gI =0.3, 0.25
        sigEE, sigIE = 0.2, 0.40
        sigEI, sigII = .09, .09
        conn_pars.p_local = [0.4, 0.7]
        
    if middle:
        conn_pars.p_local = [1, 1]
        
    if init_set =='C':
        make_J2x2 = lambda Jee, Jei, Jie, Jii: np.array([[Jee, -Jei], [Jie,  -Jii]])  * ssn_pars.psi
    else:
        make_J2x2 = lambda Jee, Jei, Jie, Jii: np.array([[Jee, -Jei], [Jie,  -Jii]]) * np.pi * ssn_pars.psi
        
    J_2x2 = make_J2x2(*Js0)
    s_2x2 = np.array([[sigEE, sigEI],[sigIE, sigII]])
    
    return J_2x2, s_2x2, gE, gI, conn_pars

    
    

#### CREATE GABOR FILTERS ####
class GaborFilter:
    
    def __init__(self, x_i, y_i, k, sigma_g, theta, edge_deg, degree_per_pixel, phase=0, conv_factor=None):
        
        '''
        Gabor filter class.
        Inputs:
            x_i, y_i: centre of filter
            k: preferred spatial frequency in cycles/degrees (radians)
            sigma_g: variance of Gaussian function
            theta: preferred oritnation 
            conv_factor: conversion factor from degrees to mm
        '''
        
        #convert to mm from degrees
        if conv_factor:
            self.conv_factor = conv_factor
            self.x_i=x_i/conv_factor
            self.y_i=y_i/conv_factor
        else:
            self.x_i=x_i
            self.y_i=y_i
        self.k=k 
        self.theta=theta*(np.pi/180) 
        self.phase=phase 
        self.sigma_g=sigma_g
        self.edge_deg = edge_deg
        self.degree_per_pixel = degree_per_pixel
        self.N_pixels=int(edge_deg*2/degree_per_pixel) +1 
        
        
        #create image axis
        x_axis=np.linspace(-edge_deg, edge_deg, self.N_pixels, endpoint=True)  
        y_axis=np.linspace(-edge_deg, edge_deg, self.N_pixels, endpoint=True)
        
        #construct filter as attribute
        self.filter = self.create_filter(x_axis, y_axis)

    
    def create_filter(self, x_axis, y_axis):
        '''
        Create Gabor filters in vectorised form. 
        '''
        x_axis=np.reshape(x_axis, (self.N_pixels, 1))
        #self.theta=np.pi/2 - self.theta
       
        x_i=np.repeat(self.x_i, self.N_pixels)
        x_i=np.reshape(x_i, (self.N_pixels, 1))
        diff_x= (x_axis.T - x_i)

        y_axis=np.reshape(y_axis, (self.N_pixels, 1))

        y_i=np.repeat(self.y_i, self.N_pixels)
        y_i=np.reshape(y_i, (self.N_pixels, 1))
        diff_y=((y_axis - y_i.T))
        
        spatial=np.cos(self.k*np.pi*2*(diff_x*np.cos(self.theta) + diff_y*np.sin(self.theta)) + self.phase) 
        gaussian= np.exp(-0.5 *( diff_x**2 + diff_y**2)/self.sigma_g**2)
        
        return gaussian*spatial[::-1] #same convention as stimuli
    
    
### FINDING CONSTANT FOR GABOR FILTERS ###
def find_A(
    k,
    sigma_g,
    edge_deg,
    degree_per_pixel,
    indices,
    phase=0,
    return_all=False,
):
    """
    Find constant to multiply Gabor filters.
    Input:
        gabor_pars: Filter parameters - centre already specified in function
        stimuli_pars: Stimuli parameters (high constrast and spanning all visual field)
        indices: List of orientatins in degrees to calculate filter and corresponding stimuli
    Output:
        A: value of constant so that contrast = 100
    """
    all_A = []
    all_stimuli_mean =[]
    all_stimuli_max = []
    all_stimuli_min = []

    for ori in indices:
        # generate Gabor filter and stimuli at orientation
        gabor = GaborFilter(
            theta=ori,
            x_i=0,
            y_i=0,
            edge_deg=edge_deg,
            k=k,
            sigma_g=sigma_g,
            degree_per_pixel=degree_per_pixel,
            phase=phase,
        )
        # create local_stimui_pars to pass it to the BW_Gratings
        local_stimuli_pars = StimuliPars()
        local_stimuli_pars.edge_deg=edge_deg
        local_stimuli_pars.k=k
        local_stimuli_pars.outer_radius=edge_deg * 2
        local_stimuli_pars.inner_radius=edge_deg * 2
        local_stimuli_pars.degree_per_pixel=degree_per_pixel
        local_stimuli_pars.grating_contrast=0.99
        local_stimuli_pars.jitter = 0
        local_stimuli_pars.std = 0
        
        #Create test grating
        test_grating = BW_Grating(
            ori_deg=ori,
            jitter = local_stimuli_pars.jitter,    
            stimuli_pars=local_stimuli_pars,
            phase=phase,
        )
        test_stimuli = test_grating.BW_image()
        mean_removed_filter = gabor.filter - gabor.filter.mean()
        
        all_stimuli_mean.append(test_stimuli.mean())
        all_stimuli_min.append(test_stimuli.min())
        all_stimuli_max.append(test_stimuli.max())
        
        # multiply filter and stimuli
        output_gabor = mean_removed_filter.ravel() @ test_stimuli.ravel()

        # calculate value of A
        A_value = 100 / (output_gabor)

        # create list of A
        all_A.append(A_value)

    # find average value of A
    all_A = np.array(all_A)
    A = all_A.mean()
                
    if return_all == True:
        output = A, all_gabors, all_test_stimuli
    else:
        output = A

    return output


#rng = numpy.random.default_rng(12345)
#CREATE INPUT STIMULI
def create_grating_pairs(n_trials, stimuli_pars):
    '''
    Create input stimuli gratings. Both the refence and the target are jitted by the same angle. 
    Input:
       stimuli pars
       n_trials - batch size
    
    Output:
        dictionary containing reference target and label 
    
    '''
    #initialise empty arrays
    training_gratings=[]
    ref_ori = stimuli_pars.ref_ori
    offset = stimuli_pars.offset
    
    data_dict = {'ref':[], 'target': [], 'label':[]}
    for i in range(n_trials):
        #uniform_dist_value = rng.uniform(low = 0, high = 1)
        if numpy.random.uniform(0,1,1) < 0.5:
        #if  uniform_dist_value < 0.5:
            target_ori = ref_ori - offset
            label = 1
        else:
            target_ori = ref_ori + offset
            label = 0
        jitter_val = stimuli_pars.jitter_val
        jitter = numpy.random.uniform(-jitter_val, jitter_val, 1)
        #jitter = rng.uniform(low = -jitter_val, high = jitter_val)
        
        #create reference grating
        ref = BW_Grating(ori_deg = ref_ori, jitter=jitter, stimuli_pars = stimuli_pars).BW_image().ravel()

        #create target grating
        target = BW_Grating(ori_deg = target_ori, jitter=jitter, stimuli_pars = stimuli_pars).BW_image().ravel()
        
        data_dict['ref'].append(ref)
        data_dict['target'].append(target)
        data_dict['label'].append(label)
        #data_dict = {'ref':ref, 'target': target, 'label':label}
        
    data_dict['ref'] = np.asarray(data_dict['ref'])
    data_dict['target'] = np.asarray(data_dict['target'])
    data_dict['label'] = np.asarray(data_dict['label'])

    return data_dict



def load_param_from_csv(results_filename, epoch):
    
    '''
    Load parameters from csv file given file name and desired epoch.
    '''
    
    all_results = pd.read_csv(results_filename, header = 0)
    if epoch == -1:
        epoch_params = all_results.tail(1)
    else:
        epoch_params = all_results.loc[all_results['epoch'] == epoch]
    params = []
    J_m = [np.abs(epoch_params[i].values[0]) for i in ['J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m']]
    J_s = [np.abs(epoch_params[i].values[0]) for i in ['J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s']]


    J_2x2_m = make_J2x2_o(*J_m)
    J_2x2_s = make_J2x2_o(*J_s)
    params.append(J_2x2_m)
    params.append(J_2x2_s)

    
    if 'c_E' in all_results.columns:
        c_E = epoch_params['c_E'].values[0]
        c_I = epoch_params['c_I'].values[0]
        params.append(c_E)
        params.append(c_I)
    
    if 'sigma_orisE' in all_results.columns:
        sigma_oris = np.asarray([epoch_params['sigma_orisE'].values[0], epoch_params['sigma_orisI'].values[0]])
        params.append(sigma_oris)
    
    if 'f_E' in all_results.columns:
        f_E = epoch_params['f_E'].values[0]
        f_I = epoch_params['f_I'].values[0]
        params.append(f_E)
        params.append(f_I)
    
    if 'kappa_preE' in all_results.columns:
        kappa_pre = np.asarray([epoch_params['kappa_preE'].values[0], epoch_params['kappa_preI'].values[0]])
        kappa_post = np.asarray([epoch_params['kappa_postE'].values[0], epoch_params['kappa_postI'].values[0]])
        params.append(kappa_pre)
        params.append(kappa_post)
        
    return params


def create_grating_single(stimuli_pars, n_trials = 10):

    all_stimuli = []
    jitter_val = stimuli_pars.jitter_val
    ref_ori = stimuli_pars.ref_ori

    for i in range(0, n_trials):
        jitter = numpy.random.uniform(-jitter_val, jitter_val, 1)

        #create reference grating
        ref = BW_Grating(ori_deg = ref_ori, jitter=jitter, stimuli_pars = stimuli_pars).BW_image().ravel()
        all_stimuli.append(ref)
    
    return np.vstack([all_stimuli])


        
def save_matrices(run_dir, matrix_sup, matrix_mid):
    np.save(os.path.join(run_dir+'_sup.npy'), matrix_sup) 
    np.save(os.path.join(run_dir+'_mid.npy'), matrix_mid) 
    
    
def load_matrix_response(results_dir, layer): 
    run_dir = os.path.join(results_dir, 'response_matrix_')
    
    response_matrix_contrast_02 = np.load(run_dir+'0.2'+str(layer)+'.npy')
    response_matrix_contrast_04= np.load(run_dir+'0.4'+str(layer)+'.npy')
    response_matrix_contrast_06 = np.load(run_dir+'0.6'+str(layer)+'.npy')
    response_matrix_contrast_08 = np.load(run_dir+'0.8'+str(layer)+'.npy')
    response_matrix_contrast_099 = np.load(run_dir+'0.99'+str(layer)+'.npy')
    
    return response_matrix_contrast_02, response_matrix_contrast_04, response_matrix_contrast_06, response_matrix_contrast_08, response_matrix_contrast_099


class BW_Grating:
    """ """

    def __init__(
        self,
        ori_deg,
        stimuli_pars,
        jitter=0,
        phase=0,
        crop_f=None,
    ):
        self.ori_deg = ori_deg
        self.jitter = jitter
        self.outer_radius = stimuli_pars.outer_radius  # in degrees
        self.inner_radius = stimuli_pars.inner_radius  # in degrees
        self.grating_contrast = stimuli_pars.grating_contrast
        self.std = stimuli_pars.std
        degree_per_pixel = stimuli_pars.degree_per_pixel
        pixel_per_degree = 1 / degree_per_pixel
        self.pixel_per_degree = pixel_per_degree
        edge_deg = stimuli_pars.edge_deg
        size = int(edge_deg * 2 * pixel_per_degree) + 1
        self.size = size
        k = stimuli_pars.k
        spatial_frequency = k * degree_per_pixel  # 0.05235987755982988
        self.phase = phase
        self.crop_f = crop_f
        self.smooth_sd = self.pixel_per_degree / 6
        self.spatial_freq = spatial_frequency or (1 / self.pixel_per_degree)
        self.grating_size = round(self.outer_radius * self.pixel_per_degree)
        self.angle = ((self.ori_deg + self.jitter) - 90) / 180 * numpy.pi
        
    def BW_image(self):
        _BLACK = 0
        _WHITE = 255
        _GRAY = round((_WHITE + _BLACK) / 2)

        # Generate a 2D grid of coordinates
        x, y = numpy.mgrid[
            -self.grating_size : self.grating_size + 1.0,
            -self.grating_size : self.grating_size + 1.0,
        ]

        # Calculate the distance from the center for each pixel
        edge_control_dist = numpy.sqrt(numpy.power(x, 2) + numpy.power(y, 2))
        edge_control = numpy.divide(edge_control_dist, self.pixel_per_degree)

        # Create a matrix (alpha_channel) that is 255 (white) within the inner_radius and exponentially fades to 0 as the radius increases
        overrado = numpy.nonzero(edge_control > self.inner_radius)
        d = self.grating_size * 2 + 1
        annulus = numpy.ones((d, d))

        annulus[overrado] *= numpy.exp(
            -1
            * ((edge_control[overrado] - self.inner_radius) * self.pixel_per_degree)
            ** 2
            / (2 * (self.smooth_sd**2))
        )
        alpha_channel = annulus * _WHITE

        # Generate the grating pattern, which is a centered and tilted sinusoidal matrix
        spatial_component = (
            2
            * math.pi
            * self.spatial_freq
            * (y * numpy.sin(self.angle) + x * numpy.cos(self.angle))
        )
        gabor_sti = _GRAY * (
            1 + self.grating_contrast * numpy.cos(spatial_component + self.phase)
        )

        # Set pixels outside the grating size to gray
        gabor_sti[edge_control_dist > self.grating_size] = _GRAY
        
        # Add Gaussian white noise to the grating
        #noise = numpy.random.normal(loc=0, scale=self.std, size=gabor_sti.shape)
        #noisy_gabor_sti = gabor_sti + noise

        # Expand the grating to have three colors andconcatenate it with alpha_channel
        gabor_sti_final = numpy.repeat(gabor_sti[:, :, numpy.newaxis], 3, axis=-1)
        gabor_sti_final_with_alpha = numpy.concatenate(
            (gabor_sti_final, alpha_channel[:, :, numpy.newaxis]), axis=-1
        )
        gabor_sti_final_with_alpha_image = Image.fromarray(
            gabor_sti_final_with_alpha.astype(numpy.uint8)
        )

        # Create a background image filled with gray
        background = numpy.full((self.size, self.size, 3), _GRAY, dtype=numpy.uint8)
        final_image = Image.fromarray(background)

        # Paste the grating into the final image: paste the grating into a bounding box and apply the alpha channel as a mask
        center_x, center_y = self.size // 2, self.size // 2
        bounding_box = (center_x - self.grating_size, center_y - self.grating_size)
        final_image.paste(
            gabor_sti_final_with_alpha_image,
            box=bounding_box,
            mask=gabor_sti_final_with_alpha_image,
        )

        # Sum the image over color channels
        final_image_np = numpy.array(final_image, dtype=numpy.float16)
        image = numpy.sum(final_image_np, axis=2)

        # Crop the image if crop_f is specified
        if self.crop_f:
            image = image[self.crop_f : -self.crop_f, self.crop_f : -self.crop_f]
    
        noise = numpy.random.normal(loc=0, scale=self.std, size=image.shape)
        #print('noise std ', np.std(noise))
        image = image + noise
        
        return image


def smooth_data(vector, sigma = 1):

    '''
    Smooth fixed point. Data is reshaped into 9x9 grid
    '''
    
    new_data = []
    for trial_response in vector:

        trial_response = trial_response.reshape(9,9,-1)
        smoothed_data = numpy.asarray([ndimage.gaussian_filter(numpy.reshape(trial_response[:, :, i], (9,9)), sigma = sigma) for i in range(0, trial_response.shape[2])]).ravel()
        new_data.append(smoothed_data)
    
    return np.vstack(np.asarray(new_data))  

def select_neurons(fp, layer):
    if layer=='mid':
        
        E_indices = np.linspace(0, 647, 648).round().reshape(8, 81, -1)[0:9:2].ravel().astype(int)
        I_indices =np.linspace(0, 647, 648).round().reshape(8, 81, -1)[1:9:2].ravel().astype(int)
                                                                                         
    if layer =='sup':                                                                               
        E_indices = np.linspace(0, 80, 81).astype(int)
        I_indices = np.linspace(81, 161, 81).astype(int)
                                                                       
    return np.asarray(fp[E_indices]), np.asarray(fp[I_indices])    
    
def my_mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.T)
    inv_covmat = scipy.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()


from scipy import ndimage
def pre_process(data, n_trials = 300):
    
    #Select E neurons
    data_cropped = data[:, :81]
    
    #Smooth data
    data_smooth = numpy.asarray([ndimage.gaussian_filter(numpy.reshape(i, (9,9)), sigma = 1) for i in data_cropped])

    #Standarise 
    data_norm = standardize_data(numpy.reshape(data_smooth, (n_trials, -1)))
    
    return data_norm