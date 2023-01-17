import jax
from jax import random
import math
from PIL import Image
from random import random
from scipy.stats import norm
import jax.numpy as np
from jax import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy 
from numpy.random import binomial



#####  ORIGINAL UTIL ####

def Euler2fixedpt(dxdt, x_initial, Tmax, dt, xtol=1e-5, xmin=1e-0, Tmin=200, PLOT=False, inds=None, verbose=True, silent=False):
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

    if PLOT:
        if inds is None:
            N = x_initial.shape[0] # x_initial.size
            inds = [int(N/4), int(3*N/4)]
        xplot = x_initial[inds][:,None]
    Nmax = int(np.round(Tmax/dt))
    Nmin = int(np.round(Tmin/dt)) if Tmax > Tmin else (Nmax/2)
    xvec = x_initial 
    CONVG = False
    
    for n in range(Nmax):
        dx = dxdt(xvec) * dt
        xvec = xvec + dx
        
        if PLOT:
            #xplot = np.asarray([xplot, xvvec[inds]])
            xplot = np.hstack((xplot,xvec[inds][:,None]))
        
        
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

    if PLOT:
        import matplotlib.pyplot as plt
        plt.figure(244459)
        plt.plot(np.arange(n+2)*dt, xplot.T, 'o-')

    return xvec, CONVG

### END OF ORIGINAL UTIL ###

def Euler2fixedpt_fullTmax(dxdt, x_initial, Tmax, dt, xtol=1e-5, xmin=1e-0, Tmin=200, PLOT=False, inds=None, verbose=True, silent=False):
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

    if PLOT:
        if inds is None:
            N = x_initial.shape[0] # x_initial.size
            inds = [int(N/4), int(3*N/4)]
        xplot = x_initial[inds][:,None]
    Nmax = int(np.round(Tmax/dt))
    Nmin = int(np.round(Tmin/dt)) if Tmax > Tmin else (Nmax/2)
    xvec = x_initial 
    CONVG = False
    y = []
    
    for n in range(Nmax):
        dx = dxdt(xvec) * dt
        xvec = xvec + dx
        y.append(np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max())
    
    y = np.asarray(y)  
    avg_dx = y[int(Nmax/2):int(Nmax)].mean()/xtol
    CONVG = np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max() < xtol
    return xvec, CONVG, avg_dx

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
    
    
    def find_A(self, indices, return_all=False):
        '''
        Find constant to multiply Gabor filters.
        Input:
            gabor_pars: Filter parameters - centre already specified in function
            stimuli_pars: Stimuli parameters (high constrast and spanning all visual field)
            indices: List of orientatins in degrees to calculate filter and corresponding stimuli
        Output:
            A: value of constant so that contrast = 100
        '''
        all_A=[]
        all_gabors=[]
        all_test_stimuli=[]

        for ori in indices:

            #generate Gabor filter and stimuli at orientation
            gabor=GaborFilter(theta=ori, x_i=0, y_i=0, edge_deg=self.edge_deg, k=self.k, sigma_g=self.sigma_g, degree_per_pixel=self.degree_per_pixel)
            test_grating=BW_Grating(ori_deg=ori, edge_deg=self.edge_deg, k=self.k, degree_per_pixel=self.degree_per_pixel, outer_radius=self.edge_deg*2, inner_radius=self.edge_deg*2, grating_contrast=0.99)
            test_stimuli=test_grating.BW_image()

            #multiply filter and stimuli
            output_gabor=np.matmul(gabor.filter.ravel(), test_stimuli.ravel())


            all_gabors.append(gabor.filter)
            all_test_stimuli.append(test_stimuli)


            #calculate value of A
            A_value=100/(output_gabor) 

            #create list of A
            all_A.append(A_value)


        #find average value of A
        all_A=np.array(all_A)
        A=all_A.mean()

        all_gabors=np.array(all_gabors)
        all_test_stimuli=np.array(all_test_stimuli)

        #print('Average A is {}'.format(A))

        if return_all==True:
            output =  A , all_gabors, all_test_stimuli
        else:
            output=A

        return output

    
    
##### CREATING GRATINGS ######    
"""
Author: Samuel Bell (sjb326@cam.ac.uk) 
jia_grating.py Copyright (c) 2020. All rights reserved.
This file may not be used or modified without the author's explicit written permission.
These files are hosted at https://gitlab.com/samueljamesbell/vpl-modelling. All other locations are mirrors of the original repository.

This file is a Python port of stimuli developed by Ke Jia.
"""

_BLACK = 0
_WHITE = 255
_GRAY = round((_WHITE + _BLACK) / 2)


class JiaGrating:

    def __init__(self, ori_deg, size, outer_radius, inner_radius, pixel_per_degree, grating_contrast, phase, jitter, std = 0, spatial_frequency=None, ):
        self.ori_deg = ori_deg
        self.size = size

        self.outer_radius = outer_radius #in degrees
        self.inner_radius = inner_radius #in degrees
        self.pixel_per_degree = pixel_per_degree
        self.grating_contrast = grating_contrast
        self.phase = phase
        self.jitter =  jitter
        self.std = std

        self.smooth_sd = self.pixel_per_degree / 6
        self.spatial_freq = spatial_frequency or (1 / self.pixel_per_degree)
        self.grating_size = round(self.outer_radius * self.pixel_per_degree)
        self.angle = ((self.ori_deg + self.jitter) - 90) / 180 * math.pi

    def image(self):
        x, y = numpy.mgrid[-self.grating_size:self.grating_size+1., -self.grating_size:self.grating_size+1.]

        d = self.grating_size * 2 + 1
        annulus = numpy.ones((d, d))

        edge_control = numpy.divide(numpy.sqrt(numpy.power(x, 2) + numpy.power(y, 2)), self.pixel_per_degree)

        overrado = numpy.nonzero(edge_control > self.inner_radius)

        for idx_x, idx_y in zip(*overrado):
            annulus[idx_x, idx_y] = annulus[idx_x, idx_y] * numpy.exp(-1 * ((((edge_control[idx_x, idx_y] - self.inner_radius) * self.pixel_per_degree) ** 2) / (2 * (self.smooth_sd ** 2))))    

        gabor_sti = _GRAY * (1 + self.grating_contrast * numpy.cos(2 * math.pi * self.spatial_freq * (y * numpy.sin(self.angle) + x * numpy.cos(self.angle)) + self.phase))

        gabor_sti[numpy.sqrt(numpy.power(x, 2) + numpy.power(y, 2)) > self.grating_size] = _GRAY
        
        #New noise - Gaussian white noise
        noise = numpy.random.normal(loc=0, scale=self.std, size = (d,d))
        noisy_gabor_sti = gabor_sti + noise

        # Original Noise
        #noise = numpy.floor(numpy.sin(norm.rvs(size=(d, d))) * _GRAY) + _GRAY
        
        #noise_mask = binomial(1, 1-self.snr, size=(d, d)).astype(int)
        
       #masked_noise = noise * noise_mask

        #signal_mask = 1 - noise_mask
        #masked_gabor_sti = signal_mask * gabor_sti

        #noisy_gabor_sti = masked_gabor_sti + masked_noise
        # End noise

        gabor_sti_final = numpy.repeat(noisy_gabor_sti[:, :, numpy.newaxis], 3, axis=-1)
        alpha_channel = annulus * _WHITE
        gabor_sti_final_with_alpha = numpy.concatenate((gabor_sti_final, alpha_channel[:, :, numpy.newaxis]), axis=-1)
        gabor_sti_final_with_alpha_image = Image.fromarray(gabor_sti_final_with_alpha.astype(numpy.uint8))

        center_x = int(self.size / 2)
        center_y = int(self.size / 2)
        bounding_box = (center_x - self.grating_size, center_y - self.grating_size)

        background = numpy.full((self.size, self.size, 3), _GRAY, dtype=numpy.uint8)
        final_image = Image.fromarray(background)

        final_image.paste(gabor_sti_final_with_alpha_image, box=bounding_box, mask=gabor_sti_final_with_alpha_image)
        #print(numpy.mean(noisy_gabor_sti) / numpy.std(noisy_gabor_sti))

        return final_image


class BW_Grating(JiaGrating):
    '''
    Sub-class of Jia Grating.
    Sums stimuli over channels and option to crop stimulus field. 
    '''
    
    def __init__(self, ori_deg, outer_radius, inner_radius, degree_per_pixel, grating_contrast, edge_deg, phase=0, jitter=0, std = 0, k=None, crop_f=None):
        
        self.crop_f=crop_f
        pixel_per_degree=1/degree_per_pixel
        size=int(edge_deg*2 *pixel_per_degree) + 1
        spatial_frequency = k*degree_per_pixel
        
        
                
        super().__init__( ori_deg, size, outer_radius, inner_radius, pixel_per_degree, grating_contrast, phase, jitter, std, spatial_frequency)
        
    def BW_image(self):
        
        #generate image using Jia Grating function
        original=numpy.array(self.image(), dtype=numpy.float16)
        
        #sum image over channels
        image=numpy.sum(original, axis=2) 
        
        #crop image
        if self.crop_f:
            image=image[self.crop_f:-self.crop_f, self.crop_f:-self.crop_f]            
        return image 


#CREATE INPUT STIMULI
def make_gratings(ref_ori, target_ori, jitter_val=5, **stimuli_pars, ):
    '''
    Create reference and target stimulus given orientations using same jitter (randomly generated per trial) for reference and target.
    '''
    #generate jitter for reference and target
    jitter = numpy.random.uniform(-jitter_val, jitter_val, 1)
    
    #create reference grating
    ref = BW_Grating(ori_deg = ref_ori, jitter=jitter, **stimuli_pars).BW_image().ravel()

    #create target grating
    target = BW_Grating(ori_deg = target_ori, jitter=jitter, **stimuli_pars).BW_image().ravel()
    
    return ref, target
    

    
def create_gratings(ref_ori, number, offset, jitter_val, **stimuli_pars):
    '''
    Create input stimuli gratings.
    Input:
        training_data: list of orientations, where each item of the list is [ref_ori, target_ori]. Length of list is number of trials
    Output:
        training_gratings: array of 1D reference and target stimuli. Shape is (n_trials, 2, n_pixels) - 2
    
    '''
    
    #initialise empty arrays
    labels_list=[]
    training_gratings=[]
   
    
    for i in range(number):
        
        if numpy.random.uniform(0,1,1) < 0.5:
            target_ori = ref_ori - offset
            label = 1
        else:
            target_ori = ref_ori + offset
            label = 0
        
        ref, target = make_gratings(ref_ori, target_ori, jitter_val,**stimuli_pars ) 

        data_dict = {'ref':ref, 'target': target, 'label':label}
        training_gratings.append(data_dict)

    return training_gratings





    
    
### FINDING CONSTANT FOR GABOR FILTERS ###
def find_A(conv_factor, k, sigma_g, edge_deg,  degree_per_pixel, indices, return_all=False):
    '''
    Find constant to multiply Gabor filters.
    Input:
        gabor_pars: Filter parameters - centre already specified in function
        stimuli_pars: Stimuli parameters (high constrast and spanning all visual field)
        indices: List of orientatins in degrees to calculate filter and corresponding stimuli
    Output:
        A: value of constant so that contrast = 100
    '''
    all_A=[]
    all_gabors=[]
    all_test_stimuli=[]
    
    for ori in indices:
    
        #generate Gabor filter and stimuli at orientation
        gabor=GaborFilter(theta=ori, x_i=0, y_i=0, edge_deg=edge_deg, k=k, sigma_g=sigma_g, degree_per_pixel=degree_per_pixel)
        test_grating=BW_Grating(ori_deg=ori, edge_deg=edge_deg, k=k, degree_per_pixel=degree_per_pixel, outer_radius=edge_deg*2, inner_radius=edge_deg*2, grating_contrast=0.99)
        test_stimuli=test_grating.BW_image()

        #multiply filter and stimuli
        output_gabor=np.matmul(gabor.filter.ravel(), test_stimuli.ravel())
        
        
        all_gabors.append(gabor.filter)
        all_test_stimuli.append(test_stimuli)


        #calculate value of A
        A_value=100/(output_gabor) 

        #create list of A
        all_A.append(A_value)
    
    
    #find average value of A
    all_A=np.array(all_A)
    A=all_A.mean()

    all_gabors=np.array(all_gabors)
    all_test_stimuli=np.array(all_test_stimuli)
    
    if return_all==True:
        output =  A , all_gabors, all_test_stimuli
    else:
        output=A
    
    return output



#CREATE FILTERS
def create_gabor_filters(ssn, conv_factor, k, sigma_g, edge_deg,  degree_per_pixel):
    
    e_filters=[] #array of filters

    #Iterate over SSN map
    for i in range(ssn.ori_map.shape[0]):
        for j in range(ssn.ori_map.shape[1]):
            gabor=GaborFilter(x_i=ssn.x_map[i,j], y_i=ssn.y_map[i,j], edge_deg=edge_deg, k=k, sigma_g=sigma_g, theta=ssn.ori_map[i,j], conv_factor=conv_factor, degree_per_pixel=degree_per_pixel)

            e_filters.append(gabor.filter.ravel())
    e_filters=np.array(e_filters)

    #create inhibitory filters
    i_constant= 1
    i_filters=np.multiply(i_constant, e_filters)
    all_filters=np.vstack([e_filters, i_filters]) #shape - (n_neurons, n_pixels in image(n_pixels_x_axis*n_pixels_y_axis))
    
    #create filters with phase equal to pi
    e_off_filters = - e_filters
    i_off_filters = - i_filters
    
    
    #SSN_filters=np.vstack([e_filters, i_filters, e_off_filters, i_off_filters])
    SSN_filters = np.vstack([e_filters, i_filters])
    
    A= find_A(return_all =False, conv_factor=conv_factor, k=k, sigma_g=sigma_g, edge_deg=edge_deg,  degree_per_pixel=degree_per_pixel, indices=np.sort(ssn.ori_map.ravel()))
    
    
    
    return SSN_filters, A



## PLOT INITIALIZATION HISTOGRAM
import numpy
def vmap_eval2(opt_pars, ssn_pars, grid_pars, conn_pars, test_data, filter_pars,  conv_pars):
    '''
    For a given value of the weights, calculate the loss for all the stimuli.
    Output:
        losses: size(n_stimuli)
        Accuracy: scalar
    '''
    
    eval_vmap = vmap(eval_model, in_axes = ({'b_sig': None, 'logJ_2x2': None, 'logs_2x2': None, 'w_sig': None, 'c_E':None, 'c_I':None}, None, None, {'PERIODIC': None, 'p_local': [None, None], 'sigma_oris': None},  {'ref':0, 'target':0, 'label':0}, {'conv_factor': None, 'degree_per_pixel': None, 'edge_deg': None, 'k': None, 'sigma_g': None}, {'Tmax': None, 'dt': None, 'silent': None, 'verbose': None, 'xtol': None}) )
    losses, pred_labels, dots = eval_vmap(opt_pars, ssn_pars, grid_pars, conn_pars, test_data, filter_pars,  conv_pars)
        
    accuracy = np.sum(test_data['label'] == pred_labels)/len(test_data['label']) 
    
    return losses, accuracy, dots

def vmap_eval3(opt_pars, ssn_pars, grid_pars, conn_pars, test_data, filter_pars,  conv_pars):
    '''
    Iterates through all values of 'w' to give the losses at each stimuli and weight, and the accuracy at each weight
    Output:
        losses: size(n_weights, n_stimuli )
        accuracy: size( n_weights)
    '''

    eval_vmap = vmap(vmap_eval2, in_axes = ({'b_sig': None, 'logJ_2x2': None, 'logs_2x2': None, 'w_sig': 0, 'c_E':None, 'c_I':None}, None, None, {'PERIODIC': None, 'p_local': [None, None], 'sigma_oris': None},  {'ref':None, 'target':None, 'label':None}, {'conv_factor': None, 'degree_per_pixel': None, 'edge_deg': None, 'k': None, 'sigma_g': None}, {'Tmax': None, 'dt': None, 'silent': None, 'verbose': None, 'xtol': None}) )
    losses, accuracies, dots = eval_vmap(opt_pars, ssn_pars, grid_pars, conn_pars, test_data, filter_pars,  conv_pars)
    print(losses.shape)
    print(accuracies.shape)
    return losses, accuracies
    
    
def test_accuracies(opt_pars, ssn_pars, grid_pars, conn_pars, filter_pars,  conv_pars, stimuli_pars, trials = 5, p = 0.9, printing=True):
    
    key = random.PRNGKey(7)
    N_neurons = 25
    accuracies = []
    key, _ = random.split(key)
    opt_pars['w_sig'] = random.normal(key, shape = (trials, N_neurons)) / np.sqrt(N_neurons)
    
    train_data = create_data(stimuli_pars)
    
    print(opt_pars['w_sig'].shape)
    val_loss, accuracies = vmap_eval3(opt_pars, ssn_pars, grid_pars, conn_pars, train_data, filter_pars,  conv_pars)
    
    #calcualate how many accuracies are above 90
    higher_90 = np.sum(accuracies[accuracies>p]) / len(accuracies)

    if printing:
        print('grating contrast = {}, jitter = {}, noise std={}, acc (% >90 ) = {}'.format(stimuli_pars['grating_contrast'], stimuli_pars['jitter_val'], stimuli_pars['std'], higher_90))
    return higher_90, accuracies


def initial_acc( opt_pars, ssn_pars, grid_pars, conn_pars, filter_pars,  conv_pars, stimuli_pars, jitter_max,  std_max, p = 0.9):
    '''
    Find initial accuracy for varying jitter and noise levels. 
    
    '''

    
    list_noise  = np.linspace(20, std_max, 5)
    list_jitters = np.linspace(0, jitter_max, 5)
   
    
    low_acc=[]
    all_accuracies=[]
    
    
    for noise in list_noise:
        for jitter in list_jitters:
            
            stimuli_pars['std'] = noise
            stimuli_pars['jitter_val'] = jitter
            higher_90, acc = test_accuracies(opt_pars, ssn_pars, grid_pars, conn_pars, filter_pars, conv_pars, stimuli_pars, p=p,  trials=100, printing=False)
            
            #save low accuracies
            if higher_90 < 0.05:
                low_acc.append([jitter, noise, higher_90])

            all_accuracies.append([jitter, noise, acc])
    
    plot_histograms(all_accuracies)
        
    
    return all_accuracies, low_acc


def plot_histograms(all_accuracies):
    
    #n_rows =  int(np.sqrt(len(all_accuracies)))
    #n_cols = int(np.ceil(len(all_accuracies) / n_rows))
    n_cols = 5
    n_rows = 5
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 20))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    count = 0

    
   #plot histograms
    for k in range(n_rows):
        for j in range (n_cols):
            axs[k,j].hist(all_accuracies[count][2])
            axs[k,j].set_xlabel('Initial accuracy')
            axs[k,j].set_ylabel('Frequency')
            axs[k,j].set_title('std = '+str(np.round(all_accuracies[count][1], 2))+ ' jitter = '+str(np.round(all_accuracies[count][0], 2)), fontsize=10)
            count+=1
            if count==len(all_accuracies):
                break
    
    fig.show()
    
###ANALYSE RESULTS    
def param_ratios(results_file):
    results = pd.read_csv(results_file, header = 0)
    res = results.to_numpy()
    Js=res[:,1:5]
    ss = res[:,5:9]
    #print(Js.type())
    print(results.columns[1:5])
    print("J ratios = ", np.array((Js[-1,:]/Js[0,:] -1)*100, dtype=int))
    print(results.columns[5:9])
    print(ss[-1,:]/ss[0,:])
    print("s ratios = ", np.array((ss[-1,:]/ss[0,:] -1)*100, dtype=int))
    

def plot_results(results_file, title=None):
    
    results = pd.read_csv(results_file, header = 0)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,8))

    results.plot(x='epoch', y=["J_EE", "J_EI", "J_IE", "J_II"], ax=axes[0,0])
    results.plot(x='epoch', y=["s_EE", "s_EI", "s_IE", "s_II"], ax = axes[0,1])
    results.plot(x='epoch', y=["c_E", "c_I"], ax = axes[1,0])
    results.plot(x='epoch', y = 'val_accuracy', ax = axes[1,1])
    
    if title:
        fig.suptitle(title)
    fig.show()