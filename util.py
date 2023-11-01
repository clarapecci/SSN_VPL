import jax
from jax import random
import math
from PIL import Image
from random import random
from scipy.stats import norm
import jax.numpy as np
from jax import random
from jax import vmap
import pandas as pd
import matplotlib.pyplot as plt
import numpy 
from torch.utils.data import DataLoader
from numpy.random import binomial
from pdb import set_trace
import os



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
            
            #set_trace()
            
        
        
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
'''
def Euler2fixedpt(dxdt, x_initial, Tmax, dt, xtol=1e-5, xmin=1e-0, Tmin=200, PLOT=False, inds=None, verbose=False, silent=False, Tfrac_CV=0):
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
    verbose: if True print convergence criteria even if passed (function always prints out a warning if it doesn't converge).
    Tfrac_var: if not zero, maximal temporal CV (coeff. of variation) of state vector components, over the final
               Tfrac_CV fraction of Euler timesteps, is calculated and printed out.
               
    OUT:
    xvec = found fixed point solution
    CONVG = True if determined converged, False if not
    """

    if PLOT:
        if inds is None:
            x_dim = x_initial.size
            inds = [int(x_dim/4), int(3*x_dim/4)]
        xplot = x_initial.flatten()[inds][:,None]

    Nmax = int(np.round(Tmax/dt))
    Nmin = int(np.round(Tmin/dt)) if Tmax > Tmin else int(Nmax/2)
    xvec = x_initial
    CONVG = False

    if Tfrac_CV > 0:
        xmean = np.zeros_like(xvec)
        xsqmean = np.zeros_like(xvec)
        Nsamp = 0

    for n in range(Nmax):
        dx = dxdt(xvec) * dt
        xvec = xvec + dx
        if PLOT:
            xplot = np.hstack((xplot, xvec.flatten()[inds][:,None]))
        
        if Tfrac_CV > 0 and n >= (1-Tfrac_CV) * Nmax:
            xmean = xmean + xvec
            xsqmean = xsqmean + xvec**2
            Nsamp = Nsamp + 1

        if n > Nmin:
            if np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max() < xtol:
                if verbose:
                    print("      converged to fixed point at iter={},      as max(abs(dx./max(xvec,{}))) < {} ".format(n, xmin, xtol))
                CONVG = True
                break

    if not CONVG and not silent: # n == Nmax:
        print("\n Warning 1: reached Tmax={}, before convergence to fixed point.".format(Tmax))
        print("       max(abs(dx./max(abs(xvec), {}))) = {},   xtol={}.\n".format(xmin, np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max(), xtol))

        if Tfrac_CV > 0:
            xmean = xmean/Nsamp
            xvec_SD = np.sqrt(xsqmean/Nsamp - xmean**2)
            # CV = xvec_SD / xmean
            # CVmax = CV.max()
            CVmax = xvec_SD.max() / xmean.max()
            print(f"max(SD)/max(mean) of state vector in the final {Tfrac_CV:.2} fraction of Euler steps was {CVmax:.5}")

        #mybeep(.2,350)
        #beep

    if PLOT:
        import matplotlib.pyplot as plt
        plt.figure(244459)
        plt.plot(np.arange(n+2)*dt, xplot.T, 'o-')

    return xvec, CONVG
'''


# this is copied from scipy.linalg, to make compatible with jax.numpy
def toeplitz(c, r=None):
    """
    Construct a Toeplitz matrix.
    The Toeplitz matrix has constant diagonals, with c as its first column
    and r as its first row.  If r is not given, ``r == conjugate(c)`` is
    assumed.
    Parameters
    ----------
    c : array_like
        First column of the matrix.  Whatever the actual shape of `c`, it
        will be converted to a 1-D array.
    r : array_like
        First row of the matrix. If None, ``r = conjugate(c)`` is assumed;
        in this case, if c[0] is real, the result is a Hermitian matrix.
        r[0] is ignored; the first row of the returned matrix is
        ``[c[0], r[1:]]``.  Whatever the actual shape of `r`, it will be
        converted to a 1-D array.
    Returns
    -------
    A : (len(c), len(r)) ndarray
        The Toeplitz matrix. Dtype is the same as ``(c[0] + r[0]).dtype``.
    See also
    --------
    circulant : circulant matrix
    hankel : Hankel matrix
    Notes
    -----
    The behavior when `c` or `r` is a scalar, or when `c` is complex and
    `r` is None, was changed in version 0.8.0.  The behavior in previous
    versions was undocumented and is no longer supported.
    Examples
    --------
    >>> from scipy.linalg import toeplitz
    >>> toeplitz([1,2,3], [1,4,5,6])
    array([[1, 4, 5, 6],
           [2, 1, 4, 5],
           [3, 2, 1, 4]])
    >>> toeplitz([1.0, 2+3j, 4-1j])
    array([[ 1.+0.j,  2.-3.j,  4.+1.j],
           [ 2.+3.j,  1.+0.j,  2.-3.j],
           [ 4.-1.j,  2.+3.j,  1.+0.j]])
    """
    c = np.asarray(c).ravel()
    if r is None:
        r = c.conjugate()
    else:
        r = np.asarray(r).ravel()
    # Form a 1D array of values to be used in the matrix, containing a reversed
    # copy of r[1:], followed by c.
    vals = np.concatenate((r[-1:0:-1], c))
    a, b = np.ogrid[0:len(c), len(r) - 1:-1:-1]
    indx = a + b
    # `indx` is a 2D array of indices into the 1D array `vals`, arranged so
    # that `vals[indx]` is the Toeplitz matrix.
    return vals[indx]






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
    Nmax = (np.round(Tmax/dt)).astype(int)
    #Nmin = (np.round(Tmin/dt)).astype(int) if Tmax > Tmin else (Nmax/2)
    xvec = x_initial 
    CONVG = False
    y = []
    
    for n in range(Nmax.astype(int)):
        dx = dxdt(xvec) * dt
        xvec = xvec + dx
        y.append(np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max())
    
    y = np.asarray(y)  
    avg_dx = y[int(Nmax/2):int(Nmax)].mean()/xtol
    CONVG = np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max() < xtol
    return xvec, CONVG, avg_dx


def take_log(J_2x2):
    
    signs=np.array([[1, -1], [1, -1]])
    logJ_2x2 =np.log(J_2x2*signs)
    
    return logJ_2x2

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
            
            #CHANGED K HERE
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

    
### FINDING CONSTANT FOR GABOR FILTERS ###
def find_A(conv_factor, k, sigma_g, edge_deg,  degree_per_pixel, indices, phase = 0, return_all=False):
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
        gabor=GaborFilter(theta=ori, x_i=0, y_i=0, edge_deg=edge_deg, k=k, sigma_g=sigma_g, degree_per_pixel=degree_per_pixel, phase = phase)
        test_grating=BW_Grating(ori_deg=ori, edge_deg=edge_deg, k=k, degree_per_pixel=degree_per_pixel, outer_radius=edge_deg*2, inner_radius=edge_deg*2, grating_contrast=0.99, phase = phase)
        test_stimuli=test_grating.BW_image()
        
        mean_removed_filter = gabor.filter - gabor.filter.mean()
        #multiply filter and stimuli
        #output_gabor=gabor.filter.ravel()@test_stimuli.ravel()
        output_gabor=mean_removed_filter.ravel()@test_stimuli.ravel()
        
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
        self.angle = ((self.ori_deg + self.jitter) - 90) / 180 * numpy.pi
      

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


## PLOT INITIALIZATION HISTOGRAM
import numpy
def vmap_eval2(opt_pars, ssn_pars, grid_pars, conn_pars, test_data, filter_pars,  conv_pars):
    '''
    For a given value of the weights, calculate the loss for all the stimuli.
    Output:
        losses: size(n_stimuli)
        Accuracy: scalar
    '''
    
    eval_vmap = vmap(model, in_axes = ({'b_sig': None, 'logJ_2x2': None, 'logs_2x2': None, 'w_sig': None, 'c_E':None, 'c_I':None}, None, None, {'PERIODIC': None, 'p_local': [None, None], 'sigma_oris': None},  {'ref':0, 'target':0, 'label':0}, {'conv_factor': None, 'degree_per_pixel': None, 'edge_deg': None, 'k': None, 'sigma_g': None}, {'Tmax': None, 'dt': None, 'silent': None, 'verbose': None, 'xtol': None}) )
    losses, pred_labels = eval_vmap(opt_pars, ssn_pars, grid_pars, conn_pars, test_data, filter_pars,  conv_pars)
        
    accuracy = np.sum(test_data['label'] == pred_labels)/len(test_data['label']) 
    
    return losses, accuracy

def vmap_eval3(opt_pars, ssn_pars, grid_pars, conn_pars, test_data, filter_pars,  conv_pars):
    '''
    Iterates through all values of 'w' to give the losses at each stimuli and weight, and the accuracy at each weight
    Output:
        losses: size(n_weights, n_stimuli )
        accuracy: size( n_weights)
    '''

    eval_vmap = vmap(vmap_eval2, in_axes = ({'b_sig': None, 'logJ_2x2': None, 'logs_2x2': None, 'w_sig': 0, 'c_E':None, 'c_I':None}, None, None, {'PERIODIC': None, 'p_local': [None, None], 'sigma_oris': None},  {'ref':None, 'target':None, 'label':None}, {'conv_factor': None, 'degree_per_pixel': None, 'edge_deg': None, 'k': None, 'sigma_g': None}, {'Tmax': None, 'dt': None, 'silent': None, 'verbose': None, 'xtol': None}) )
    losses, accuracies = eval_vmap(opt_pars, ssn_pars, grid_pars, conn_pars, test_data, filter_pars,  conv_pars)
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
    Js=res[:,2:6]
    ss = res[:,6:10]
    #print(Js.type())
    print(results.columns[2:6])
    print("J ratios = ", np.array((Js[-1,:]/Js[0,:] -1)*100, dtype=int))
    print(results.columns[6:10])
    print(ss[-1,:]/ss[0,:])
    print("s ratios = ", np.array((ss[-1,:]/ss[0,:] -1)*100, dtype=int))
    

def plot_losses(training_losses, save_file = None):
    plt.plot(training_losses.T, label = ['Binary cross entropy', 'Avg_dx', 'R_max', 'w', 'b', 'Total'] )
    plt.legend()
    if save_file:
        plt.savefig(save_file)
    else:
        plt.show()
    
def ratio_w(new_pars, opt_pars ):
    all_ratios = []
    for i in range(len(new_pars['w_sig'])):
        all_ratios.append((new_pars['w_sig'][i] / opt_pars['w_sig'][i] - 1 )*100)
        
    return np.asarray(all_ratios)



    
#import numpy as np
import h5py


def save_h5(filename, dic):

    """

    saves a python dictionary or list, with items that are themselves either

    dictionaries or lists or (in the case of tree-leaves) numpy arrays

    or basic scalar types (int/float/str/bytes) in a recursive

    manner to an hdf5 file, with an intact hierarchy.

    """

    with h5py.File(filename, 'w') as h5file:

        recursively_save_dict_contents_to_group(h5file, '/', dic)
        

def recursively_save_dict_contents_to_group(h5file, path, dic):

    """

    ....

    """

    if isinstance(dic,dict):

        iterator = dic.items()

    elif isinstance(dic,list):

        iterator = enumerate(dic)

    else:

        ValueError('Cannot save %s type'%type(item))



    for key, item in iterator: #dic.items():

        if isinstance(dic,list):

            key = str(key)

        if isinstance(item, numpy.ndarray) or np.isscalar(item):

            h5file[path + key] = item

        elif isinstance(item, dict) or isinstance(item,list):

            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)

        else:

            raise ValueError('Cannot save %s type'%type(item))



def load(filename,ASLIST=False):

    """

    Default: load a hdf5 file (saved with io_dict_to_hdf5.save function above) as a hierarchical

    python dictionary (as described in the doc_string of io_dict_to_hdf5.save).

    if ASLIST is True: then it loads as a list (on in the first layer) and gives error if key's are not convertible

    to integers. Unlike io_dict_to_hdf5.save, a mixed dictionary/list hierarchical version is not implemented currently

    for .load

    """

    with h5py.File(filename, 'r') as h5file:

        out = recursively_load_dict_contents_from_group(h5file, '/')

        if ASLIST:

            outl = [None for l in range(len(out.keys()))]

            for key, item in out.items():

                outl[int(key)] = item

            out = outl

        return out


def recursively_load_dict_contents_from_group(h5file, path):

    ans = {}

    for key, item in h5file[path].items():

        if isinstance(item, h5py._hl.dataset.Dataset):

            ans[key] = item[()]

        elif isinstance(item, h5py._hl.group.Group):

            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')

    return ans




def load_param_from_csv(results_filename, epoch):
    
    all_results = pd.read_csv(results_filename, header = 0)
    epoch_params = all_results.loc[all_results['epoch'] == epoch]
    params = []
    J_m = [np.abs(epoch_params[i].values[0]) for i in ['J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m']]
    J_s = [np.abs(epoch_params[i].values[0]) for i in ['J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s']]
    c_E = epoch_params['c_E'].values[0]
    c_I = epoch_params['c_I'].values[0]

    J_2x2_m = make_J2x2_o(*J_m)
    J_2x2_s = make_J2x2_o(*J_s)
    params.append(J_2x2_m)
    params.append(J_2x2_s)
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

def create_stimuli(stimuli_pars, ref_ori, number = 10, jitter_val = 5):

    all_stimuli = []

    for i in range(0, number):
        jitter = numpy.random.uniform(-jitter_val, jitter_val, 1)

        #create reference grating
        ref = BW_Grating(ori_deg = ref_ori, jitter=jitter, **stimuli_pars).BW_image().ravel()
        all_stimuli.append(ref)
    
    return np.vstack([all_stimuli])


        
def save_matrices(run_dir, contrast, matrix_sup, matrix_ref):
    np.save(os.path.join(run_dir+str(contrast)+'sup.npy'), matrix_sup) 
    np.save(os.path.join(run_dir+str(contrast)+'mid.npy'), matrix_ref) 
    
    
def load_matrix_response(results_dir, layer): 
    run_dir = os.path.join(results_dir, 'response_matrix_')
    
    response_matrix_contrast_02 = np.load(run_dir+'0.2'+str(layer)+'.npy')
    response_matrix_contrast_04= np.load(run_dir+'0.4'+str(layer)+'.npy')
    response_matrix_contrast_06 = np.load(run_dir+'0.6'+str(layer)+'.npy')
    response_matrix_contrast_08 = np.load(run_dir+'0.8'+str(layer)+'.npy')
    response_matrix_contrast_099 = np.load(run_dir+'0.99'+str(layer)+'.npy')
    
    return response_matrix_contrast_02, response_matrix_contrast_04, response_matrix_contrast_06, response_matrix_contrast_08, response_matrix_contrast_099



####FUNCTIONS IN TRAINING #############
