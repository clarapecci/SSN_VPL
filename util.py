#import numpy as np

import jax.numpy as np

def Euler2fixedpt(dxdt, x_initial, Tmax, dt, xtol=1e-5, xmin=1e-0, Tmin=200, PLOT=False, inds=None, verbose=True):
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
    Nmin = int(np.round(Tmin/dt)) if Tmax > Tmin else int(Nmax/2)
    xvec = x_initial
    CONVG = False
    for n in range(Nmax):
        dx = dxdt(xvec) * dt
        xvec = xvec + dx
        if PLOT:
            #xplot = np.asarray([xplot, xvvec[inds]])
            xplot = np.hstack((xplot,xvec[inds][:,None]))

        if n > Nmin:
            if np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max() < xtol:
                if verbose:
                    print("      converged to fixed point at iter={},      as max(abs(dx./max(xvec,{}))) < {} ".format(n, xmin, xtol))
                CONVG = True
                break

    if not CONVG: # n == Nmax:
        print("\n Warning 1: reached Tmax={}, before convergence to fixed point.".format(Tmax))
        print("       max(abs(dx./max(abs(xvec), {}))) = {},   xtol={}.\n".format(xmin, np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max(), xtol))
        #mybeep(.2,350)
        #beep

    if PLOT:
        import matplotlib.pyplot as plt
        plt.figure(244459)
        plt.plot(np.arange(n+2)*dt, xplot.T, 'o-')

    return xvec, CONVG



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

class GaborFilter:
    
    def __init__(self, x_i, y_i, k, sigma_g, theta, x_axis, y_axis, dx, dy, phase=0, conv_factor=None):
        
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
            self.x_i=x_i/conv_factor
            self.y_i=y_i/conv_factor
        else:
            self.x_i=x_i
            self.y_i=y_i
        self.k=k #preferred spatial frequency
        self.theta=theta*(np.pi/180) #preferred orientation - convert to radians
        self.phase=phase 
        self.sigma_g=sigma_g
        self.N_pixels=len(x_axis)
        
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

    
    
    
    
"""
Author: Samuel Bell (sjb326@cam.ac.uk) 
jia_grating.py Copyright (c) 2020. All rights reserved.
This file may not be used or modified without the author's explicit written permission.
These files are hosted at https://gitlab.com/samueljamesbell/vpl-modelling. All other locations are mirrors of the original repository.

This file is a Python port of stimuli developed by Ke Jia.
"""

import math

import numpy as np
from PIL import Image

from random import random
from scipy.stats import norm
from numpy.random import binomial

from numpy import pi


_BLACK = 0
_WHITE = 255
_GRAY = round((_WHITE + _BLACK) / 2)


class JiaGrating:

    def __init__(self, ori_deg, size, outer_radius, inner_radius, pixel_per_degree, grating_contrast, phase, jitter=0, snr=1.0, spatial_frequency=None):
        self.ori_deg = ori_deg
        self.size = size

        self.outer_radius = outer_radius #in pixels
        self.inner_radius = inner_radius #in pixels
        self.pixel_per_degree = pixel_per_degree
        self.grating_contrast = grating_contrast
        self.phase = phase
        self.jitter = jitter
        self.snr = snr

        self.smooth_sd = self.pixel_per_degree / 6
        self.spatial_freq = spatial_frequency or (1 / self.pixel_per_degree)
        self.grating_size = round(self.outer_radius * self.pixel_per_degree)
        self.angle = ((self.ori_deg + self.jitter) - 90) / 180 * math.pi

    def image(self):
        x, y = np.mgrid[-self.grating_size:self.grating_size+1., -self.grating_size:self.grating_size+1.]

        d = self.grating_size * 2 + 1
        annulus = np.ones((d, d))

        edge_control = np.divide(np.sqrt(np.power(x, 2) + np.power(y, 2)), self.pixel_per_degree)

        overrado = np.nonzero(edge_control > self.inner_radius)

        for idx_x, idx_y in zip(*overrado):
            annulus[idx_x, idx_y] = annulus[idx_x, idx_y] * np.exp(-1 * ((((edge_control[idx_x, idx_y] - self.inner_radius) * self.pixel_per_degree) ** 2) / (2 * (self.smooth_sd ** 2))))    

        gabor_sti = _GRAY * (1 + self.grating_contrast * np.cos(2 * math.pi * self.spatial_freq * (y * np.sin(self.angle) + x * np.cos(self.angle)) + self.phase))

        gabor_sti[np.sqrt(np.power(x, 2) + np.power(y, 2)) > self.grating_size] = _GRAY

        # Noise
        noise = np.floor(np.sin(norm.rvs(size=(d, d))) * _GRAY) + _GRAY

        noise_mask = binomial(1, 1-self.snr, size=(d, d)).astype(int)
        masked_noise = noise * noise_mask

        signal_mask = 1 - noise_mask
        masked_gabor_sti = signal_mask * gabor_sti

        noisy_gabor_sti = masked_gabor_sti + masked_noise
        # End noise

        gabor_sti_final = np.repeat(noisy_gabor_sti[:, :, np.newaxis], 3, axis=-1)
        alpha_channel = annulus * _WHITE
        gabor_sti_final_with_alpha = np.concatenate((gabor_sti_final, alpha_channel[:, :, np.newaxis]), axis=-1)
        gabor_sti_final_with_alpha_image = Image.fromarray(gabor_sti_final_with_alpha.astype(np.uint8))

        center_x = int(self.size / 2)
        center_y = int(self.size / 2)
        bounding_box = (center_x - self.grating_size, center_y - self.grating_size)

        background = np.full((self.size, self.size, 3), _GRAY, dtype=np.uint8)
        final_image = Image.fromarray(background)

        final_image.paste(gabor_sti_final_with_alpha_image, box=bounding_box, mask=gabor_sti_final_with_alpha_image)

        return final_image

class BW_Grating(JiaGrating):
    
    def __init__(self, ori_deg, size, outer_radius, inner_radius, pixel_per_degree, grating_contrast, phase, jitter=0, snr=1.0, spatial_frequency=None, crop_f=None):
        
        self.crop_f=crop_f
         
        super().__init__( ori_deg, size, outer_radius, inner_radius, pixel_per_degree, grating_contrast, phase, jitter, snr, spatial_frequency)
        
    def BW_image(self):
        original=np.asarray(self.image())
        image=np.sum(original, axis=2) 
        
        if self.crop_f:
            image=image[self.crop_f:-self.crop_f, self.crop_f:-self.crop_f]            
        return image
    
 