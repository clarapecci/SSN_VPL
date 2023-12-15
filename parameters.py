import os
import numpy
from pdb import set_trace
from dataclasses import dataclass
import jax
import jax.numpy as np




# SSN grid parameters
class grid_pars():
    gridsize_Nx: int = 9 
    ''' grid-points across each edge - gives rise to dx = 0.8 mm '''
    gridsize_deg: float = 2 * 1.6  
    ''' edge length in degrees - visual field '''
    magnif_factor: float = 2.0
    ''' converts deg to mm (mm/deg) '''
    hyper_col: float = 0.4  
    ''' ? are our grid points represent columns? (mm) '''

#Gabor filter parameters for SSN middle
class filter_pars():
    sigma_g = numpy.array(0.27)
    '''Standard deviation of Gaussian in Gabor '''
    conv_factor = numpy.array(2)
    ''' Convert from degrees to mm'''
    k: float = 1 # Clara approximated it by 1; Ke used 1 too
    '''Spatial frequency of Gabor filter'''
    edge_deg: float = grid_pars.gridsize_deg
    '''Axis of Gabor filter goes from -edge_deg, to +edge_deg'''
    degree_per_pixel = numpy.array(0.05) 
    ''' Converts from degrees to number of pixels ''' # convert degree to number of pixels (129 x 129), this could be calculated from earlier params '''

    
#Training data parameters    
@dataclass
class StimuliPars(): #the attributes are changed within SSN_classes for a local instance
    inner_radius: float = 2.5 # inner radius of the stimulus
    outer_radius: float = 3.0 # outer radius of the stimulus: together with inner_radius, they define how the edge of the stimulus fades away to the gray background
    grating_contrast: float = 0.8 # from Current Biology 2020 Ke's paper
    std: float =  200.0 #noise at the moment but  this is a Gaussian white noise added to the stimulus
    jitter_val: float = 5.0 # uniform jitter between [-5, 5] to make the training stimulus vary
    k: float = filter_pars.k # It would be great to get rid of this because FILTER_PARS HAS IT but then it is used when it is passed to new_two_stage_training at BW_Grating
    edge_deg: float = filter_pars.edge_deg # same as for k
    degree_per_pixel = filter_pars.degree_per_pixel # same as for k
    ref_ori: float = 55
    offset: float = 4.0
stimuli_pars = StimuliPars()

#SSN constant parameters
class ssn_pars():
    n = 2.0 # power law parameter
    k = 0.04 # power law parameter
    tauE = 20.0  # time constant for excitatory neurons in ms
    tauI = 10.0  # time constant for inhibitory neurons in ms~
    psi = 0.774 # when we make J2x2 to normalize
    A = None # normalization param for Gabors to get 100% contrast, see find_A
    A2 = None # normalization param for Gabors to get 100% contrast, see find_A
    phases = 4# or 4
    

#Conn pars middle layer
class conn_pars_m():
    PERIODIC: bool = False
    p_local = None

#Conn pars superficial layer
class conn_pars_s():
    PERIODIC: bool = False
    p_local = None

    
# Convergence parameters
class conv_pars():
    dt: float = 1.0
    '''Step size during convergence '''
    xtol: float = 1e-03
    '''Convergence tolerance  '''
    Tmax: float = 250.0
    '''Maximum number of steps to be taken during convergence'''
    Rmax_E = 40
    '''Maximum firing rate for E neurons - rates above this are penalised'''
    Rmax_I = 80
    '''Maximum firing rate for I neurons - rates above this are penalised '''
    lambda_rmax = 1
    lambda_rmean = 1

#Training parameters
@dataclass
class TrainingPars():
    eta = 10e-4
    batch_size = 50
    N_readout = 125 
    ''' sig_noise = 1/sqrt(dt_readout * N_readout), for dt_readout = 0.2, N_readout = 125, sig_noise = 2.0 '''
    epochs = 1000
    num_epochs_to_save = 101
    first_stage_acc = 0.79
    '''Paremeters of sigmoid layer are trained in the first stage until this accuracy is reached '''
training_pars = TrainingPars()

#Loss weights
class loss_pars():
    lambda_dx = 1
    ''' Constant for loss with respect to convergence of Euler function'''
    lambda_r_max = 1
    ''' Constant for loss with respect to maximum rates in the network'''
    lambda_w = 1
    ''' Constant for L2 regularizer of sigmoid layer weights'''
    lambda_b = 1
    ''' Constant for L2 regulazier of sigmoid layer bias '''
    
    
ssn_ori_map_loaded = np.load(os.path.join(os.getcwd(), 'orientation_maps', 'ssn_map_uniform_good.npy'))

