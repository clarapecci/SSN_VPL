import jax.numpy as np
from util import Euler2fixedpt
from jax import random

from SSN_classes_jax_on_only import _SSN_Base, _SSN_AMPAGABA_Base
        

class SSN2DTopoV1_ONOFF(_SSN_Base):
    _Lring = 180

    def __init__(self, ssn_pars, grid_pars,  conn_pars, J_2x2, s_2x2, **kwargs):
        self.Nc = grid_pars.gridsize_Nx**2 #number of columns
        Ni = Ne = 2 * self.Nc 
        
        n=ssn_pars.n
        k=ssn_pars.k
        tauE= ssn_pars.tauE
        tauI=ssn_pars.tauI
        tau_vec = np.hstack([tauE * np.ones(Ne), tauI * np.ones(Ni)])
        tau_s=ssn_pars.tau_s
  
        super(SSN2DTopoV1_ONOFF, self).__init__(n=n, k=k, Ne=Ne, Ni=Ni,
                                    tau_vec=tau_vec, **kwargs)
        
        self.grid_pars = grid_pars
        self.conn_pars = conn_pars
        self._make_maps(grid_pars)
        if conn_pars is not None: # conn_pars = None allows for ssn-object initialization without a W
            self.make_W(J_2x2, s_2x2, **conn_pars)

    @property
    def neuron_params(self):
        return dict(n=self.n, k=self.k,
                    tauE=self.tau_vec[0], tauI=self.tau_vec[self.Ne])
    @property
    def maps_vec(self):
        return np.vstack([self.x_vec, self.y_vec, self.ori_vec]).T

    @property
    def center_inds(self):
        """ indices of center-E and center-I neurons """
        return np.where((self.x_vec==0) & (self.y_vec==0))[0]

    @property
    def x_vec_degs(self):
        return self.x_vec / self.grid_pars.magnif_factor

    @property
    def y_vec_degs(self):
        return self.y_vec / self.grid_pars.magnif_factor

    def xys2inds(self, xys=[[0,0]], units="degree"):
        """
        indices of E and I neurons at location (x,y) (by default in degrees).
        In:
            xys: array-like list of xy coordinates.
            units: specifies unit for xys. By default, "degree" of visual angle.
        Out:
            inds: shape = (2, len(xys)), inds[0] = vector-indices of E neurons
                                         inds[1] = vector-indices of I neurons
        """
        inds = []
        for xy in xys:
            if units == "degree": # convert to mm
                xy = self.grid_pars.magnif_factor * np.asarray(xy)
            distsq = (self.x_vec - xy[0])**2 + (self.y_vec - xy[1])**2
            inds.append([np.argmin(distsq[:self.Ne]), self.Ne + np.argmin(distsq[self.Ne:])])
        return np.asarray(inds).T

    def xys2Emapinds(self, xys=[[0,0]], units="degree"):
        """
        (i,j) of E neurons at location (x,y) (by default in degrees).
        In:
            xys: array-like list of xy coordinates.
            units: specifies unit for xys. By default, "degree" of visual angle.
        Out:
            map_inds: shape = (2, len(xys)), inds[0] = row_indices of E neurons in map
                                         inds[1] = column-indices of E neurons in map
        """
        vecind2mapind = lambda i: np.array([i % self.grid_pars.gridsize_Nx,
                                            i // self.grid_pars.gridsize_Nx])
        return vecind2mapind(self.xys2inds(xys)[0])

    def vec2map(self, vec):
        assert vec.ndim == 1
        Nx = self.grid_pars.gridsize_Nx
        if len(vec) == self.Nc:
            map = np.reshape(vec, (Nx, Nx))
        elif len(vec) == self.Ne:
            map = (np.reshape(vec[:self.Nc], (Nx, Nx)),
                   np.reshape(vec[self.Nc:], (Nx, Nx)))
        elif len(vec) == self.N:
            map = (np.reshape(vec[:self.Nc], (Nx, Nx)),
                   np.reshape(vec[self.Nc:self.Nc*2], (Nx, Nx)),
                   np.reshape(vec[self.Nc*2:self.Nc*3], (Nx, Nx)),
                   np.reshape(vec[self.Nc*3:], (Nx, Nx)))
            
       
        return map

    def _make_maps(self, grid_pars=None):
        if grid_pars is None:
            grid_pars = self.grid_pars
        else:
            self.grid_pars = grid_pars

        self._make_retinmap()
        self._make_orimap()

        return self.x_map, self.y_map, self.ori_map

    def _make_retinmap(self, grid_pars=None):
        """
        make square grid of locations with X and Y retinotopic maps
        """
        if grid_pars is None:
            grid_pars = self.grid_pars
        else:
            self.grid_pars = grid_pars
        if not hasattr(grid_pars, "gridsize_mm"):
            self.grid_pars.gridsize_mm = grid_pars.gridsize_deg * grid_pars.magnif_factor
        Lx = Ly = self.grid_pars.gridsize_mm
        Nx = Ny = grid_pars.gridsize_Nx
        dx = dy = Lx/(Nx - 1)
        self.grid_pars.dx = dx # in mm
        self.grid_pars.dy = dy # in mm

        xs = np.linspace(0, Lx, Nx)
        ys = np.linspace(0, Ly, Ny)
        [X, Y] = np.meshgrid(xs - xs[len(xs)//2], ys - ys[len(ys)//2]) # doing it this way, as opposed to using np.linspace(-Lx/2, Lx/2, Nx) (for which this fails for even Nx), guarantees that there is always a pixel with x or y == 0
        Y = -Y # without this Y decreases going upwards

        self.x_map = X
        self.y_map = Y
        
        self.x_vec = np.tile(X.ravel(), (4,))
        self.y_vec = np.tile(Y.ravel(), (4,))
      
        
        return self.x_map, self.y_map

    def _make_orimap(self, hyper_col=None, nn=30, X=None, Y=None):
        '''
        Makes the orientation map for the grid, by superposition of plane-waves.
        hyper_col = hyper column length for the network in retinotopic degrees
        nn = (30 by default) # of planewaves used to construct the map

        Outputs/side-effects:
        OMap = self.ori_map = orientation preference for each cell in the network
        self.ori_vec = vectorized OMap
        '''
        if hyper_col is None:
             hyper_col = self.grid_pars.hyper_col
        else:
             self.grid_pars.hyper_col = hyper_col
        X = self.x_map if X is None else X
        Y = self.y_map if Y is None else Y

        z = np.zeros_like(X)
        key = random.PRNGKey(87)
        for j in range(nn):
            kj = np.array([np.cos(j * np.pi/nn), np.sin(j * np.pi/nn)]) * 2*np.pi/(hyper_col)
            
            ## JAX CHANGES ##
            
            key, subkey = random.split(key)
            sj = 2 *random.randint(key=key, shape=[1,1], minval=0, maxval=2)-1 #random number that's either + or -1.
            key, subkey = random.split(key)
            phij = random.uniform(key, shape=[1,1], minval=0, maxval=1)*2*np.pi

            tmp = (X*kj[0] + Y*kj[1]) * sj + phij
            z = z + np.exp(1j * tmp)

        # ori map with preferred orientations in the range (0, _Lring] (i.e. (0, 180] by default)
        self.ori_map = (np.angle(z) + np.pi) * SSN2DTopoV1_ONOFF._Lring/(2*np.pi)
        # #for debugging/testing:
        # self.ori_map = 180 * (self.y_map - self.y_map.min())/(self.y_map.max() - self.y_map.min())
        # self.ori_map[self.ori_map.shape[0]//2+1:,:] = 180
        self.ori_vec = np.tile(self.ori_map.ravel(), (4,))
        
        return self.ori_map

    def _make_distances(self, PERIODIC):
        Lx = Ly = self.grid_pars.gridsize_mm
        absdiff_ring = lambda d_x, L: np.minimum(np.abs(d_x), L - np.abs(d_x))
        if PERIODIC:
            absdiff_x = absdiff_y = lambda d_x: absdiff_ring(d_x, Lx + self.grid_pars.dx)
        else:
            absdiff_x = absdiff_y = lambda d_x: np.abs(d_x)
        
        
        xs = np.reshape(self.x_vec, (4, self.Nc, 1)) # (cell-type, grid-location, None)
        ys = np.reshape(self.y_vec, (4, self.Nc, 1)) # (cell-type, grid-location, None)
        oris = np.reshape(self.ori_vec, (4, self.Nc, 1)) # (cell-type, grid-location, None)
        
        # to generalize the next two lines, can replace 0's with a and b in range(2) (pre and post-synaptic cell-type indices)
        xy_dist = np.sqrt(absdiff_x(xs[0] - xs[0].T)**2 + absdiff_y(ys[0] - ys[0].T)**2)
        ori_dist = absdiff_ring(oris[0] - oris[0].T, SSN2DTopoV1_ONOFF._Lring)
        self.xy_dist = xy_dist
        self.ori_dist = ori_dist

        return xy_dist, ori_dist  

    
    def make_W(self, J_2x2, s_2x2, p_local, sigma_oris=45, Jnoise=0,
                Jnoise_GAUSSIAN=True, MinSyn=1e-4, CellWiseNormalized=True,
                                                    PERIODIC=True): #, prngKey=0):
        """
        make the full recurrent connectivity matrix W
        In:
         J_2x2 = total strength of weights of different pre/post cell-type
         s_2x2 = ranges of weights between different pre/post cell-type
         p_local = relative strength of local parts of E projections
         sigma_oris = range of wights in terms of preferred orientation difference

        Output/side-effects:
        self.W
        """
        conn_pars = locals()
        conn_pars.pop("self")
        self.conn_pars = conn_pars

        if hasattr(self, "xy_dist") and hasattr(self, "ori_dist"):
            xy_dist = self.xy_dist
            ori_dist = self.ori_dist
        else:
            xy_dist, ori_dist = self._make_distances(PERIODIC)

        if np.isscalar(sigma_oris): sigma_oris = sigma_oris * np.ones((2,2))

        if np.isscalar(p_local) or len(p_local) == 1:
            p_local = np.asarray(p_local) * np.ones(2)

        Wblks = [[1,1],[1,1]]
        # loop over post- (a) and pre-synaptic (b) cell-types
        for a in range(2):
            for b in range(2):
                if b == 0: # E projections
                    W = np.exp(-xy_dist/s_2x2[a,b] -ori_dist**2/(2*sigma_oris[a,b]**2))
                elif b == 1: # I projections
                    W = np.exp(-xy_dist**2/(2*s_2x2[a,b]**2) -ori_dist**2/(2*sigma_oris[a,b]**2))

                if Jnoise > 0: # add some noise
                    if Jnoise_GAUSSIAN:
                        ##JAX CHANGES##
                        #jitter = np.random.standard_normal(W.shape)
                        key = random.PRNGKey(87)
                        key, subkey=random.split(key)
                        jitter = random.normal(key, W.shape)
                    else:
                        ##JAX CHANGES##
                       #jitter = 2* np.random.random(W.shape) - 1
                        key = random.PRNGKey(87)
                        key, subkey=random.split(key)
                        jitter = 2* random.uniform(key, W.shape) - 1
                    W = (1 + Jnoise * jitter) * W

                # sparsify (set small weights to zero)
                W = np.where(W < MinSyn, 0, W) # what's the point of this if not using sparse matrices

                # row-wise normalize
                tW = np.sum(W, axis=1)
                if not CellWiseNormalized:
                    tW = np.mean(tW)
                W = W / tW

                # for E projections, add the local part
                # NOTE: alterntaively could do this before adding noise & normalizing
                if b == 0:
                    W = p_local[a] * np.eye(*W.shape) + (1-p_local[a]) * W

                Wblks[a][b] = J_2x2[a, b] * W

        self.W = np.block(Wblks)
        
        B=np.ones((2,2))
        self.W=np.kron(B, self.W)
        
        return self.W

    
    
    '''
    def _make_inp_ori_dep(self, ONLY_E=False, ori_s=None, sig_ori_EF=32, sig_ori_IF=None, gE=1, gI=1):
        """
        makes the orintation dependence factor for grating or Gabor stimuli
        (a la Ray & Maunsell 2010)
        """
        if ori_s is None:  # set stim ori to pref ori of grid center E cell (same as I cell)
            ##JAX CHANGES##
            #ori_s = self.ori_vec[(self.x_vec==0) & (self.y_vec==0) & (self.EI==b"E")]
            ori_s = self.ori_vec[(self.x_vec==0) & (self.y_vec==0) & self.condition]
        if sig_ori_IF is None:
            sig_ori_IF = sig_ori_EF

        distsq = lambda x: np.minimum(np.abs(x), SSN2DTopoV1._Lring - np.abs(x))**2
        dori = self.ori_vec - ori_s
        if not ONLY_E:
            ori_fac = np.hstack((gE * np.exp(-distsq(dori[:self.Ne])/(2* sig_ori_EF**2)),
                                 gI * np.exp(-distsq(dori[self.Ne:])/(2* sig_ori_IF**2))))
        else:
            ori_fac = gE * np.exp(-distsq(dori[:self.Ne])/(2* sig_ori_EF**2))

        return ori_fac

    def make_grating_input(self, radius_s, sigma_RF=0.4, ONLY_E=False,
            ori_s=None, sig_ori_EF=32, sig_ori_IF=None, gE=1, gI=1, contrast=1):
        """
        make grating external input centered on the grid-center, with radius "radius",
        with edge-fall-off scale "sigma_RF", with orientation "ori_s",
        with the orientation tuning-width of E and I parts given by "sig_ori_EF"
        and "sig_ori_IF", respectively, and with amplitue (maximum) of the E and I parts,
        given by "contrast * gE" and "contrast * gI", respectively.
        If ONLY_E=True, it only makes the E-part of the input vector.
        """
        # make the orintation dependence factor:
        ori_fac = self._make_inp_ori_dep(ONLY_E, ori_s, sig_ori_EF, sig_ori_IF, gE, gI)

        # make the spatial envelope:
        sigmoid = lambda x: 1/(1 + np.exp(-x))
        M = self.Ne if ONLY_E else self.N
        r_vec = np.sqrt(self.x_vec_degs[:M]**2 + self.y_vec_degs[:M]**2)
        spat_fac = sigmoid((radius_s - r_vec)/sigma_RF)

        return contrast * ori_fac * spat_fac

    def make_gabor_input(self, sigma_Gabor=0.5, ONLY_E=False,
            ori_s=None, sig_ori_EF=32, sig_ori_IF=None, gE=1, gI=1, contrast=1):
        """
        make the Gabor stimulus (a la Ray & Maunsell 2010) centered on the
        grid-center, with sigma "sigma_Gabor",
        with orientation "ori_s",
        with the orientation tuning-width of E and I parts given by "sig_ori_EF"
        and "sig_ori_IF", respectively, and with amplitue (maximum) of the E and I parts,
        given by "contrast * gE" and "contrast * gI", respectively.
        """
        # make the orintation dependence factor:
        ori_fac = self._make_inp_ori_dep(ONLY_E, ori_s, sig_ori_EF, sig_ori_IF, gE, gI)

        # make the spatial envelope:
        gaussian = lambda x: np.exp(- x**2 / 2)
        M = self.Ne if ONLY_E else self.N
        r_vec = np.sqrt(self.x_vec_degs[:M]**2 + self.y_vec_degs[:M]**2)
        spat_fac = gaussian(r_vec/sigma_Gabor)

        return contrast * ori_fac * spat_fac

    # TODO:
    # def make_noise_cov(self, noise_pars):
    #     # the script assumes independent noise to E and I, and spatially uniform magnitude of noise
    #     noise_sigsq = np.hstack( (noise_pars.stdevE**2 * np.ones(self.Ne),
    #                            noise_pars.stdevI**2 * np.ones(self.Ni)) )
    #
    #     spatl_filt = ...

    def make_eLFP_from_inds(self, LFPinds):
        """
        makes a single LFP electrode signature (normalized spatial weight
        profile), given the (vectorized) indices of recorded neurons (LFPinds).

        OUT: e_LFP with shape (self.N,)
        """
        # LFPinds was called LFPrange in my MATLAB code
        if LFPinds is None:
            LFPinds = [0]
        e_LFP = 1/len(LFPinds) * np.isin(np.arange(self.N), LFPinds) # assuming elements of LFPinds are all smaller than self.Ne, e_LFP will only have 1's on E elements
        # eI = 1/len(LFPinds) * np.isin(np.arange(self.N) - self.Ne, LFPinds) # assuming elements of LFPinds are all smaller than self.Ne, e_LFP will only have 1's on I elements

        return e_LFP

    def make_eLFP_from_xy(self, probe_xys, LFPradius=0.2, unit_xys="degree", unit_rad="mm"):
        """
        makes 1 or multiple LFP electrodes signatures (normalized spatial weight
        profile over E cells), given the (x,y) retinotopic coordinates of LFP probes.

        IN: probe_xys: shape (#probes, 2). Each row is the (x,y) coordinates of
                 a probe/electrode (by default given in degrees of visual angle)
             LFPradius: positive scalar. radius/range of LFP (by default given in mm)
            unit_xys: either "degree" or "mm", unit of LFP_xys
            unit_rad: either "degree" or "mm", unit of LFPradius
        OUT: e_LFP: shape (self.N, #probes) = (self.N, LFP.xys.shape[0])
             Each column is the normalized spatial profile of one probe.
        """
        if unit_rad == "degree":
            LFPradius = self.grid_pars.magnif_factor * LFPradius

        e_LFP = []
        for xy in probe_xys:
            if unit_xys == "degree": # convert to mm
                xy = self.grid_pars.magnif_factor * np.asarray(xy)
            e_LFP.append(1.0 * ( (self.EI == b"E") &
            (LFPradius**2 > (self.x_vec - xy[0])**2 + (self.y_vec - xy[1])**2)))

        return np.asarray(e_LFP).T
        
        '''
    
class SSN2DTopoV1_AMPAGABA_ONOFF(SSN2DTopoV1_ONOFF, _SSN_AMPAGABA_Base):
    pass
    