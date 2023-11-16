import jax.numpy as np
from jax import random
import matplotlib.pyplot as plt
import jax
from jax import jit
from functools import partial
from pdb import set_trace
import numpy


from util import find_A, GaborFilter
import util

class _SSN_Base(object):
    def __init__(self, n, k, Ne, Ni, tau_vec=None, W=None):
        self.n = n
        self.k = k
        self.Ne = Ne
        self.Ni = Ni
        self.N = self.Ne + self.Ni

        ## JAX CHANGES ##
        self.EI=[b"E"]*(self.Ne) + [b"I"]*(self.N - self.Ne)
        self.condition= np.array([bool(self.EI[x]==b"E") for x in range(len(self.EI))])
        
        if tau_vec is not None:
            self.tau_vec = tau_vec # rate time-consants of neurons. shape: (N,)
        # elif  not hasattr(self, "tau_vec"):
        #     self.tau_vec = np.random.rand(N) * 20 # in ms
        if W is not None:
            self.W = W # connectivity matrix. shape: (N, N)
        # elif  not hasattr(self, "W"):
        #     W = np.random.rand(N,N) / np.sqrt(self.N)
        #     sign_vec = np.hstack(np.ones(self.Ne), -np.ones(self.Ni))
        #     self.W = W * sign_vec[None, :] # to respect Dale

    @property
    def neuron_params(self):
        return dict(n=self.n, k=self.k)

    @property
    def dim(self):
        return self.N

    @property
    def tau_x_vec(self):
        """ time constants for the generalized state-vector, x """
        return self.tau_vec


    def powlaw(self, u):
        return  self.k * np.maximum(0,u)**self.n

    def drdt(self, r2, inp_vec):
        out = ( -r2 + self.powlaw(self.W @ r2 + inp_vec) ) / self.tau_vec
        return out

    def drdt_multi(self, r, inp_vec, print_dt = False):
        """
        Compared to self.drdt allows for inp_vec and r to be
        matrices with arbitrary shape[1]
        """
        return (( -r + self.powlaw(self.W @ r + inp_vec) ).T / self.tau_vec ).T

    def dxdt(self, x, inp_vec):
        """
        allowing for descendent SSN types whose state-vector, x, is different
        than the rate-vector, r.
        """
        return self.drdt(x, inp_vec)

    def gains_from_v(self, v):
        return self.n * self.k * np.maximum(0,v)**(self.n-1)

    def gains_from_r(self, r):
        return self.n * self.k**(1/self.n) * r**(1-1/self.n)

    def DCjacobian(self, r):
        """
        DC Jacobian (i.e. zero-frequency linear response) for
        linearization around rate vector r
        """
        Phi = self.gains_from_r(r)
        return -np.eye(self.N) + Phi[:, None] * self.W

    def jacobian(self, DCjacob=None, r=None):
        """
        dynamic Jacobian for linearization around rate vector r
        """
        if DCjacob is None:
            assert r is not None
            DCjacob = self.DCjacobian(r)
        return DCjacob / self.tau_x_vec[:, None] # equivalent to np.diag(tau_x_vec) * DCjacob

    def jacobian_eigvals(self, DCjacob=None, r=None):
        Jacob = self.jacobian(DCjacob=DCjacob, r=r)
        return np.linalg.eigvals(Jacob)

    def inv_G(self, omega, DCjacob, r=None):
        """
        inverse Green's function at angular frequency omega,
        for linearization around rate vector r
        """
        if DCjacob is None:
            assert r is not None
            DCjacob = self.DCjacobian(r)
        return -1j*omega * np.diag(self.tau_x_vec) - DCjacob

   ######## USE IN FIXED POINT FUNCTION #################
    
    def fixed_point_r(self, inp_vec, r_init=None, Tmax=500, dt=1, xtol=1e-5, PLOT=False, save=None):
        
        if r_init is None:
            r_init = np.zeros(inp_vec.shape) # np.zeros((self.N,))
        drdt = lambda r : self.drdt(r, inp_vec)
        if inp_vec.ndim > 1:
            drdt = lambda r : self.drdt_multi(r, inp_vec)
        r_fp, avg_dx = self.Euler2fixedpt_fullTmax(drdt, r_init, Tmax, dt, xtol=xtol, PLOT=PLOT, save=save)

        return r_fp, avg_dx

############################################################    
    def fixed_point_r_plot(self, inp_vec, r_init=None, Tmax=500, dt=1, xtol=1e-5, PLOT=True, save = None, inds=None, print_dt = False):
        if r_init is None:
            r_init = np.zeros(inp_vec.shape) # np.zeros((self.N,))
        drdt = lambda r : self.drdt(r, inp_vec)
        print('Inp vec shape ', inp_vec.shape)
        if inp_vec.ndim > 1:
            drdt = lambda r : self.drdt_multi(r, inp_vec)
        xvec, CONVG = util.Euler2fixedpt(dxdt=drdt, x_initial=r_init, Tmax=Tmax, dt=dt, xtol=xtol, PLOT=PLOT, save=save, inds=inds)

        return xvec, CONVG

    def fixed_point(self, inp_vec, x_init=None, Tmax=500, dt=1, xtol=1e-5, PLOT=False):
        
        if x_init is None:
            x_init = np.zeros((self.dim,))
        dxdt = lambda x : self.dxdt(x, inp_vec)
        x_fp, CONVG = util.Euler2fixedpt(dxdt, x_init, Tmax, dt, xtol, PLOT)
        if not CONVG:
            print('Did not reach fixed point.')
        #else:
        #    return x_fp
        return x_fp, CONVG

    def make_noise_cov(self, noise_pars):
        # the script assumes independent noise to E and I, and spatially uniform magnitude of noise
        noise_sigsq = np.hstack( (noise_pars.stdevE**2 * np.ones(self.Ne),
                                  noise_pars.stdevI**2 * np.ones(self.Ni)) )
        spatl_filt = np.array(1)

        return noise_sigsq, spatl_filt
    
    #@partial(jax.jit, static_argnums=(0, 1, 3, 4, 5, 6, 7, 8), device = jax.devices()[1])
    def Euler2fixedpt_fullTmax(self, dxdt, x_initial, Tmax, dt, xtol=1e-5, xmin=1e-0, Tmin=200, PLOT= False, save=None):
        
        Nmax = int(Tmax/dt)
        xvec = x_initial 
        CONVG = False
        y = np.zeros(((Nmax)))        
        
        
        if PLOT:
                #if inds is None:
                #    N = x_initial.shape[0] # x_initial.size
                #    inds = [int(N/4), int(3*N/4)]
                #xplot = x_initial[inds][:,None]
                
                xplot_all = np.zeros(((Nmax+1)))
                xplot_all = xplot_all.at[0].set(np.sum(xvec))
                


                def loop(n, carry):
                    xvec, y, xplot_all = carry
                    dx = dxdt(xvec) * dt
                    xvec = xvec + dx
                    y = y.at[n].set(np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max())
                    xplot_all = xplot_all.at[n+1].set(np.sum(xvec))
                    return (xvec, y, xplot_all)

                xvec, y, xplot_all = jax.lax.fori_loop(0, Nmax, loop, (xvec, y, xplot_all))
            
            
        else:
                
                def loop(n, carry):
                    xvec, y = carry
                    dx = dxdt(xvec) * dt
                    xvec = xvec + dx
                    y = y.at[n].set(np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max())
                    return (xvec, y)

                xvec, y = jax.lax.fori_loop(0, Nmax, loop, (xvec, y))
        
        avg_dx = y[int(Nmax/2):int(Nmax)].mean()/xtol
        
        #CONVG = np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max() < xtol
        
        if PLOT:
            import matplotlib.pyplot as plt
            plt.figure(244459)
            plt.plot(np.arange(Nmax+1)*dt, xplot_all) #Nmax+2
            plt.title('Converged to sum of '+str(np.sum(xvec)))
            
            if save:
                    plt.savefig(save+'.png')
            plt.show()
            plt.close()
        
        
        return xvec, avg_dx


##############################################################        

class SSN2DTopoV1_ONOFF(_SSN_Base):
    _Lring = 180

    def __init__(self, ssn_pars, grid_pars,  conn_pars, filter_pars, J_2x2, gE, gI, sigma_oris =None, s_2x2 = None, ori_map = None, number_phases = 2, **kwargs):
        self.Nc = grid_pars.gridsize_Nx**2 #number of columns
        Ni = Ne = 2 * self.Nc 
        
        n=ssn_pars.n
        k=ssn_pars.k
        tauE= ssn_pars.tauE
        self.tauE = tauE
        self.tauI = tauI
        tauI=ssn_pars.tauI
        self.phases = ssn_pars.phases
        tau_vec = np.hstack([tauE * np.ones(self.Nc), tauI * np.ones(self.Nc)])
        tau_vec = np.kron(np.array([1,1]), tau_vec )
        
  
        super(SSN2DTopoV1_ONOFF, self).__init__(n=n, k=k, Ne=Ne, Ni=Ni,
                                    tau_vec=tau_vec, **kwargs)
        
        self.grid_pars = grid_pars
        self.conn_pars = conn_pars
        self._make_retinmap()
        
        if ori_map==None:
            self.ori_map = self._make_orimap()
        else:
            self.input_ori_map(ori_map)
        
        self.gE, self.gI = gE, gI
        

       
        self.edge_deg = filter_pars.edge_deg
        self.sigma_g = filter_pars.sigma_g
        self.k = filter_pars.k
        self.conv_factor =  filter_pars.conv_factor
        self.degree_per_pixel = filter_pars.degree_per_pixel
        
        self.A=ssn_pars.A

        #Create Gabor filters
        self.gabor_filters, self.A = self.create_gabor_filters()
        
            
        
        #if conn_pars is not None: # conn_pars = None allows for ssn-object initialization without a W
            
        self.make_W(J_2x2, s_2x2, sigma_oris)

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
    
    def input_ori_map(self, ori_map):
        self.ori_map= ori_map
        self.ori_vec = np.tile(self.ori_map.ravel(), (self.phases*2,))
        self._make_distances()
        self._make_retinmap()

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
        
        self.x_vec = np.tile(X.ravel(), (self.phases*2,))
        self.y_vec = np.tile(Y.ravel(), (self.phases*2,))
      
        
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
        #key = random.PRNGKey(87)
        #numpy.random.seed(6)
        for j in range(nn):
            kj = np.array([np.cos(j * np.pi/nn), np.sin(j * np.pi/nn)]) * 2*np.pi/(hyper_col)
            
            ## JAX CHANGES ##
            #key, subkey = random.split(key)
            #sj = 2 *random.randint(key=key, shape=[1,1], minval=0, maxval=2)-1 #random number that's either + or -1.
            #key, subkey = random.split(key)
            #phij = random.uniform(key, shape=[1,1], minval=0, maxval=1)*2*np.pi
            
            #NUMPY RANDOM
            sj = 2 * numpy.random.randint(0, 2)-1 #random number that's either + or -1.
            phij = numpy.random.rand()*2*np.pi

            tmp = (X*kj[0] + Y*kj[1]) * sj + phij
            z = z + np.exp(1j * tmp)


        # ori map with preferred orientations in the range (0, _Lring] (i.e. (0, 180] by default)
        self.ori_map = (np.angle(z) + np.pi) * SSN2DTopoV1_ONOFF._Lring/(2*np.pi)
        self.ori_vec = np.tile(self.ori_map.ravel(), (4,))

        return self.ori_map

    def _make_distances(self):
        Lx = Ly = self.grid_pars.gridsize_mm
        absdiff_ring = lambda d_x, L: np.minimum(np.abs(d_x), L - np.abs(d_x))
        cosdiff_ring = lambda d_x, L: np.sqrt(2 * (1 - np.cos(d_x * 2 * np.pi/L))) * L / 2/ np.pi
        PERIODIC = self.conn_pars.PERIODIC
        if PERIODIC:
            absdiff_x = absdiff_y = lambda d_x: absdiff_ring(d_x, Lx + self.grid_pars.dx)
        else:
            absdiff_x = absdiff_y = lambda d_x: np.abs(d_x)
        
        xs = np.reshape(self.x_vec, (self.phases*2, self.Nc, 1)) # (cell-type, grid-location, None)
        ys = np.reshape(self.y_vec, (self.phases*2, self.Nc, 1)) # (cell-type, grid-location, None)
        oris = np.reshape(self.ori_vec, (self.phases*2, self.Nc, 1)) # (cell-type, grid-location, None)
        
        # to generalize the next two lines, can replace 0's with a and b in range(2) (pre and post-synaptic cell-type indices)
        xy_dist = np.sqrt(absdiff_x(xs[0] - xs[0].T)**2 + absdiff_y(ys[0] - ys[0].T)**2)
        ori_dist = cosdiff_ring(oris[0] - oris[0].T, SSN2DTopoV1_ONOFF._Lring)
        self.xy_dist = xy_dist
        self.ori_dist = ori_dist

        return xy_dist, ori_dist  

    

    
    
    def create_gabor_filters(self):
        
        #Create array of filters
        e_filters=[] 
        if self.phases==4:
            e_filters_pi2 = []

        #Iterate over SSN map
        for i in range(self.ori_map.shape[0]):
            for j in range(self.ori_map.shape[1]):
                gabor=GaborFilter(x_i=self.x_map[i,j], y_i=self.y_map[i,j], edge_deg=self.edge_deg, k=self.k_filt, sigma_g=self.sigma_g, theta=self.ori_map[i,j], conv_factor=self.conv_factor, degree_per_pixel=self.degree_per_pixel)
                
                e_filters.append(gabor.filter.ravel())
                
                if self.phases==4:
                    gabor_2 = GaborFilter(x_i=self.x_map[i,j], y_i=self.y_map[i,j], edge_deg=self.edge_deg, k=self.k_filt, sigma_g=self.sigma_g, theta=self.ori_map[i,j], conv_factor=self.conv_factor, degree_per_pixel=self.degree_per_pixel, phase = np.pi/2)
                    e_filters_pi2.append(gabor_2.filter.ravel())
                
                        
        #i_constant= gI / gE
        e_filters_o =np.array(e_filters)
        e_filters = self.gE * e_filters_o
        i_filters = self.gI * e_filters_o

        #create filters with phase equal to pi
        e_off_filters = - e_filters
        i_off_filters = - i_filters
        

        if self.phases ==4:
            e_filters_o_pi2 =np.array(e_filters_pi2)

            e_filters_pi2 = self.gE * e_filters_o_pi2
            i_filters_pi2 = self.gI * e_filters_o_pi2

            #create filters with phase equal to -pi/2
            e_off_filters_pi2 = - e_filters_pi2
            i_off_filters_pi2 = - i_filters_pi2
            SSN_filters=np.vstack([e_filters, i_filters, e_filters_pi2, i_filters_pi2,  e_off_filters, i_off_filters, e_off_filters_pi2, i_off_filters_pi2])
        
        
        else:
            SSN_filters=np.vstack([e_filters, i_filters, e_off_filters, i_off_filters])
            
        
        if self.A == None:
            A= find_A(return_all =False, k=self.k_filt, sigma_g=self.sigma_g, edge_deg=self.edge_deg,  degree_per_pixel=self.degree_per_pixel, indices=np.sort(self.ori_map.ravel()))
            self.A = A
            
        
        #Normalise Gabor filters
        SSN_filters = SSN_filters*self.A
        
        if self.phases ==4:
            if self.A2 ==None:
                A2 = find_A(return_all =False, k=self.k_filt, sigma_g=self.sigma_g, edge_deg=self.edge_deg,  degree_per_pixel=self.degree_per_pixel, indices=np.sort(self.ori_map.ravel()), phase = np.pi/2)
                self.A2 = A2
                
                
            SSN_filters=np.vstack([e_filters*self.A, i_filters*self.A, e_filters_pi2*self.A2, i_filters_pi2*self.A2,  e_off_filters*self.A, i_off_filters*self.A, e_off_filters_pi2*self.A2, i_off_filters_pi2*self.A2])
        
        #remove mean so that input to constant grating is 0
        SSN_filters = SSN_filters - np.mean(SSN_filters, axis=1)[:, None]
        self.gabor_filters = SSN_filters

        return SSN_filters, self.A
    
    def select_type(self, vec, map_number):
        out = vec[(map_number-1)*self.Nc:map_number*self.Nc]
        return out
    
    def select_type_old(self, vec, select='E_ON'):
    
        assert vec.ndim == 1
        maps = self.vec2map(vec)

        if select=='E_ON':
            output = maps[0]

        if select =='I_ON':
            output=maps[1]

        if select == 'E_OFF':
            output=maps[2]

        if select == 'I_OFF':
            output = maps[3]
    
        return output
    

    def apply_bounding_box(self, vec, size = 3.2, select=1):
        
        Nx = self.grid_pars.gridsize_Nx
        #map_vec = self.select_type(vec, select)
        map_vec = self.select_type(vec, map_number = select).reshape(Nx,Nx)

        size = int(size / (self.grid_pars.dx)) +1

        start = int((self.grid_pars.gridsize_Nx - size) / 2)   
        
        map_vec = jax.lax.dynamic_slice(map_vec, (start, start), (size, size))
        #map_vec = map_vec[start:start+size, start:start+size]

        return map_vec
    





class SSN2DTopoV1_ONOFF_local(SSN2DTopoV1_ONOFF):

    def __init__(self, ssn_pars, grid_pars,  conn_pars, filter_pars, J_2x2, gE, gI, ori_map = None, **kwargs):
        self.phases = ssn_pars.phases
        self.Nc = grid_pars.gridsize_Nx**2 #number of columns
        Ni = Ne = self.phases * self.Nc 
        n=ssn_pars.n
        
        self.k=ssn_pars.k
        tauE= ssn_pars.tauE
        tauI=ssn_pars.tauI
        
        tau_vec = np.hstack([tauE * np.ones(self.Nc), tauI * np.ones(self.Nc)])
        tau_vec = np.kron(np.ones((1, self.phases)), tau_vec ).squeeze()
        self.tauE = tauE
        self.tauI = tauI
        #tau_vec = np.kron(np.array([1,1]), tau_vec )
  
        super(SSN2DTopoV1_ONOFF, self).__init__(n=n, k=self.k, Ne=Ne, Ni=Ni,
                                    tau_vec=tau_vec, **kwargs)
        
        self.grid_pars = grid_pars
        self.conn_pars = conn_pars
        
        self._make_retinmap()
        if ori_map==None:
            self.ori_map = self._make_orimap()
        else:
            self.input_ori_map(ori_map)
            
        self.gE, self.gI = gE, gI
        
        #Gabor filter parameters
        self.edge_deg = filter_pars.edge_deg
        self.sigma_g = filter_pars.sigma_g
        self.k_filt = filter_pars.k
        self.conv_factor =  filter_pars.conv_factor
        self.degree_per_pixel = filter_pars.degree_per_pixel
        
        self.A=ssn_pars.A
        if ssn_pars.phases==4:
            self.A2 = ssn_pars.A2
                
        self.gabor_filters, self.A = self.create_gabor_filters()
        
        self.make_local_W(J_2x2)
        

    def drdt(self, r, inp_vec):
        r1 = np.reshape(r, (-1, self.Nc))
        out = ( -r + self.powlaw(np.ravel(self.W @ r1) + inp_vec) ) / self.tau_vec
        return out
    
    def make_local_W(self, J_2x2):
        #self.W = np.kron(np.ones((2,2)), np.asarray(J_2x2))
        self.W = np.kron(np.eye(self.phases), np.asarray(J_2x2))
        