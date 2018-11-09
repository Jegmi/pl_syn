#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 16:14:43 2017

@author: jannes
"""
from __future__ import print_function


#import argparse
import torch as T
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#from torchvision import datasets, transforms
#from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from time import time

#from scipy.stats import multivariate_normal as mvn
#from scipy.stats.distributions import norm as npNorm
#import pymc3 as pm
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
#from torch.distributions.binomial import Binomial as Bino
from torch.distributions.normal import Normal as Normal
#from pymc3.distributions.multivariate import MvNormal
#from numpy.random import multivariate_normal as mvnrdm
#import scipy as sp
import matplotlib
import copy

from helpers.helpers import save_obj, load_obj

import itertools as it

import pandas as pd

#clear_all()

import os
import sys
# add root folder to import path
sys.path.append(os.getcwd() + '/../')


from scipy.integrate import RK45
# fun, t0, y0, t_bound, max_step=inf, rtol=0.001, atol=1e-06, vectorized=False
# take RK: re-use old time step. No wrapper overhead.


from helpers.helpers import smooth, plt_idxs, eps_lin, save_obj
from my_plotters.my_plotters import plt_filter,my_savefig

matplotlib.rcParams.update({'font.size': 14})

np.random.seed(43)
T.manual_seed(70)


#a = -10
#b = 10
#
#funs = [phi_dash_over_phi, 
#        x_minus_phi_dash_over_phi,
#        lambda x: -phi_dash_over_phi(x)*x_minus_phi_dash_over_phi(x),
#        lambda x: -phi_dash_over_phi(x)*S.g(x)*x]
#
#nams = ["$f(z) = (\Phi'/\Phi)(z)$",
#        "$z + f(z)$",
#        "$-f(z) \cdot (z+f(z))$",
#        "$-f(z) \cdot \Phi(z) \cdot z$"
#        ]
#
#xplt = np.linspace(a,b,99)
#for fun,tag in zip(funs,nams):
#    yplt = [float(fun(T.tensor([x]))) for x in xplt]
#    plt.plot(xplt,yplt,label=tag)
#    
#    
#    
#plt.ylim([-1,-a])
#plt.gca().legend()
#plt.gca().grid()
#plt.xlabel('z')
#plt.savefig('function.pdf',dpi=400)
#def erfcx(x):
#    # bad
#    if x < 5:
#        out = sp.special.erfc(x) * np.exp(x*x)
#    else:
#        y = 1. / x
#        z = y * y
#        s = y*(1.+z*(-0.5+z*(0.75+z*(-1.875+z*(6.5625-29.53125*z)))))
#        out = s * 0.564189583547756287
#        
#    return out



    
def wrap_run(dim,para,bayesian,step_max,dt=0.005,pars_extra={},
             cov0=None,hidden=None, gen_hidden=False,prefix='',
             gen_plot_arrays=False,plt_num=3,mdims=None,init_dict=None):
    """ create a synapse, run it, plot it OR generate hidden from it """
    # init synapse
    if bayesian:
        name = prefix + para + '_b_'
    else: 
        name = prefix + para + '_c_' + str(cov0)[:6]
                
    syn = Synapse(name=name, para=para, dim=dim, steps=step_max, dt=dt, mdims=mdims)

    # init parameters    
    pars_extra.update({'bayesian':bayesian})
    
    syn.set_pars(pars_user=pars_extra)

    # init initial conditions    
    if cov0:
        if init_dict is not None:
            init_dict['s2'] = cov0
        else:
            init_dict={'s2':cov0}
        syn.set_vars(init_dict=init_dict)                            
    else:
        if init_dict is not None:
            syn.set_vars(init_dict=init_dict)
        else:
            syn.set_vars()        
                                                    
    total_steps = syn.pars['xSteps']*syn.pars['step_out']
    steps_per_hour = 2*10**5 # 216000 <- Cluster
    
    print('Computation steps:',total_steps)
    print('Output steps:',syn.pars['xSteps'])
    print('Expected simulation time [h]:',total_steps/steps_per_hour)
    print('Simulated time [min]:',total_steps*dt/60)
    print('tau OU in [min]:',syn.pars['OU']['tau']/60)        
                    
    ## run simulation
    if gen_hidden:
        # generat hidden

        print('generate hidden')
        if hidden is not None: 
            print('warning: ignoring the provided hidden')
        syn.run(run_synapse=False)
        
        #save_obj(syn.export_hidden())        
        return(syn.export_hidden())
    
    else:    
        # run with hidden
        if hidden is not None:
            syn.run(hidden=hidden)
        else:            
            print('run without hidden')
            syn.run()
    
        # evaluate it
        syn.get_MSE()
        
        if gen_plot_arrays:

            traj = syn.export_hidden(keys_readout=('w','m','s2'))
            for k,v in traj.items():                
                traj[k] = np.array(v)
            save_obj(traj,'arr_'+name)
        else:
            if plt_num > 0:
                if 'mdims' in syn.pars:
                    # use old plotting dims                
                    dims = syn.pars['mdims']
                else: # generate new plotting dims
                    dims = plt_idxs(plt_num=plt_num,dim = syn.pars['dim'])
                                
                for dim in dims:
                    syn.plt('wmq',dim,save=True)
                    plt.show()
                    plt.close()
                    
                    syn.plt('wi MSE',dim,save=True)                
                    plt.show()
                    plt.close()
                    
                syn.plt('w MSE',save=True)                
                plt.show()
                plt.close()
                #syn.wrap_plt(keys= ['w'], dims = dims)                
        return(syn)
            

        # TODO: use cumulated mean
        #b = np.cumsum(a[::-1])/np.arange(1,len(a)+1)
        
        
def gen_spike_times(protocol,t_wait):
    """ turn protocol into spike time array """
    bnum = int(protocol['burst_num'])
    
    # spike pairs per burst (eg 5 or 1)
    n = int(protocol['pair_count_max'])

    # distance within a burst (eg 50ms -> 20 Hz)
    Tb = protocol['pair_distance']
    
    # delta_T [s] (STDP distance eg 5ms)
    dT = protocol['delta_T']
            
    # spike times in burst  
    tf_b = np.arange(n)*Tb
    
#    print(protocol)
#    print(tf_b)
    
    # for all bursts with intermediate time 
    tfs = np.concatenate([tf_b + (b_idx+1)*t_wait for b_idx in range(bnum)])
            
    # all spike times, with final dummy spike
    tfs = np.concatenate([tfs,
                          tfs + np.abs(dT),
                          np.array([tfs[-1]+np.abs(dT)+t_wait])
                          ])                
    
    return(np.sort(tfs))
        
        
class partial_tau_of_xeps():
    """ class to compute derivative of x_epsilon with respct to tau_u """
    
    def __init__(self):
        self.spike_times = []
        self.vals = []
        
    def add_spike(self,tf):
        self.spike_times.append(tf)
        
    # also acts as function
    def __call__(self,t,tau):
        # compute convl for all firing times
        if len(self.spike_times) == 0:
            return 0
        else:
            # time between past firing and now
            dT_over_tau = (t-np.array(self.spike_times))/tau
                        
            # remove last firing time if outdated
            if dT_over_tau[-1] > 5:
                self.spike_times.pop(-1)
            
            # compute a value based on list
            return(np.sum(dT_over_tau/tau*np.exp(-dT_over_tau)))        
        
# =============================================================================
#       Synaptic filter class        # 
# =============================================================================
        
class Synapse():
    """ create a filter 
    Attr:
        arr mu: estimated hidden
        arr cov: hidden covariance
        dic priorPar: parameters of the process
    """
            
    def __init__(self, name, para, dim=2, steps=1000, dt=0.01, mdims=None):
        """Return a Bayesian Synapes object. Here we only query the parameters 
           that remain constant during the life of the synapse. To run the obj,
           the full init must take place.
        
        """
                
        ##### General read in
        
        # hack to cope with dim=1. Init everything with dim two, 
        # and block second dim when computing u, ubar and u_sig2.
        if dim == 1:            
            dim = 2
            self.dim_is_1 = True
        else:
            self.dim_is_1 = False
            
        # init parameter dict with constant parameters        
        self.pars = {'dim':dim, 'xSteps':steps, 'para':para, 'dt':dt}
        if mdims is not None:
            self.pars.update({'mdims':mdims})
                
        # standard noise of synaptic dimension
        self.eta_dim = MVN(T.zeros(self.pars['dim']),T.eye(self.pars['dim']))
        # for evolution of membrane pot.
        self.eta_0 = Normal(0,1)
        self.eta_fun = lambda dim: MVN(T.zeros(dim),T.eye(dim)).sample()

        # time steps
        self.xSteps = self.pars['xSteps']                                
        self.size = [self.pars['dim'],self.pars['xSteps']]

        self.k = 0
        # for sparse output
        self.k_out = 0     
        
        # for read out
        self.k_readout = ['w','Sx','Sy','u','g','eta']
        # add sampled weights
         

        self.errorLog = []

        self.para = self.pars['para']
        self.name = name        
        self.title = None
            
        ##### short hands
        
        self.g = lambda x : T.erf(x*0.7071067811865475)*0.5 + 0.5
        
#        cut = -5.5*T.ones(dim)
#        self.gdash_over_g = lambda x: T.exp(Normal(0,1).log_prob(T.max(cut,x)))/self.g(T.max(cut,x))
#        self.gdash_over_g = lambda x: T.exp(Normal(0,1).log_prob(x))/g(x)
                        
        # results
        self.res = {}        
        self.sh = {}
                
        ##### variables
        # dictionary with all output quantities for all learning rules
        self.vars = {}
        self.out = {}
        if 'mdims' in self.pars: # memory dimensions            
            self.mvars = {}
            self.mout = {}
            self.init_containers(self.size, mdims=self.pars['mdims'])
        else:
            self.init_containers(self.size)
                        
        self.size_in_MB = self.approx_size()
        
    def _x_minus_phi_dash_over_phi(self,x):
        return(x + self._phi_dash_over_phi(x))
    
    
    def _phi_dash_over_phi(self,x):
        """     
            sqrt(2/pi)*erfcx(-x/sqrt(pi))
        
        The approximation to get erfcx comes from a c library:
                erfcx(x)
        
        /*
               computes exp(x^2)*erfcc(x). only gives reasonable answers if x
               is positive.
        
               note:  int_y^inf dz exp(-z*z/2) / sqrt(2*pi) = (1/2) erfcc(y/sqrt(2))
        */
        """
    
        z=T.abs(x*0.70710678118654746)
        t=1.0/(1.0+0.5*z)
        ans=t*T.exp(-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+
                t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+
                t*(-0.82215223+t*0.17087277)))))))))

        out = ans
        
        ii = x > 0.0
        out[ii] = 2.0*T.exp(z[ii]*z[ii]) - ans[ii]
                                
        return(0.79788456080286541/out)                                        


    def set_pars(self,pars_user={}):
        """ set many defaults and only query via pars_user """
        # was already initalised in __init__
        pars = self.pars        

        # number_of_constant_parameters_to_be_changed
        dN = len({'dim','xSteps','dt','para'}.intersection(set(pars_user)))
        if dN > 0:
            print('Warning: updated globals: synapse containers are mismatched')

        # include new dictionary
        pars.update(pars_user)
                
        # then test if all important quantites were defined
        
        if not 'testing' in pars:
            pars['testing'] = False
        if pars['testing']==True:
            print("At hastags #NOWJ, I'm using testing=True")
        
        
        
        ## for plotting
        if not 'cols' in pars:
            pars['cols']=['orange','blue']
            
        if not 'labels' in pars:
            pars['labels'] = ['True','Estimated']

        # plot options
        if not 'plot-flags' in pars:
            pars['plot-flags'] = {}
            
        if 'all-trajectories' not in pars['plot-flags']:
            pars['plot-flags']['all-trajectories'] = False

            
        if not 'save-output' in pars['plot-flags']:
            pars['plot-flags']['save-output'] = False
        
        if not 'save-array' in pars['plot-flags']:
            pars['plot-flags']['save-array'] = False
        
        
        if 'time_between_output' not in pars:
            pars['time_between_output'] = 1800
        
        
        # bayesian or classic synapse?
        if not 'bayesian' in pars:
            pars['bayesian'] = True
          
        # bias or not?    
        if not 'bias-active' in pars:
            if pars['para'] == 'gauss-exp':
                pars['bias-active'] = True

            elif pars['para'] == 'Peter':
                pars['bias-active'] = False
                
            elif pars['para'] == 'logNorm-sigm':
                pars['bias-active'] = False
                    
            elif pars['para'] == 'logNorm-lin':
                pars['bias-active'] = False

        if 'eps_smooth' not in pars:
            pars['eps_smooth'] = False

        if 'mu_bounds' not in pars:
            # in terms of std
            pars['mu_bounds'] = None


        if 'step_out' not in pars:
            pars['step_out'] = 0

        # Peter Latham's ideas
        if 'PL' not in pars:
            # V* - V generates spikes (rather than just V*)
            pars['PL'] = {}
            
        if 'ON' not in pars['PL']:                    
            pars['PL']['ON'] = False
        
        if 'k_samp' not in pars['PL']:
            pars['PL']['k_samp'] = 0.0877
        
        if 'opt' not in pars['PL']:                    
            pars['PL']['opt'] = 1
                
        # expected spike train <g> is computed via sampling rather than mean mu
        if 'Sample' not in pars['PL']:
            pars['PL']['Sample'] = False        

        if 'alpha' not in pars['PL']:
            pars['PL']['alpha'] = 1

        if 'beta' not in pars['PL']:
            pars['PL']['beta'] = 0

        if 'b_samp' not in pars['PL']:
            pars['PL']['b_samp'] = True



        # bias or not?    
        if not 'g0' in pars:
            if pars['para'] == 'gauss-exp':
                pars['g0'] = 1
                
            elif pars['para'] == 'logNorm-sigm':
                pars['g0'] = 50

            elif pars['para'] == 'Peter':
                pars['g0'] = 10

                    
            elif pars['para'] == 'logNorm-lin':
                pars['g0'] = 50
         

        ## membrane noise
        # intial sigmoidal width
        if not 'sig0_u' in pars:
            pars['sig0_u'] = 1 # 
        
        if not 'tau_u' in pars:
            pars['tau_u'] = 0.03

        if not 'tau_running_ave' in pars:
            pars['tau_running_ave'] = 1 # [sec]


        ## input firing
        
            # x firing rates
#            sig_x_rates = (0.5*np.log(10))
#            # from big to small rates 
#            pars['nu'] = (np.sort(np.exp(sig_x_rates*np.random.randn(dim))))[::-1]
        
        # load if provided
        if 'nu_high' not in pars:
            pars['nu_high'] = 8
        if 'nu_low' not in pars:
            pars['nu_low'] = 8
        
        if 'nu' not in pars:                                    
            # create input rates (since its binary, take care of uneven dim)            
            dim = pars['dim']
            nu_input = np.sort(np.exp(0.5*np.log(10)*np.random.randn(dim)))[::-1]
            
            #nu_input = np.repeat([pars['nu_low'],pars['nu_high']],
            #                                 np.ceil(pars['dim']/2))
            
            # remove extra elemented added in the case of uneven dim
            #nu_input = nu_input if pars['dim'] % 2 == 0 else nu_input[1:] 
                        
            if pars['bias-active']:
                nu_input[-1] = 1/pars['dt']

            # init rates as tensor, and put bias first, then higher rates.
            #nu_input[::-1] = nu_input
            pars['nu'] = nu_input
        
        if pars['bias-active']:
            pars['nu'][0] = 1/pars['dt']
            
                                
        ## Weights        
        if not 'sign' in pars:
            # init as only ones -- but could be random as well
            pars['sign'] = T.tensor(
                    (1)**np.random.binomial(1,0.5,self.pars['dim']),
                            dtype=T.float32)

        # OU
        if not 'mu_ou' in pars:
            # set OU mean for each method
            
            if pars['para'] == 'gauss-exp':
                # scaling
                pars['mu_ou'] = 0 / pars['dim']

            elif pars['para'] == 'Peter':
                pars['mu_ou']= -0.669
                
            elif pars['para'] == 'logNorm-sigm':
                # scaling
                pars['mu_ou'] = -0.669

            elif pars['para'] == 'logNorm-lin':
                # scaling
                pars['mu_ou'] = -0.669
                            
            
        if not 'sig2_ou' in pars:
            # set ou variance for reach method
            
            if pars['para'] == 'gauss-exp':
                pars['sig2_ou']= (0.863)**2/pars['dim']

            elif pars['para'] == 'Peter':
                pars['sig2_ou']= (0.863)**2

                        
            elif pars['para'] == 'logNorm-sigm':
                # scaling
                pars['sig2_ou'] = (0.863)**2

            elif pars['para'] == 'logNorm-lin':
                # scaling # TODO: scaling = np.log(pars['dim'])    ???
                pars['sig2_ou'] = (0.863)**2
                

        # spike response kernl
        if 'y-rsp' not in pars:
            pars['y-rsp'] = {}

        if 'ON' not in pars['y-rsp']:
            pars['y-rsp']['ON'] = False

        if 'tau' not in pars['y-rsp']:
            pars['y-rsp']['tau'] = [0.01,-0.05] 
        pars['y-rsp']['tau'] = T.tensor(np.array(pars['y-rsp']['tau']),
                                                            dtype=T.float32) 

        if 'a0' not in pars['y-rsp']:
            # amplitude of kernl
            pars['y-rsp']['a0'] = 0.05
            
        # normalise kernls (now a vector)
        pars['y-rsp']['a0'] = pars['y-rsp']['a0']/pars['y-rsp']['tau']
                        
                
        if not 'OU' in pars:
            # set up OU
            pars['OU'] = {}
                                                
        if not 'mu' in pars['OU']:
            pars['OU']['mu'] = T.ones_like(pars['sign'])*pars['mu_ou']
        if not 'sig2' in pars['OU']:
            pars['OU']['sig2'] = T.ones_like(pars['sign'])*pars['sig2_ou']
            
        if not 'tau' in pars['OU']:
            pars['OU']['tau'] = 10**2 # [sec]
            
        if not 'flag' in pars['OU']:
            pars['OU']['flag'] = True
                    
        ## Neuron offset
        if not 'th' in pars:
            # set basline and max firing rate [Hz] of output neuron respectively 
            if pars['para'] == 'gauss-exp':
                pars['th'] = 0 
                
            elif pars['para'] == 'logNorm-sigm':
                # TODO:
                # try to scale with synapse .. but maybe this should be 0?
                pars['th'] = 0 #- pars['dim']**(1+pars['mu_ou']
                        #)*np.mean(np.array(pars['nu']))*pars['dt']
        
            elif pars['para'] == 'logNorm-lin':
                pars['th'] = 0.01
            
            elif pars['para'] == 'Peter':
                pars['th'] = 0.0
            
            
        # create short hands from new paramters
        self.load_sh()

        return(pars)            


    
    def set_vars(self,protocol=None,init_dict=None,k=0):
        """ generate initial for:
            w0 (m), cov0 (s2), u0, sig2_u0        
            
            three use cases: 
                (1) load from protocol
                (2) specify in init_dict     
                (3) load according to pars
        """
        # short hand
        pars = self.pars
        # important: this resets the internal step index k
        self.k = k

        
        ## init filter
        
        # from protocol
        if protocol != None:
            
            self.vars['m'][:,k] = self.pars['sign']*protocol['mu0']
            self.vars['s2'][:,k] = T.abs(self.pars['sign'])*protocol['var0']
            
            # send out warning if overdetermined
            if init_dict != None:
                if ('m' in init_dict) or ('s2' in init_dict):
                    print('Warning: Ignore init_dict[m,s2] and use protocol instead')

        # load m,s2 from dict
        elif init_dict != None:                        
            if 'm' in init_dict: 
                self.vars['m'][:,k] = self.pars['sign']*init_dict['m']
                
            if 's2' in init_dict:
                self.vars['s2'][:,k] = T.pow(self.pars['sign'],2)*init_dict['s2']
            
        # load m,s2 from OU
        else:
            if pars['para'] in ('gauss-exp','Peter'):
#                print(pars['OU'])
#                print('')
#                print(self.vars['s2'][:,k])
                self.vars['m'][:,k] = pars['OU']['mu']            
                self.vars['s2'][:,k] = pars['OU']['sig2']
            else:
                # setting it to the true value of wstar = exp(lambda) 
                # not setting the filter to its mean which is exp(mu + 0.5sig2)
                self.vars['m'][:,k] = T.exp(pars['OU']['mu'])
                self.vars['s2'][:,k] = T.pow(self.vars['m'][:,k],2)*(
                                            T.exp(pars['OU']['sig2'])-1)
        
        # other quantitites from dict or from user:
        
        # to check more quickly:
        if init_dict == None:
            init_dict = {}
                            
        #load u (otherwise set zero)
        self.out['u'][k] = init_dict['u'] if 'u' in init_dict else 0

        # load ubar or set to u0
        self.out['ubar'][k] = init_dict['ubar'] if 'ubar' in init_dict else self.out['u'][k]

        # load sig2_ u otherwise set to mean of filter            
        self.out['sig2_u'][k] = init_dict['sig2_u'] if 'sig2_u' in init_dict else T.mean(self.vars['m'][:,k])
                           
        # set weights, else init as filter
        if 'w' in init_dict:
            self.vars['w'][:,k] = init_dict['w'] 
        else:
            print('no initial w provided. Sample from N(m,s).')
            self.vars['w'][:,k] = self.vars['m'][:,k] + T.sqrt(
                                self.vars['s2'][:,k])*self.eta_dim.sample()
        
        # set releases
        if self.para == 'Peter':
            self.vars['w_r'] = T.exp(self.vars['m'][:,k] + T.sqrt(
                                self.vars['s2'][:,k])*self.eta_dim.sample()) #T.exp(self.vars['w'][:,k])             
            
        else: 
            # JANNES
            idx_neg = self.vars['w'][:,k] <= 0            
            if T.sum(idx_neg)>0:
                print('make positive here! and make m positive as well')
                self.vars['w'][idx_neg,k] = -1*self.vars['w'][idx_neg,k]
                
            #self.vars['w'][:,k] = T.abs(self.vars['w'][:,k])
            self.vars['w_r'] = self.vars['w'][:,k]
            
            
            
        # old snippet
        #    init_cond['mu0'] = T.distributions.uniform.Uniform(-0.5,0.5).sample(
        #                        pars['sign'].shape)*pars['sign']



        
        
# =============================================================================
#     Function for init variables
# =============================================================================        
        
    def _get_bits_of_torch_type(self,dtype):
        """ string to number """
        dtype2bit = {
                'torch.float32': 32, 'torch.float': 32,
                'torch.float64': 64, 'torch.double': 64,
                'torch.float16': 16, 'torch.half' : 16,
                'torch.uint8' : 8,	'torch.ByteTensor' : 8,
                'torch.int16' : 16, 'torch.short' : 16,
                'torch.int32' : 32, 'torch.int' : 32,
                'torch.int64' : 64, 'torch.long': 64
                }
        return(dtype2bit[str(dtype)])
        
        
    def approx_size(self):
        """ after init containers this fct estimates memory allocation """
        total_size = 0

        for v in self.vars.values():
            
            bits = self._get_bits_of_torch_type(v.dtype)
            num_entries = float(T.prod(T.tensor(v.size())))
        
            total_size += bits*num_entries

        for v in self.out.values():
            
            bits = self._get_bits_of_torch_type(v.dtype)
            num_entries = float(T.prod(T.tensor(v.size())))
        
            total_size += bits*num_entries
        
        total_size_MB = total_size*1.25*10**(-7)
        
        print('Expected MB demand:',total_size_MB)
        return(total_size_MB)
                
        
    def init_containers(self,size,mdims=None):
        """ set up the desired variables based on paradigm
            if arg memory mdims provided: disregarda all other time series
        """
        
        dim, xSteps = size
        if mdims is not None:
            # for convenient usage: store an __iter__
            if isinstance(mdims,int):
                mdims = [mdims]
            self.pars['mdims'] = mdims
            # init all variables
            self.init_containers([dim,2])
            
            # init memory containers for mem-dimensions
            for var in ('m','s2','w'):
                self.mvars[var] = T.zeros([len(mdims),xSteps],dtype=T.float32)
            
            for out in ('w MSE','q'): # q: procentage of within sig range            
                self.mout[out] = T.zeros(xSteps,dtype=T.float32)
                                                
        else:                                             
            ## INIT PRE CONTAINERs
            
            # true weights
            self.vars['w'] = T.zeros(size)
            self.pars['sign'] = T.ones(self.pars['dim'],dtype=T.float32)
        
            # spike trains in 
            self.vars['Sx'] = T.zeros(size,dtype=T.uint8)
                        
            # presynaptic actiavation
            self.vars['x'] =  T.zeros(self.pars['dim'])
            
            # presynaptic activation, 2d model             
            self.vars['xdot'] =  T.zeros(self.pars['dim'])
            
            # presynaptic release: sampled from posterior
            self.vars['w_r'] =  T.zeros(self.pars['dim'])
            
            # k value: m_i/s_i^2 = k_i
            self.vars['k'] =  T.zeros(self.pars['dim'])
            
            # learning rate
            self.vars['eta'] = T.zeros(size)

                        
            # filtering: mean and variance
            self.vars['m'] = T.zeros(size)
            self.vars['s2'] = T.zeros(size)
            
            ## INIT POST containers
            
            # membrane potential, mean and variance and target
            self.out['u'] = T.zeros(xSteps)                        
            self.out['ubar'] = T.zeros(xSteps)            
            self.out['sig2_u'] = T.zeros(xSteps)

            # spike train
            self.out['Sy'] = T.zeros(xSteps,dtype=T.uint8)

            # running average 
            self.out['sig2_u_rm'] = T.zeros(xSteps)
            self.out['ubar_rm'] = T.zeros(xSteps)
            # excitation
            self.out['g'] = T.zeros(xSteps)
            self.out['gbar'] = T.zeros(xSteps)
            # saturation factor
            self.out['chi'] = T.zeros(xSteps)
            # error signal
            self.out['delta'] = T.zeros(xSteps)
            
            # Mean squared error (summed over synapses)
            self.out['w MSE'] = T.zeros(xSteps)
            
            # MSE for u
            self.out['u MSE'] = T.zeros(xSteps)
                        
            # refractory variable, init as vector
            self.out['a'] = T.zeros((2,xSteps))


            


            
    def load_sh(self):
        """ init a short hand dict """
        # short hand
        p = self.pars
        
        self.sh['dt_sqrt'] = T.sqrt(T.tensor(p['dt']))        
                
        # for computing du
        self.sh['dt/tau_u'] = p['dt']/p['tau_u']
        self.sh['1-dt/tau_u'] = 1 - p['dt']/p['tau_u']
        
        # running average membrane 
        self.sh['dt/tau_run'] = T.tensor(p['dt']/p['tau_running_ave'],
                                                           dtype=T.float32)
        
        # to save compt time
        self.sh['dt/tau_OU'] = p['dt']/p['OU']['tau']        

        # input firing probability
        self.sh['nu*dt'] = p['nu']*p['dt']

        # gain fct coef
        self.sh['g0*dt'] = p['dt']*p['g0']
        
        # constant vector in fron of Wiener process: sigma_OU
        self.sh['sig_pi'] = T.sqrt(2*p['OU']['sig2']/p['OU']['tau'])
        self.sh['c'] = p['OU']['mu'] + p['OU']['sig2']
        
        # eliminate negative times (used to indictate kernl directions)
        self.sh['1-dt/tau_a'] = 1-p['dt']/T.abs(p['y-rsp']['tau'])



# =============================================================================
#     Components of simulations
# =============================================================================

        

    def _prior_update(self,m,s2):
        """ compute prior update for mean and variance """
        
        # Bayesian synapses
        if self.pars['bayesian']:            
        
            # log normal prior
            if self.para in ('logNorm-sigm','logNorm-lin'):
                # efficieny (statistical measure
                gam = s2/T.pow(m,2)
    
                # mean         update
                dm = - self.sh['dt/tau_OU']*m*(
                        T.log(m)
                        + 0.5*T.log(1+gam) 
                        - self.sh['c'])
                
                # variance update
                ds2 = - 2*T.pow(m,2)*self.sh['dt/tau_OU']*(
                         gam*(T.log(m) - self.pars['OU']['mu'])
                        +(1 + 1.5*gam)*T.log(1+gam)
                        - self.pars['OU']['sig2'])
                                
            if self.para in ('gauss-exp','Peter'):
                                
                # prior
                dm = - (m - self.pars['OU']['mu'])*self.sh['dt/tau_OU']
                ds2 = - 2*(s2 - self.pars['OU']['sig2'])*self.sh['dt/tau_OU']
                                                
                
        # classical synapses do not have a prior
        else:
            dm, ds2 = (0,0)
            
        return(dm,ds2)
        
        
    def _likelihood_update(self,v,o):
        """ update based on likelihood """
        k = self.k

        # observerable                    
        y = T.tensor(o['Sy'][k],dtype=T.float32)

        # pre factor for peter
        alpha = self.pars['PL']['alpha'] 
        # from predictive coding        
        beta = self.pars['PL']['beta'] if self.pars['PL']['ON'] else 0


        if self.para is not 'Peter':
            # expected membrane potential
            if self.dim_is_1:
                # stupid hack because 1d arrays don't work well
                u = v['m'][0,k]*v['x'][0] 
            else:
                u = v['m'][:,k].dot(v['x']) 
        
        
            self.out['ubar'][k] = (alpha + beta)*u + self.pars['th']
                
            # spike response kernl
            if self.pars['y-rsp']['ON']:
                self.out['ubar'][k] += T.sum(self.out['a'][:,k])
                
                # update refractory variable            
                self.out['a'][:,k+1] = self.out['a'][:,k]*self.sh['1-dt/tau_a']
                if y == 1:                
                    self.out['a'][:,k+1] += self.pars['y-rsp']['a0']
            
            
            # only Bayesian synapses have an expected membrane variance
            if self.pars['bayesian']:
                # expected membrane noise
                if self.dim_is_1:
                    self.out['sig2_u'][k] = v['s2'][0,k]*(T.pow(v['x'][0],2))
                else:
                    self.out['sig2_u'][k] = v['s2'][:,k].dot(T.pow(v['x'],2))
                    
                # in Peter's predctive coding: cov[dV] = 0.5 cov[u] --> factor of 2 
                if self.pars['PL']['ON']:
                    self.out['sig2_u'][k] = 2*self.out['sig2_u'][k]
                    
            else: 
                self.out['sig2_u'][k] = 0
                      
        if self.para == 'Peter':
            # PETER
                
                        
            # compute mean and var of spiking rate:
            varx = T.tensor(self.pars['nu']*self.pars['tau_u']/2,
                                        dtype=T.float32)

            meanx2 = T.tensor(
                        np.power(self.pars['nu'],2)*self.pars['tau_u']**2,
                                      dtype=T.float32)
                              
            # compute log normal synapse from log variables
            if self.pars['bayesian']:
                M,S2 = self.get_weights()
            else:
                M = T.exp(v['m'][:,k])
                # NOT SURE ABOUT THIS ONE
                #JANNES
                    
            # compute V std: this could be done outside as well!                                   
            

            
            if not self.pars['testing']:
#                print('uncomment this!')
                if self.dim_is_1:
                    u = M[0]*v['x'][0]

                    if self.pars['bayesian']:
                        self.out['sig2_u'][k] = (alpha**2 + beta**2)*(
                                S2[0]*(varx + meanx2) + T.pow(M[0],2))
                                        
                else:
                    if self.pars['bayesian']:
                        self.out['sig2_u'][k] = (alpha**2 + beta**2)*(
                        S2.dot(varx + meanx2) + T.pow(M,2).dot(varx))
                    
                    u = M.dot(v['x'])            
                
                self.out['ubar'][k] = (alpha + beta)*u + self.pars['th']
    
                # NOTE: I have excluded the 'i' term.                                                     
                if self.pars['bayesian']:
                    sigV = T.sqrt(self.pars['sig0_u']**2 + self.out['sig2_u'][k])
                else: 
                    sigV = self.pars['sig0_u']
                    
                # expected activity
                if self.dim_is_1:            
                    V_vec = o['ubar'][k] + beta*(v['w_r'] - M[0])*v['x'][0]
                else: 
                    V_vec = o['ubar'][k] + beta*(v['w_r'] - M)*v['x']
#                
                    
                    
                    
            elif self.pars['testing']:
                # compute 
                self.out['ubar'][k] = M.dot(v['x'])*alpha + self.pars['th']  #+ beta*self.cur_noise 
                V_vec = self.out['ubar'][k]
                
                self.out['sig2_u'][k] = alpha**2*(T.pow(v['x'],2).dot(S2)) + beta**2
                sigV = T.sqrt(self.pars['sig0_u']**2 + self.out['sig2_u'][k])
                
                # next testing: same but w/ w_r time series.
                # cur_noise = w_r*x --> nothing else needs chance. 
                
                # next testing: same but w/ running average.
                                
            eq = {}

            # post synaptic factors
            z = V_vec/sigV            
            eq['Phi'] = self.g(z)
            eq['delta'] = y - eq['Phi']*self.sh['g0*dt']

            # pre synaptic factors
            eq['xi*sigma^2'] = v['s2'][:,k]*alpha*M*v['x']/sigV
            
            # mean update: sig^2*xi*phi'/phi*(y - phi)
            dm = eq['delta']*self._phi_dash_over_phi(z)*eq['xi*sigma^2']
            
            if self.pars['bayesian']:
                # var update:  sig^4*xi^2*(y*[phi''/phi - (phi'/phi)^2] - phi'')            
                ds2 = - T.pow(eq['xi*sigma^2'],2)*self._phi_dash_over_phi(z)*(
                    y*self._x_minus_phi_dash_over_phi(z) 
                    + eq['Phi']*z*self.sh['g0*dt'])

            # read outs for debugging 
                self.vars['eta'][:,k] = self.vars['w_r']                                
#            self.eq = eq
            
#            self.sigV = sigV
#            self.V_vec = V_vec
#            self.varx = varx
#            self.meanx2 = meanx2     
#            self.M = M
#            self.S2 = S2


            #if T.isnan(wr).sum().item() > 0:
             #           print(self.k,'w_r contains nan!')


#            if self.k == 0:
#                self.dm = []
#                self.extra = []
#            self.extra.append(M)
#            self.dm.append(np.array(dm))


            
        elif self.para == 'gauss-exp':
            
            # expected activity 
            gbardt = (T.exp(o['ubar'][k] + 0.5*o['sig2_u'][k])
                        *self.sh['g0*dt']).item()
        
            # saturation term
            if gbardt > 1:
                print(k,'gdt exploded:',gbardt)
                gbardt = 1
                self.out['chi'][k] = 1


            # get firing rate
            self.out['gbar'][k] = gbardt/self.pars['dt']
                
            # error signal
            self.out['delta'][k] = y - gbardt
    
            # learning rate factor
            self.vars['eta'][:,k] = v['s2'][:,k]*v['x']
            
            # updates: mean and variance
            dm = v['eta'][:,k]*o['delta'][k]
            
            if self.pars['bayesian']:
                ds2 = - T.pow(v['eta'][:,k]*o['gbar'][k],2)*self.pars['dt']                

        # sigmoidal update specifics
        elif self.para == 'logNorm-sigm':
            # sigmoidal width
            sig_V = T.sqrt(self.pars['sig0_u']**2 + np.pi/8*o['sig2_u'][k])
    
            # expected activity
            sigm = T.tanh(o['ubar'][k]/sig_V)*0.5 + 0.5        
            self.out['gbar'][k] = (sigm.item())*self.pars['g0']
    
            # saturation term
            self.out['chi'][k] = 1-sigm        
            
            # error signal
            self.out['delta'][k] = y - self.out['gbar'][k]*self.pars['dt']
    
            # learning rate factor
            self.vars['eta'][:,k] = v['s2'][:,k]*v['x']/sig_V
            
            # as in derivations
            xi = (1-sigm)*v['eta'][:,k]
            gam = v['s2'][:,k]/T.pow(v['m'][:,k],2)
            
            # update: mean
            dm = xi*o['delta'][k]
            
            # update: variance
            if self.pars['bayesian']:
                #print('bayesian')
                ds2 = (v['m'][:,k]*(3*gam + T.pow(gam,2)))*dm - y*T.pow(xi,2)
                        
                     
        # linear update specifics
        elif self.para == 'logNorm-lin':
    
            # expected activity
            o['gbar'][k] = o['ubar'][k]*self.pars['g0']
                
            # error signal
            o['delta'][k] = y - o['gbar'][k]*self.pars['dt']
    
            # learning rate factor            
            # don't divide by zero.
            if o['gbar'][k] > 0:
                v['eta'][:,k] = v['s2'][:,k]*v['x']/o['gbar'][k]
            # limes
            else: 
                v['eta'][:,k] = 0 # TODO: check what's the right limes
                            
            # as in derivations
            gam = v['s2'][:,k]/T.pow(v['m'][:,k],2)
            
            # update: mean
            dm = v['eta'][:,k]*o['delta'][k]
                    
            
            # update: variance
            if self.pars['bayesian']:
                ds2 = (v['m'][:,k]*(3*gam + T.pow(gam,2))
                                    )*dm - y*T.pow(v['eta'][:,k],2)


        # for non-Bayesian synapses, there's not variance update            
        if self.pars['bayesian'] is not True:
            ds2 = 0
            
        return(dm,ds2)
        
        
    def _update_x(self,Sx=None,readout_sampled_from_eta=False):
        """ if a spike vector Sx (T.float32) is provided, then:
            update internal vars[Sx] (T.uint) 
            update internal vars[x] (T.float32) 
            
            else: try to readout spike vector from internal vars            
        """
        
        if Sx is not None:            
            # store spike

            self.vars['Sx'][:,self.k] = T.tensor(Sx,dtype=T.uint8)
        else:
            # load spike
            Sx = T.tensor(self.vars['Sx'][:,self.k],dtype=T.float32)

        # update synaptic input        
        self.vars['x'] = self.vars['x']*self.sh['1-dt/tau_u'] + Sx
        # take care of bias        
        
        
        if self.pars['bias-active']:
            self.vars['x'][0] = 1

        # sample from posterior when a spike arrives.
        # weights that are not sampled, remain constant.
        # they are down regulated by x_eps.
        if self.pars['PL']['ON']:
            if self.pars['PL']['Sample']:     
                # compute log normal synapse from log variables
                
                if readout_sampled_from_eta == False:
                
                    if self.para is not 'Peter':                
                        M,S2 = self.get_weights()
                        sample = M + T.sqrt(S2)*self.eta_dim.sample()        
                        self.vars['w_r'] = (1-Sx)*self.vars['w_r'] + Sx*(sample) #std        
                        
                    if self.para is 'Peter':
                                
                        sample = T.exp(self.vars['m'][:,self.k] + 
                                        T.sqrt(self.vars['s2'][:,self.k])*
                                        self.eta_dim.sample())
                    self.vars['w_r'] = (1-Sx)*self.vars['w_r'] + Sx*(sample)

                elif readout_sampled_from_eta == True:
                    # JANNES
                    self.vars['w_r'] = self.vars['eta'][:,self.k]
        
    def get_weights(self,k=None):
        """ take log weight mean and var and output log normal weight pars """
        
        if k is None:
            k = self.k
        
        mu = self.vars['m'][:,k]
        sig2 = self.vars['s2'][:,k]
            
        if self.para is 'Peter':
            M = T.exp(mu + sig2/2)
            S2 = T.pow(M,2)*(T.exp(sig2) - 1)
            return(M,S2)
                    
        else: 
            return(mu,sig2)
                    
        
    def _hidden_update(self,v,o):
        """ update spike, membranes ect depending on method """
        k = self.k
        para = self.para

        beta = self.pars['PL']['beta']
        alpha = self.pars['PL']['alpha']

                
        ## create observation by bernulli sampling input rates        
        Sx = T.tensor(np.random.binomial(1,self.sh['nu*dt']),dtype=T.float32)
        self._update_x(Sx)

    
        ## OU update
        if para in ('gauss-exp','Peter'):
            # update hidden
            self.vars['w'][:,k+1] = v['w'][:,k] + self.sh['dt/tau_OU']*(
                    self.pars['OU']['mu'] - v['w'][:,k]) + ( 
                    self.sh['dt_sqrt']*self.sh['sig_pi']*self.eta_dim.sample())            

            
        ## log norm update
        elif para in ('logNorm-sigm','logNorm-lin'):
                    
#            print(v['w'][:,k])
#            print(k)
#            print('in')
            
            # update hidden
            self.vars['w'][:,k+1] = v['w'][:,k]*(
                     1
                     - self.sh['dt/tau_OU']*(T.log(v['w'][:,k]) - self.sh['c']) 
                     + self.sh['sig_pi']*self.sh['dt_sqrt']*self.eta_dim.sample())

#            print(v['w'][:,k+1])
        
        # compute membrane potential
        if para in ('gauss-exp','logNorm-sigm','logNorm-lin'):
                        
            
            
            # membrane pot.
            if self.dim_is_1:
                u = v['w'][0,k+1]*v['x'][0] 
            else:

                u = v['w'][:,k+1].dot(v['x'])                                       

            # predictive coding
            if self.pars['PL']['ON']:
                if self.pars['PL']['Sample']:
                    if self.dim_is_1:
                        u_PC = v['w_r'][0]*v['x'][0]
                    else:
                        u_PC = v['w_r'].dot(v['x'])                        
                else:
                    # running average membrane
                    u_PC = self.out['ubar_rm'][k]
                    print('subtracting this makes no sense!')
            # ordinary coding
            else:
                u_PC = 0


            # refractory variable
            if self.pars['y-rsp']['ON']:
                u += T.sum(self.out['a'][:,k+1])
            
            # write spike generating membrane pot. (could be dV if PL='ON')

            self.out['u'][k+1] = (alpha*u + 
                                  beta*u_PC +
                                  self.pars['th'])
            
            # running averages (of target pot.)
            self.out['ubar_rm'][k+1] = self.sh['dt/tau_run']*o['u'][k+1] + (
                    1-self.sh['dt/tau_run'])*self.out['ubar_rm'][k]
            
            self.out['sig2_u_rm'][k+1] = self.sh['dt/tau_run']*T.pow(
                    o['u'][k+1] - self.out['ubar_rm'][k],2) + (
                    self.out['sig2_u_rm'][k]*(1-self.sh['dt/tau_run']))
#            self.out['ubar_rm'][k+1] = self.sh['dt/tau_run']*u + (
#                    1-self.sh['dt/tau_run'])*self.out['ubar_rm'][k]
#            
#            self.out['sig2_u_rm'][k+1] = self.sh['dt/tau_run']*T.pow(
#                    u - self.out['ubar_rm'][k],2) + (
#                    self.out['sig2_u_rm'][k]*(1-self.sh['dt/tau_run']))
                    
            # get output firing probability
            if para == 'gauss-exp':
                # spike and make sure it's bounded
                gdt = (self.pars['g0']*T.exp(o['u'][k+1])).item()*self.pars['dt']
                                                    
            elif para == 'logNorm-sigm':              
                # spike and make sure it's bounded
    
                gdt = (T.tanh(o['u'][k+1]/self.pars['sig0_u'])*0.5 + 0.5).item(
                        )*self.sh['g0*dt']
    #            print(gdt)                         
            elif para == 'logNorm-lin':
                gdt = o['u'][k+1].item()*self.sh['g0*dt']

        elif para == 'Peter':                        
            # PETER            
            
            
            
            #print(w_star)
            #print(self.vars['w_r'])
            
            #old and good:
            #NOWJ
            if not self.pars['testing']:
                w_star = T.exp(v['w'][:,k]) 
                o['u'][k+1] = (alpha*w_star + beta*self.vars['w_r']).dot(v['x']) + self.pars['th'] 


            elif self.pars['testing']:            
                # new and for testing
                w_star = T.exp(v['w'][:,k])
                self.cur_noise = self.eta_0.sample()
                o['u'][k] = alpha*(w_star.dot(v['x'])) + self.pars['th'] + beta*self.cur_noise
            
                # next testing: same but w/ w_r time series.                                
                # self.cur_noise = self.vars['w_r'].dot(v['x'])
                
                # running average
                # self.cur_noise = (u_rm,u_rv)
                
                # o['u'][k+1] = (alpha*w_star + beta*self.vars['w_r']).dot(v['x']) + self.pars['th']
                
            
#            print('min',T.min(  w_star - self.vars['w_r']  ))
#            print('max',T.max(  w_star - self.vars['w_r']  ))
            
            gdt = (self.g(o['u'][k+1]/self.pars['sig0_u'])).item()*self.sh['g0*dt']
            
            self.gdt = gdt
            
        # check if still bounded
        if gdt > 1:
            print('activation exploded (time, value):',k,gdt)
            gdt = 1
                        
        # activation read out
        self.out['g'][k] = gdt/self.pars['dt']
            
        # generate output spike:
        if para in ('gauss-exp','logNorm-sigm','logNorm-lin','Peter'):

            if not self.pars['testing']:
                self.out['Sy'][k+1] = int(np.random.binomial(1,gdt))
                
                # if spike response is active 
                if self.pars['y-rsp']['ON']:
                    # decay kernl
                    self.out['a'][:,k+1] = self.out['a'][:,k]*self.sh['1-dt/tau_a']
                    
                    # and there's a spike ramp up kernl 
                    if self.out['Sy'][k+1] == 1:
                        self.out['a'][:,k+1] += self.pars['y-rsp']['a0']

            elif self.pars['testing']:
                self.out['Sy'][k] = int(np.random.binomial(1,gdt))
            
                
            
    def _hidden_readout(self,hidden,keys_readout=None):
        """ update intenral variables with hidden """
                
        if keys_readout==None:            
            keys_readout = copy.copy(self.k_readout)
                    
        # iterate over hidden and write to vars and out containers (pointers)
        for k in keys_readout:
            # select container for writing
            if k in self.vars.keys():
                self.vars[k] = hidden[k]
            else:
                self.out[k] = hidden[k]


            



# =============================================================================
#   Run a protocol 
# =============================================================================

    def protocol_to_hidden(self,protocol):
        """ generate Sx and Sy according to protocol """
        
        
        # spike pairs per burst
        n = protocol['pair_count_max']
#        self.n = n
        
        # distance within a burst
        Tb = int(protocol['pair_distance']/self.pars['dt'])
#        self.Tb = Tb
        
        # delta_T [s]
        dT = int(protocol['delta_T']/self.pars['dt'])
#        self.dT = dT 
        

        # first spike idx
        idx_1st = np.arange(n)*Tb
        self.idx_1st = idx_1st
        
        idx_2nd = np.arange(n)*Tb + np.abs(dT)
        self.idx_2nd = idx_2nd 
                        
        # re-init
        self.out['Sy'] = self.out['Sy']*0
        self.vars['Sx'] = self.vars['Sx']*0
    
        # timing
        if dT > 0:
            self.vars['Sx'][:,idx_1st] = 1            
            self.out['Sy'][idx_2nd] = 1                        
        else:
            self.vars['Sx'][:,idx_2nd] = 1
            self.out['Sy'][idx_1st] = 1
               
        # read out: create a hidden object
        return(self.export_hidden())
        

# =============================================================================
#     Run Bayesian Synapses
# =============================================================================
            
    def export_hidden(self,keys_readout=None):
        """ export Sx, Sy, u, w to run it with a new model """
        
        if 'mdims' in self.pars:
            print('Warning: Cant export hidden because in save memory mode time series are not stored. Exit method.')

        else:
            export = {}
    
            if keys_readout==None:            
                keys_readout = copy.copy(self.k_readout)
    
            # iterate over hidden in vars and out container and extract them                
            for k in keys_readout:
                # select container for reading
                if k in self.vars.keys():
                    export[k] = self.vars[k]
                else:
                    export[k] = self.out[k]
            
            return(export)


    def log_normal_sample(self,mu,sig2,input_type='LOG'):
        """ draw samples from log-normal. 
            LOG : mu, sig2 represent the mean and var of the log normal
            W : mu, sig2 are mean and var of the log normal dist
            kLOG : mu is log mean and k*M = S2 (Works for large negative mu)
        """
        
        if input_type=='LOG':
#            return(T.exp(mu + T.sqrt(sig2)*self.eta_dim.sample()))
            return(T.exp(mu + T.sqrt(sig2)*self.eta_fun(len(mu))))
            
            
        elif input_type == 'W':
            #sig2_sample = T.log(1+ sig2/T.pow(mu,2))
            sig2_sample = T.log( T.pow(mu,2) + sig2) - 2*T.log(mu)
            return(self.log_normal_sample(T.log(mu) - 0.5*sig2_sample,sig2_sample))
            #return(T.exp(mu_sample + T.sqrt(sig2_sample)*self.eta_dim.sample()))                                            
        elif input_type == 'kLOG':
            # good indices                
            if 'warning' not in self.pars['PL']:
                print('Warning: log_normal_sample ignores s2 and uses k*M instead')
                self.pars['PL']['warning'] = True
            sig2_sample = T.log(T.exp(mu) + self.pars['PL']['k_samp']) - mu
            return(self.log_normal_sample(mu - 0.5*sig2_sample,sig2_sample))
                                            

    def run_peter(self):
        """ for peter's update rule """
        
        # time
        t0 = int(time())
        t_out = self.pars['time_between_output']

        # shorthand
        v = self.vars
        o = self.out
        p = self.pars
        sh = self.sh
        # pre factor for peter
        alpha = p['PL']['alpha'] 
        # from predictive coding        
        beta = p['PL']['beta']
        
        # smoothing 
        gamma = 1 - p['dt']/p['tau_running_ave']

        ks_count = 0        
        k_till_out = self.xSteps / min(1000,self.xSteps)

        # expected input rates
        varx = T.tensor(p['nu']*p['tau_u']/2,dtype=T.float32)
        meanx = T.tensor(p['nu']*p['tau_u'],dtype=T.float32)
        meanx2 = T.pow(meanx,2)
                            
        print('PL opt',p['PL']['opt'])
        
        # loop over time steps
        self.K = self.k # exact copy to start 
        while self.K < self.xSteps:

            # this k is ALWAYS self.k shifts back and forth
            k = self.k
            
            # compute log normal synapse from log variables
            
#            if 'warning: M' not in p:
#                print('Taken Bayesian M as maximum likelihood.')
#                p['warning: M'] = True
            
            if p['bayesian']:
                M,S2 = self.get_weights()
#                M = T.exp(v['m'][:,k])
                if self.K % k_till_out == 0:
                    v['k'] = v['k'] + S2/M
                    ks_count += 1
                    #print('vk:',v['k'])
            else:
                M = T.exp(v['m'][:,k])

            ###### World 
            #Sx = T.tensor(np.random.binomial(1,sh['nu*dt']),dtype=T.float32)            
            ii_Sx = np.where(np.random.binomial(1,sh['nu*dt']))[0]
            n_Sx = len(ii_Sx)
            # IMPLEMENT            
            if p['eps_smooth']==False:
                v['x'] = v['x']*sh['1-dt/tau_u']
                if n_Sx > 0:
                    v['x'][ii_Sx] += 1
            else:    
                v['x'],v['xdot'] = (
                        v['x']*sh['1-dt/tau_u'] + v['xdot']*sh['dt/tau_u'],
                        v['xdot']*sh['1-dt/tau_u'] - v['x']*sh['dt/tau_u'])

                if n_Sx > 0:
                    v['xdot'][ii_Sx] += p['tau_u']*p['tau_u']*0.4
                    # 0.4 is the normalization for tau = gamma = 0.01ms                            
            
            v['w'][:,k+1] = v['w'][:,k] + sh['dt/tau_OU']*(
                    p['OU']['mu'] - v['w'][:,k]) + ( 
                    sh['dt_sqrt']*sh['sig_pi']*self.eta_dim.sample())            
            
            if 'warning: k' not in p:
                print('Sampling from k*m for Bayesian.')
                p['warning: k'] = True
                
            if beta != 0 and n_Sx > 0:
                                        
                if p['bayesian']:                                            
    #                 draw from filtering dist: "bayesian sampling"
                    if p['PL']['b_samp']: 
                        M_sample = self.log_normal_sample(
                                v['m'][ii_Sx,k],v['s2'][ii_Sx,k])
                    elif p['PL']['k_samp'] > 0:
                        # k sampling
                        M_sample = self.log_normal_sample(
                                M[ii_Sx],M[ii_Sx]*p['PL']['k_samp'],
                                                        input_type='W')
                    elif p['PL']['k_samp'] == 0:
                        M_sample = M
                        
                else:
    #                 E[w] = exp(lambda), var[w] = k*E[w]
                    if p['PL']['k_samp'] > 0:
                        M_sample = self.log_normal_sample(v['m'][ii_Sx,k],None,
                                                      input_type='kLOG')
                    else:
                        M_sample = M
                    
                                    
                if T.sum(M_sample<0) > 0:
                    print(self.k,'w_sample neg')
                    ii = M_sample<0
                    print(np.where(np.array(ii)))    
                    
                v['w_r'][ii_Sx] = M_sample
                
                if T.isnan(M_sample).sum() > 0:
                    print(self.k,'w_r exploded -- resetting it to m_i')
                    ii = T.isnan(v['w_r'])
                    v['w_r'][ii] = T.exp(v['m'][ii,k])

                if T.sum(M_sample<0) > 0:
                    print(self.k,'w_r neg')
                    ii = v['w_r'] < 0
                    ii = np.array(ii)
                    print(np.where(ii))                
                    
            # draw next spike
            w_star = T.exp(v['w'][:,k])
            o['u'][k] = (alpha*w_star + beta*v['w_r']).dot(v['x']) + p['th'] 
            gdt = (self.g(o['u'][k]/p['sig0_u'])).item()*sh['g0*dt']
            # check if still bounded
            if gdt > 1:
                print('activation exploded (time, value):',k,gdt)
                gdt = 1
            o['g'][k] = gdt/p['dt']
            o['Sy'][k] = int(np.random.binomial(1,gdt))
            y = T.tensor(o['Sy'][k],dtype=T.float32) #if k > 0 else 0            

            ###### prior            
            if p['bayesian']:
                dm_prior = - (v['m'][:,k] - p['OU']['mu'])*sh['dt/tau_OU']
                ds2_prior = - 2*(v['s2'][:,k] - p['OU']['sig2'])*sh['dt/tau_OU']            
            else:
                dm_prior = 0                                
                ds2_prior = 0            
            
            ##### likelihood
            if p['PL']['opt'] == 1:               
                # w_r and x known            
                o['ubar'][k] = p['th'] + v['x'].dot(alpha*M + beta*v['w_r'])   #+ beta*self.cur_noise 

                if p['bayesian']:
                    o['sig2_u'][k] = alpha**2*(S2.dot(T.pow(v['x'],2)))
                else:
                    o['sig2_u'][k] = 0

                V_vec = o['ubar'][k]


            elif p['PL']['opt'] == 2:
                # w_r estimated, x known (problem: ubar relies on M,Sx)
                o['ubar'][k] = p['th'] + v['x'].dot(M)*(alpha + beta)   #+ beta*self.cur_noise 
                if p['bayesian']:
                    o['sig2_u'][k] = (alpha**2+beta**2)*(S2.dot(T.pow(v['x'],2)))                                        
                else:
                    o['sig2_u'][k] = 0
                V_vec = o['ubar'][k] - beta*(M - v['w_r'])*v['x']
            
            elif p['PL']['opt'] == 3:
                # w_r, x estimated (problem: ubar still relies on M)
                o['ubar'][k] = p['th'] + (alpha + beta)*meanx.dot(M)
                
                if p['bayesian']:
#                    o['sig2_u'][k] = (alpha**2 + beta**2)*(S2.dot(varx + meanx2
#                                                 ) + T.pow(M,2).dot(varx))
                    o['sig2_u'][k] = (S2.dot(meanx2))*(alpha**2 + beta**2)                    
                else:
                    o['sig2_u'][k] = 0
                # subtract and add
                V_vec = o['ubar'][k] - (alpha + beta)*meanx*M + (
                                        v['x']*(alpha*M + beta*v['w_r']))

            elif p['PL']['opt'] == 4:
                # w_r, x estimated, M taken as prior       
                # ou mean and var in weight space                
                if self.K == 0: # compute only once
                    M_prior = T.exp(p['OU']['mu'] + 0.5*p['OU']['sig2'])
                    S2_prior = T.pow(M_prior,2)*(T.exp(p['OU']['sig2'])-1)

                o['ubar'][k] = p['th'] + (alpha + beta)*meanx.dot(M_prior)
                
                if p['bayesian']:
#                    o['sig2_u'][k] = (alpha**2 + beta**2)*(
#                                        S2_prior.dot(varx + meanx2) + 
#                                        T.pow(M_prior,2).dot(varx)
#                                        )
                    o['sig2_u'][k] = (alpha**2 + beta**2)*S2_prior.dot(meanx2)
                    
                else:
                    o['sig2_u'][k] = 0
                # subtract and add
                V_vec = o['ubar'][k] - (alpha + beta)*meanx*M_prior + (
                        v['x']*(alpha*M + beta*v['w_r']))

            elif p['PL']['opt'] == 5:
                # running average
                #o['ubar'][k] = o['ubar'][k]*gamma + (1-gamma)*o['u'][k]
                
                if p['bayesian']:
                    o['sig2_u'][k+1] = o['sig2_u'][k]*gamma + (1-gamma
                                     )*T.pow(o['u'][k] - o['ubar'][k],2)
                else:
                    o['sig2_u'][k+1] = 0
                
                if self.K == 0: # compute only once
                    M_prior = T.exp(p['OU']['mu'] + 0.5*p['OU']['sig2'])
                    S2_prior = T.pow(M_prior,2)*(T.exp(p['OU']['sig2'])-1)

                o['ubar'][k] = p['th'] + (alpha + beta)*meanx.dot(M_prior)
                V_vec = o['ubar'][k] - (alpha + beta)*meanx*M_prior + (
                        v['x']*(alpha*M + beta*v['w_r']))
                                
                #V_vec = o['ubar'][k] 
            
            sigV = T.sqrt(p['sig0_u']**2 + o['sig2_u'][k])
                            
            eq = {}

            # post synaptic factors
            z = V_vec/sigV 
            
#            z_test = V_vec/p['sig0_u']
#            sigV_test = p['sig0_u']
            
#            if 'warning: slope' not in p:
#                print('Ignoring slope adaption for testing by using z_test and sigV_test')
#                p['warning: slope'] = True
            
#            if 'warning: sanity' not in p:
#                print('Setting V_vec / sigV to std values')
#                p['warning: sanity'] = True
            
            eq['Phi'] = self.g(z)
            eq['delta'] = y - eq['Phi']*sh['g0*dt']

            # pre synaptic factors
            eq['xi*sigma^2'] = v['s2'][:,k]*alpha*M*v['x']/sigV
            
            # mean update: sig^2*xi*phi'/phi*(y - phi)
            dm_like = eq['delta']*self._phi_dash_over_phi(z)*eq['xi*sigma^2']
            
            if p['bayesian']:
                # var update:  sig^4*xi^2*(y*[phi''/phi - (phi'/phi)^2] - phi'')            
                ds2_like = - T.pow(eq['xi*sigma^2'],2)*self._phi_dash_over_phi(z)*(
                    y*self._x_minus_phi_dash_over_phi(z) 
                    + eq['Phi']*z*sh['g0*dt'])
            else:
                ds2_like = 0

            #print(k,ds2_like)

            ###### Update
            if p['mu_bounds'] is not None:
                if 'mu_bounds_cut' not in p:
                    p['mb_cut'] = T.sqrt(p['OU']['sig2'][0])*p['mu_bounds']
                dm_like.clamp_(-p['mb_cut'],p['mb_cut'])
                                                
            v['m'][:,k+1] = v['m'][:,k] + dm_prior + dm_like
            v['s2'][:,k+1] = v['s2'][:,k] + ds2_prior + ds2_like
                                        
            # error: self.res_online += T.pow(v['m'][:,k] - v['w'][:,k],2)
            # filter: 
            ## Timing
            dtime = int(time())-t0
            if dtime >= t_out:
                print(dtime,'[sec]: step ',self.K)
                t_out += p['time_between_output']
                print('s2:', v['s2'][0,k])
                #print('ds2_like',ds2_like)            
                print('')
               
                                 
            # increment: 
            # for mdims do backshift and always self.k = 0, self.K increases.
            if 'mdims' in self.pars:
                # store long series (last completed step)

                # readout pre-synaptic vars
                i = 0
                for mdim in self.pars['mdims']:
                    self.mvars['w'][i,self.K] = v['w'][mdim,0]
                    self.mvars['m'][i,self.K] = v['m'][mdim,0]
                    self.mvars['s2'][i,self.K] = v['s2'][mdim,0]                    
                    i += 1
            
                # readout post-synaptic vars
                dw = T.pow(self.vars['w'][:,0] - self.vars['m'][:,0],2)
                self.mout['w MSE'][self.K] = T.sum(dw).item()
                self.mout['q'][self.K] = T.sum(dw < self.vars['s2'][:,0]).item()
                        
                self.K += 1 # NO INCREMENT in self.k and k, only in self.K                    
                # copy values back: from 1 to 0
                self._shift_back()
        
            else: # old code: keep self.k and self.K aligned
                self.k += 1                                                                
                # shift data back
                if self.pars['step_out'] > 1:
                    if self.k == self.k_out + self.pars['step_out']:                        
                        self.k_out += 1                                        
                        self._copy_vars_in_time(k_from=self.k,k_to=self.k_out)        
                        self.k = self.k_out                                    
                self.K = self.k
                if self.K == self.xSteps-1:
                    break
        
        # normalize ks
        if p['bayesian']:
            v['k'] = v['k']/ks_count 
           
    def run(self,hidden=None,run_synapse=True,k_stop=None):
        """ spiking forward propagation of filtering distribution for 
            respective model. Generate hidden if hidden is not provided."""
        
        if self.para == 'Peter':
            self.run_peter()
        else:
        
            # time
            t0 = int(time())
            t_out = self.pars['time_between_output']
    
            # write hidden varibes to internal variables for all times
            if hidden is not None:
                self._hidden_readout(hidden)
    
            # shorthand
            v = self.vars
            o = self.out
            # loop over time steps        
            while self.k < self.xSteps - 1:
                k = self.k
    
                # update pre-synaptic activation variable on internal spike train
                if hidden:            
                    # take pre computed sample w_r for Peter and do benchmark
                    self._update_x(readout_sampled_from_eta = self.para == 'Peter')
                # propagate world forward (Sx,Sy,w,u)
                else:
                    self._hidden_update(v,o)
                        
                # exclude filter if  we only need the hidden
                if run_synapse:
                        
                    # get prior updates if OU flag is active
                    if self.pars['OU']['flag']:
                        dm_pi,ds2_pi = self._prior_update(v['m'][:,k],v['s2'][:,k])
                    else:
                        dm_pi,ds2_pi = (0,0)
                    
                    # get likelihood updates
                    dm_like,ds2_like = self._likelihood_update(v,o)
    #                print('like then prior')
    #                print(dm_like,ds2_like)
                        
                    # update parameters 
                    self.vars['m'][:,k+1] = v['m'][:,k] + dm_pi + dm_like
                    self.vars['s2'][:,k+1] = v['s2'][:,k] + ds2_pi + ds2_like
    #                print(ds2_pi)
    #                print(ds2_like)
                                    
                    # check for negative variance
                    if T.sum(self.vars['s2'][:,k+1] < 10**(-6)) > 0:
                        print(self.k,'error: negative variance!')
    #                    print('ds2_pi',ds2_pi)
    #                    print('ds2_like',ds2_like)
                        print('')
                        
                        ii = self.vars['s2'][:,k+1] < 10**(-6)
                        
                        i_idx = T.arange(0,self.pars['dim'])[ii]
                        
                        A= {'err':'neg var','i':i_idx,'k':self.k+1,
                            'V_i':self.V_vec[ii],'sigV':S.sigV,
                            'w_r':self.vars['w_r'][ii], 
                            'gdash_over_g':self._phi_dash_over_phi(self.V_vec[ii]/S.sigV)
                            }
                        
                        self.errorLog.append(A)
                                                                
                        # assign minimal variance                
                        (self.vars['s2'][:,k+1])[ii] = 10**(-6)
    
                    # check for NaN
                    if T.sum(T.isnan(self.vars['s2'][:,k+1])) > 0:
                        print(self.k,'error: nan variance!')
                        print('')
                        
                        ii = T.isnan(self.vars['s2'][:,k+1])
                        
                        i_idx = T.arange(0,self.pars['dim'])[ii]
                        
                        A= {'err':'nan var','i':i_idx,'k':self.k+1,
                            'V_i':self.V_vec[ii],'sigV':S.sigV,
                            'w_r':self.vars['w_r'][ii], 
                            'gdash_over_g':self.gdash_over_g(self.V_vec[ii]/S.sigV)
                            }
                        
                        self.errorLog.append(A)
    
                        
                        
                    # check for negative synapses, when using logNorm synapses
                    if self.para not in ('gauss-exp','Peter'):                
                        if T.sum(self.vars['m'][:,k+1] < 0) > 0:
                            print(self.k,'error: negative synapses!')
                            # assign minimal weight
                            (self.vars['m'][:,k+1])[self.vars['m'][:,k+1] < 0] = 0.0001
    
                # update class wide index
                self.k += 1
                                        
                # shift data back
                if self.pars['step_out'] > 1:
                    if self.k == self.k_out + self.pars['step_out']:
                        
                        self.k_out += 1
                                        
                        self._copy_vars_in_time(k_from=self.k,k_to=self.k_out)
        
                        self.k = self.k_out
                    
    
    #            k_stop = 200
                # break loop early
                if k_stop != None:
                    if self.k == k_stop:
                        break
                    
                # print progress
                dtime = int(time())-t0
                if dtime >= t_out:
                    print(dtime,'[sec]: step ',k)
                    t_out += self.pars['time_between_output']
                    print('s2:', self.vars['s2'][0,k])
                    #print('ds2_like',ds2_like)            
                    print('')
                    
            # finished
            #if self.pars['plot-flags']['all-trajectories']:
            #    self.wrap_plt(keys= ('w','u MSE','w MSE','Sx','Sy'))
    
    
    def _copy_vars_in_time(self,k_from,k_to):
        """ copy data according to indices """

        for key in ['w','eta','m','s2','Sx']:
            # in case future steps were computed: copy them too
            self.vars[key][:,k_to:k_to+1] = self.vars[key][:,k_from:k_from+1]
        
        for key in self.out.keys():
            if len(self.out[key]) == self.pars['xSteps']:
                # in case future steps were computed: copy them too
                self.out[key][k_to:k_to+1] = self.out[key][k_from:k_from+1]            
    

    def _shift_back(self):
        """ shift k = 1 to k = 0. """

        if self.k != 0:
            print('Warning: shift_back should only be used with self.k=0')

        for key in ['w','eta','m','s2','Sx']:
            # in case future steps were computed: copy them too
            self.vars[key][:,0] = self.vars[key][:,1]
        
        for key in self.out.keys():        
            self.out[key][0] = self.out[key][1]
        
    
    
    
    def run_fast(self,protocol,compute_jac=['tau','g0','sigma_V'],outlvl=0):
        """ no thrills RK wrapper that computes jacobian 
            x contains:
                x[:dim] = mu
                x[dim:2dim] = sig2
                x[2dim:3dim] = x_eps
                x[3dim:5dim] = partial_g0 (mu,sig2)
                x[5dim:7dim] = partial_sigV (mu,sig2)
                x[7dim:9dim] = partial_tau (mu,sig2)
        """
        
        # reverse hack
        dim = 1 if self.dim_is_1 else self.pars['dim']
        
        # init 
        t0 = 0
        jacL = len(compute_jac) if compute_jac is not None else 0
        # init: mu, var, x_eps
        x0 = [protocol['mu0']]*dim + [protocol['var0']]*dim + [0]*dim

        # add dim for jacobian if computed
        x0 = np.array(x0) if jacL == 0 else np.array(x0 + [0]*dim*2*jacL)


        self.compute_jac = compute_jac
        
        # init this tau derivative spike response fct
        self.partial_tau_of_xeps = partial_tau_of_xeps()
        
        # Compute spike times:
        tfs = gen_spike_times(protocol,t_wait=self.pars['tau_u']*10)
                                   
        # is pre spike?
        self.pre =  True if protocol['delta_T'] > 0 else False
                             
        tf,xf = self.RK_wrap(lambda t,x : self.dot_x(t,x,y_spike=False),
                             t0,x0,tfs,new=True,output_level=outlvl)
        # run simulation           
        if outlvl == 0:                                 
            # formatting of output
            mu = xf[:dim]
            sig2 = xf[dim:2*dim]
            # skip x_eps and output jacobian
            jac = xf[3*dim:]
            return(mu,sig2,jac)

        else: 
            return(tf,xf)


    def RK_wrap(self,dot_x,t0,x0,t_event,new=True,output_level=0):
        """ x0 is a vector with all states, t_event are the spikes,
            pre post spikes alternate. Previous spike is store in self.pre
            output_level = 0 only final result
            output_level = 1 only at spike times
            output level = 2 all steps
        """
        
        # reverse hack
        dim = 1 if self.dim_is_1 else self.pars['dim']
        
        # init x spike
        x_add = np.zeros(len(x0))
        x_add[2*dim:3*dim] = 1
        self.x_add = x_add
        
        if output_level == 1:
            out = {}
            out['t'] = []
            out['x'] = []
        if output_level == 2:
            out = {}
            out['t'] = []
            out['x'] = []
        
        if not new: # in each iteration
            RK = RK45(dot_x,t0,x0,t_event[0])
        
        for t_jump in t_event:
            
#            print('jump time',t_jump)
#            print(x0) 
#            print('')
            
            if new:            
                RK = RK45(dot_x,t0,x0,t_jump)
            else:
                RK.t = t0
                RK.t_bound = t_jump
                RK.x = x0
                RK.status = 'running'
                                
            while RK.status is 'running':
                RK.step()

                # before jump
                if output_level == 2:
                    out['t'].append(RK.t)
                    out['x'].append(RK.y)

            # before jump    
            if output_level == 1:
                out['t'].append(RK.t)
                out['x'].append(RK.y)

                
            t0 = t_jump
            
            if np.isnan(self.dx).any():
#                print('x',self.x)
#                print('dx',self.dx)
                self.t_jump = t_jump
#                print('')

            
            
            if self.pre:
                # update x_eps only by adding spike +1

                x0 = RK.y + x_add
                
                # add spike for computation needed with in dot_x
                self.partial_tau_of_xeps.add_spike(t_jump)
            
            else: # post spike
                x0 = RK.y + self.dot_x(RK.t,RK.y,y_spike=True) # jump

            # after jump
            if output_level in (1,2):
                out['t'].append(RK.t+0.00001)
                out['x'].append(x0)

        
            # pre post alterante. Last spike is pre.
            self.pre = not self.pre if t_jump < t_event[-2] else True
    
        if output_level in (1,2):
            # format: time and rest  
            return(np.array(out['t']),np.array(out['x']))
        else:
            return(t0,x0)
                      
        
    def dot_x(self,t,x,y_spike=False):
        """ compute update (not yet tested but finalised) """
        # reverse hack
        dim = 1 if self.dim_is_1 else self.pars['dim']

        dx = np.zeros(len(x))
        
        self.dx = dx
        self.x = x
        self.dim = dim
        
        ubar = x[:dim].dot(x[2*dim:3*dim])
        
        # expected membrane noise
        sig2_u = x[dim:2*dim].dot(np.power(x[2*dim:3*dim],2))

        # sigmoidal width
        sigV_bar = np.sqrt(self.pars['sig0_u']**2 + np.pi/8*sig2_u)

        # expected activity
        self.sigm = np.tanh(ubar/sigV_bar)*0.5 + 0.5
        sigm = self.sigm
        
        gbar = sigm*self.pars['g0']
        self.gbar = gbar        
        # as in derivations
        xi = (1-sigm)*x[dim:2*dim]*x[2*dim:3*dim]/sigV_bar
        self.xi = xi
        
        gam = x[dim:2*dim]/np.power(x[:dim],2)
        
        # for var update
        pref = x[:dim]*(3*gam + np.power(gam,2))

        # updates
        # if output spike: all dt vanish, only propto y survives.
            
        # mean                                                
        if y_spike:
            dx[:dim] = xi
                    
        else:
            # absence of spike
            dx[:dim] = - xi*gbar
            
            # update for pre-synaptic activation
            dx[2*dim:3*dim] = -x[2*dim:3*dim]/self.pars['tau_u']

            
        # var update is always the same (with and without spike)        
        dx[dim:2*dim] = pref*dx[:dim]
               
        
        
        # idx to multipy dimensions and pass output to dx         
        i = 3
        # deriv wrt to nu0
        if 'g0' in self.compute_jac:        
            # mean
            if y_spike:
                dx[i*dim:(i+1)*dim] = 0
            else:
                dx[i*dim:(i+1)*dim] = xi*sigm
                
            i += 1            
            # var (is updated with partial_theta dot{m})
            dx[i*dim:(i+1)*dim] = pref*dx[(i-1)*dim:i*dim]
            i += 1
        
        if 'sigV' in self.compute_jac:
            dsigV_dxi = - xi*self.pars['sig0_u']/sigV_bar**2*(1 + ubar*sigm)
            dsigV_ddelta = - gbar*(1-sigm)*ubar*self.pars['sig0_u']/sigV_bar**3

            # mean
            if y_spike:
                dx[i*dim:(i+1)*dim] = dsigV_dxi
            else:
                dx[i*dim:(i+1)*dim] = xi*dsigV_ddelta - gbar*dsigV_dxi
            i += 1            

            # var
            dx[i*dim:(i+1)*dim] = pref*dx[(i-1)*dim:i*dim]
            i += 1
            
        if 'tau' in self.compute_jac:
            
            # compute activation of deriviative in special function
            dtau_dxeps = np.ones(dim)*self.partial_tau_of_xeps(t,self.pars['tau_u'])
            
            # TODO: remove
            self.partial_tau_of_xeps.vals.append(dtau_dxeps)
            
            dtau_darg_summed = dtau_dxeps.dot(x[:dim]/sigV_bar - 
                       np.pi/8*ubar/sigV_bar**3*x[dim:2*dim]*x[2*dim:3*dim])
            
            dtau_dxi = x[dim:2*dim]*(1-sigm)/sigV_bar*(
                        dtau_dxeps + x[2*dim:3*dim]*sigm*dtau_darg_summed)
            
            dtau_ddelta = gbar*(1-sigm)*dtau_darg_summed

            # mean
            if y_spike:                
                dx[i*dim:(i+1)*dim] = dtau_dxi
            else:                
                dx[i*dim:(i+1)*dim] = xi*dtau_ddelta - gbar*dtau_dxi                
            i += 1            
            
            # var
            dx[i*dim:(i+1)*dim] = pref*dx[(i-1)*dim:i*dim]
        

        self.dx = dx
        self.x = x
        
        return(dx)
            
                        
    
    def get_MSE(self):
        """ compute mean squared errors """

        if 'mdims' not in self.pars:
            # short hand
            o = self.out        
            dw2 = T.pow(self.vars['w'] - self.vars['m'],2)
    
            # how often is it inside? pool across time
            self.res['qi'] = np.mean(np.array(dw2 < self.vars['s2']),1)
            # pool dimensions
            self.res['q'] = np.mean(self.res['qi'])
                
            # time series for plotting: pool dimensions
            o['w MSE'] = T.sum(dw2,0)/self.pars['dim']
    
            # time series
            o['u MSE'] = T.pow(o['u'] - o['ubar'],2)
                        
            # outputs
            self.res['w MSE'] = (T.sum(o['w MSE'])/self.pars['xSteps']).item()
            self.res['u MSE'] = (T.sum(o['u MSE'])/self.pars['xSteps']).item()    
            return(self.res)

        else:
            # normalize across synapses and time
            for key,series in self.mout.items():
                self.res[key] = np.mean(np.array(series))/self.pars['dim']
            return(self.res)
        

 

    
    
# =============================================================================
#     Plotting
# =============================================================================

    
    
    def pars2title(self):
        """ turn pars into a title string """
        if self.title is None:
            pars = self.pars
            title = ''
            for k in ('dim','g0','xSteps','nu'):
                if k == 'g0':            
                    v = pars[k]
                    key = k
                elif k == 'nu':            
                    # just first rate
                    v = set(np.array(pars[k]))                    
                    if len(v) > 2:
                        v = 'many'
                    key = k
                elif k == 'xSteps':
                    v = pars[k]
                    key = 'N'
                elif k == 'dim':
                    v = pars[k]
                    key = 'D'
        
                # add to titlte string
                title += key + '=' + str(v) + ' '

            # set title for reuse
            self.title = self.name + ' ' + title[:-1]
        
        # remove trailing space
        return(self.title)
        
    def x_axis(self):
        """ generate x axis """
        if self.pars['step_out'] == 0:
            step_out = 1
            print('warning: step_out was none, set it to 1 for plotting')
        else:
            step_out = self.pars['step_out']
        
        return(np.arange(self.pars['xSteps'])*self.pars['dt']*step_out)
        
        
            
    def plt_vars(self,plt_dim,plt_keys,tlim=None,multiple_figures=True):                        
        """ plt everything related to a single synapse """
        # hidden lambda and estimator and spikes          
        
        if self.para in ('gauss-exp','logNorm-sigm','logNorm-lin','Peter'):
            # 
            numPlots = 0
            ratio = []
            
            # check number of plots and set plot ratio
            for key in (list(self.vars.keys()) + list(self.out.keys())):
                if key in plt_keys:
                    numPlots += 1
                    ratio.append(1)

            # create figure
            if multiple_figures:
                # call figures and select them by key
                [plt.figure(k) for k in plt_keys]
            else:
                f, axs = plt.subplots(numPlots,1,sharex=True,
                                  figsize=(6, 2*numPlots),
                                  gridspec_kw = {'height_ratios':ratio})
            
                # ignore axs if there's only one plot
                if numPlots > 1:
                    axs = list(axs)
                        
            # full range if no range is provided
            if tlim is None:
                tlim = (0,self.pars['xSteps']) 
        
            # time axis            
            xplt = (self.x_axis())[tlim[0]:tlim[1]]

        
            # iterate over plots
            for lkey in plt_keys:
                # get current axis, in case their are multiple
                if ((numPlots > 1) and (multiple_figures==False)):
                    plt.sca(axs.pop(0))                                                                
                else:
                    # select figure
                    plt.figure(lkey)
                                    
                # contains: yplt, use_filter, ylabel, title, 
                # col, alpha, size, legend_labels, use_legend
                p = self.key_to_plt(lkey,plt_dim)
#                self.p = p                             
                  
                # filter
                if p['use_filter']:
                    
                    # truncate if needed
                    for k in p['yplt'].keys():
                        p['yplt'][k] = p['yplt'][k][tlim[0]:tlim[1]]
                    
                    # plot
                    plt_filter(xplt,p['yplt'],p['col'],alpha=p['alpha'],size=p['size'])

                # check if key is initialised
                elif 'multi-plot' in p.keys():
                    
                    # truncate and plot
                    # iterate over keys and colors
                    for k,c in zip(p['multi-plot'],p['col']):
                        p['yplt'][k] = p['yplt'][k][tlim[0]:tlim[1]]
                        plt.scatter(xplt,p['yplt'][k],alpha=p['alpha'],
                                    s=p['size'],c=c)
                                        
                # no filter
                else:                                    
                    
                    # truncate if needed
                    p['yplt'] = p['yplt'][tlim[0]:tlim[1]]

                    # plot
                    plt.scatter(xplt,p['yplt'],alpha=p['alpha'],s=p['size'],c=p['col'])
                    
                    
                # create legend
                if p['use_legend']:                    
                    lgnd_items = []
                    for c,l in zip(p['col'],p['legend_labels']):
                        lgnd_items.append(matplotlib.patches.Patch(color=c,label=l))
                        plt.gca().legend(handles=lgnd_items,ncol=2,prop={'size': 12})
                         # mode='expand'            
                                                                     

                plt.title(p['title'])
                plt.ylabel(p['ylabel'])
                plt.xlabel('time [s]')                                                                            
                
                if self.pars['plot-flags']['save-output']:
                    # prepare file name and remove latex symbols 
                    file_name = p['title']
                    for sym in ('\\','$','{','}',':'):
                        file_name = file_name.replace(sym,'')
                    # save fig and make a copy of parameters and source file
                    
                    # omit suffix '.py'                    
                    #src_file_name = os.path.basename(__file__)[:-3]
                    
                    file_path = os.path.realpath(__file__)
                    file_dir = file_path[:-file_path[::-1].find('/')]                    
                    idx = ((file_dir[::-1])[:]).find('/',1)
                    root = file_dir[:-idx]
                    output_path = root + 'output/' + self.name + '/'
                    
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    
#                    print(output_path)
#                    print(file_dir)
#                    print(root)
#                    print(output_path + file_name)
#                    x = []**2
                    
                    plt.savefig(output_path + file_name + '.png', bbox_inches='tight',dpi = 300)
                    
                    # copy parameters                        
                    text_file = output_path + '_parameters.txt'
                    # copy only if it doesn't exist:
                    if not os.path.exists(text_file):        
                        # open text file and write to it
                        with open(text_file, 'w') as file:  
                            for k,v in self.pars.items():
                                print(k,':',v,file=file)


                    
#                    my_savefig(self.pars2title(),file_name + '.png', 
#                               self.pars,src_file_name)                
                    
                else:
                    plt.show()

# =============================================================================
#   Dictionary with plotting specifics
# =============================================================================

        
    def key_to_plt(self,lkey,plt_dim=None):
        """ for any variable label produce plot parameters. contains: 
                yplt, use_filter, ylabel, title, 
                col, alpha, size, legend_labels, use_legend
            in a dict 'p'. """

        # dictionary
        p = {}
        
        # synaptic identifier for title
        if plt_dim is not None:
            title_str = r'$\nu_{%r}=%.1f Hz$' % (plt_dim, self.pars['nu'][plt_dim].item())
        
        
        # prepare filtering variables
        if lkey in ('u','u_rm','w'):
            p['yplt'] = {}
            p['use_filter'] = True
            p['use_legend'] = True
            p['alpha'] = 0.7
            p['size'] = 1

            # put together filter plot using all the different variables            
            if lkey == 'u':
                # hidden and esitmate
                p['yplt']['hidden'] = np.array(self.out['u'])
                p['yplt']['mean'] = np.array(self.out['ubar'])
                p['yplt']['sig'] = np.sqrt(np.array(self.out['sig2_u']))                                                                        
                
                # labels and legends
                p['ylabel'] = r'membrane $u$'
                p['legend_labels'] = (r'target $u_{tar}$',
                                      r'estimated $\langle u \rangle$')
                p['col'] = self.pars['cols']
                p['title'] = 'Membrane pot. $u$ vs $u_{tar}$'
                
            elif lkey == 'u_rm':     
                # hidden and esitmate
                p['yplt']['hidden'] = np.array(self.out['u'])
                p['yplt']['mean'] = np.array(self.out['ubar_rm'])
                p['yplt']['sig'] = np.sqrt(np.array(self.out['sig2_u_rm']))
                
                # labels and legends
                p['ylabel'] = r'membrane $u$'
                p['legend_labels'] = (r'target $u_{tar}$',
                                      r'running average $\hat{u}$')
                p['col'] = self.pars['cols']
                p['title'] = 'Membrane pot. running average $u_{rm}$ vs $u_{tar}$'
                
            elif lkey == 'w':        
                # hidden and esitmate                            
                p['yplt']['hidden'] = np.array(self.vars['w'])[plt_dim]
                p['yplt']['mean'] = np.array(self.vars['m'])[plt_dim]
                p['yplt']['sig'] = np.sqrt(np.array(self.vars['s2']))[plt_dim]
                
                # labels and legends
                p['col'] = self.pars['cols']
                # special case for bias:
                if ((plt_dim == 0) and (self.pars['bias-active'])):
                    p['ylabel'] = r'bias $b$'
                    p['legend_labels'] = (r'target $b_{tar}$',
                                      r'estimated $\langle b \rangle$')
                    p['title'] = 'Bias: target vs learned'

                # ordinary weight
                else:
                    p['ylabel'] = r'weight $w$'
                    p['legend_labels'] = (r'target $w_{tar}$',
                                      r'estimated $\langle w \rangle$')
                    p['title'] =  'Weight: target vs learned ' + title_str

        # variable plot that is not a filter plot
        elif lkey in (list(self.vars.keys()) + list(self.out.keys())):
            p['use_filter'] = False
            p['use_legend'] = False
            p['alpha'] = 0.7
            p['size'] = 1
            p['col'] = 'blue'

            # load y-points
            if lkey in self.vars.keys():
                # select time series and dimension
                p['yplt'] = np.array(self.vars[lkey][plt_dim,:]) 
                                    

                # assign labels and title            
                if ((plt_dim == 0) and (self.pars['bias-active'])):

                    if lkey == 'eta':
                        p['ylabel'] = r'Plasticity $\eta$'
                        p['title'] = 'Plasticity gating bias'
                    if lkey == 'Sx':
                        p['ylabel'] = r'Input $S_x$ [Hz]'
                        p['title'] = 'Input firing bias'
                else:
                    # set ordinary synsapses 
                    if lkey == 'eta':
                        p['ylabel'] = r'Plasticity $\eta$'
                        p['title'] = 'Plasticity gating ' + title_str
                    if lkey == 'Sx':
                        p['ylabel'] = r'Input $S_x$ [Hz]'
                        p['title'] = 'Input firing ' + title_str                    
                                                    
            elif lkey in self.out.keys():                                            
                # set output
                p['yplt'] = np.array(self.out[lkey])
            
                if lkey == 'Sy':
                    p['ylabel'] = r'Output $S_y$ [Hz]'
                    p['title'] = 'Output firing'

                if lkey == 'g':
                    p['ylabel'] = r'Firing rate $g$ [Hz]'
                    p['title'] = 'Output firing rate target'

                if lkey == 'gbar':
                    p['ylabel'] = r'Firing rate $\langle g \rangle$ [Hz]'
                    p['title'] = 'Output firing rate estimate'

                if lkey == 'chi':
                    p['ylabel'] = r'Plasticity gate $\chi$'
                    p['title'] = 'Plasticity gate: 0 = no plasticity'

                if lkey == 'delta':
                    p['ylabel'] = r'Error $d\delta$'
                    p['title'] = 'Error signal'


                if lkey == 'u MSE':
                    p['ylabel'] = r'$\langle (u_t - u_t^{tar})^2 \rangle_t$'
                    p['title'] = 'Mean squared error of membrane'
                    
                if lkey == 'w MSE':
                    p['ylabel'] = r'$\langle (w_{t,i} - w_{t,i}^{tar})^2 \rangle_{t,i}$'
                    p['title'] = 'Mean squared error of weights'


            # smooth jumpy signals
            if lkey in ('delta','Sx','Sy','g','gbar','eta'):
                # normalize to firing rate
                if lkey in ('Sx','Sy'):
                    p['yplt'] = p['yplt']/self.pars['dt']
                p['yplt'] = smooth(p['yplt'],L=int(3/self.pars['dt']))
                

        # output firing rate plot
        if lkey == 'output-rates':
            p['use_filter'] = False
            p['use_legend'] = True
            p['multi-plot'] = ['g','gbar','Sy']
            p['alpha'] = 0.7
            p['size'] = 1
            p['col'] = self.pars['cols'] + ['red']
            p['ylabel'] = r'Firing rate [Hz]'
            p['title'] = 'Output firing rates'
            p['legend_labels'] = [r'target $g_{tar}$',
                                  r'estimated $\langle g \rangle$',
                                  r'smoothed spikes $S_y$']
                        
            p['yplt'] = {}
            for k in p['multi-plot']:
                # load dict                
                dt = self.pars['dt']
                
                # load 
                p['yplt'][k] = np.array(self.out[k])                
                # normalize output spikes
                if k == 'Sy':
                    p['yplt'][k] = p['yplt'][k]/dt
                # smooth outputs spikes                    
                p['yplt'][k] = smooth(p['yplt'][k], L=int(3/dt))
                        
        return(p)

    def wrap_plt2(self,dims=0):
        """ plot MSE and quick traj """
        # turn into list
        if type(dims) is int:
            dims = [dims]
        
        self.plt('m')
        
        
        

    def wrap_plt(self,keys=None,dims=None):
        """  wrap plotting: 
            inputs: plotting keys ('Sx', 'eta','w','g', 'gbar','Sy','delta',
                                   'u','u_rm','u MSE','w MSE','output-rates')
                    dims: int or iterable
                    
            If no inputs are provided. Plot everything.
        """
                            
        # decide what to plot                    
        plt_keys = {}
        
        # synaptic quantitites
        if keys is not None:
            plt_keys['loc'] = [k for k in keys if k in ('Sx', 'eta','w')]
        else:
            plt_keys['loc'] = ['Sx', 'eta','w']
            
        # dimenions to plot
        if dims is None:
            dims = np.arange(self.pars['dim'])
            
        # turn into list
        if type(dims) is int:
            dims = [dims]
                
        # plot only if there are local keys
        if len(plt_keys['loc']) > 0:
            # plot
            for d_idx in dims:                                        
                self.plt_vars(plt_dim=d_idx,plt_keys=plt_keys['loc'])                       
                plt.show()
                plt.clf()

        # somaitc quantities 
        if keys is not None:
            plt_keys['glob'] = [k for k in keys if k in 
                                    ('g', 'gbar','Sy','delta','u','u_rm',
                                     'u MSE','w MSE','output-rates')]
        else:
            plt_keys['glob'] = ['delta','u','u_rm','u MSE','w MSE','output-rates']
            
        if self.para == 'logNorm-sigm':
            plt_keys['glob'].append('chi')

        # plot
        if len(plt_keys['glob']) > 0:
            self.plt_vars(plt_dim=None, plt_keys=plt_keys['glob'])


    def plt(self,key,dim=0,xlim=None,log=False,save=False):
        
        if self.pars['bayesian']:
            prefix = 'b' 
        else: 
            prefix = str(round(self.vars['s2'][0,0].item(),3))
        
        xplt = self.x_axis()
        

        if 'mdims' not in self.pars:
            v = self.vars
            o = self.out
            idx = dim # idx selects, dim is the actual synapse idx
        else:
            v = self.mvars
            o = self.mout
            idx = list(self.pars['mdims']).index(dim) # given dim, do selctor                

        if key == 'wm': # slow
            if 'mdims' in self.pars:
                print('Sorry: wm plot not available for mdims setting.')
            else:
                self.wrap_plt(keys=['w'],dims=[dim])

        elif ((key == 'wmq') or (key == 'log wmq')): # quick
            
            if key == 'wmq':
                yerr = np.array(T.sqrt(v['s2'][idx,:]))
                yplt = np.array(v['m'][idx,:])
                yplt_true = np.array(v['w'][idx,:])
                ylab = r'log weight $\mu(t)$'
            else:
                # compute M, S2 then plot.            
                yplt = np.array(T.exp(v['m'][idx,:] + 0.5*v['s2'][idx,:]))
                yerr = np.array(yplt*T.sqrt( T.exp(v['s2'][idx,:]) - 1))
                yplt_true = np.array(T.exp(v['w'][idx,:]))
                ylab = 'weight m(t)'
            
            plt.plot(xplt,yplt,'b',label='estimated')
            plt.plot(xplt,yplt+yerr,':b')
            plt.plot(xplt,yplt-yerr,':b')
            plt.plot(xplt,yplt_true,'orange',label='target')
            plt.xlabel('time [s]')
            plt.ylabel(ylab)        
            tit_tuple = (dim,self.pars['nu'][dim])
            tit = r'Weight: target vs learned $\nu_{%r} = %.1f$' % tit_tuple
            plt.title(tit)
            if xlim:
                plt.xlim(xlim)
            if save:
                plt.savefig('./output/' + prefix + key + '_%r_%.2f.png' % (dim,self.pars['nu'][dim]),
                            bbox_inches='tight',dpi = 300)
                                    
        elif key == 'wi MSE':
            yplt = np.array(T.pow(v['w'][idx,:] - v['m'][idx,:],2))
            plt.plot(xplt,yplt)
            plt.ylabel(r'MSE $w_{%r}$' % dim)
            plt.xlabel('time [s]')     

            tit_tuple = (dim,self.pars['nu'][dim])
            tit = r'Weight: MSE $\nu_{%r} = %.1f$' % tit_tuple
            plt.title(tit)
            if xlim:
                plt.xlim(xlim)
            if save:
                plt.savefig('./output/' + prefix + 'wMSE_%r_%.2f.png' % (dim,self.pars['nu'][dim]),
                            bbox_inches='tight',dpi = 300)
            
        else:
            if key == 's':
                key = 's2'
                flag_s = True
            else:
                flag_s = False
            
            if key in o.keys():
                yplt = o[key]
            elif key in v.keys():
                print('print dim',dim)
                if flag_s: # take sqrt to get "s" instead of "s2"
                    yplt = T.sqrt(v[key][idx,:])
                    key = 's' # change back
                else:
                    yplt = v[key][idx,:]                   
                                                            
            if log:
                yplt = T.log10(yplt)                  
            
            yplt = np.array(yplt)
            
            plt.plot(xplt,yplt)
            plt.ylabel(key)
            plt.xlabel('time [s]')        
            
            if xlim:
                plt.xlim(xlim)
            if save:
                plt.savefig('./output/' + prefix + '_' + key + str(dim) + '.png',
                            bbox_inches='tight',dpi = 300)
                
            return(yplt)
                        
        
def plt_maxSyn(k,S,N_max=5,plot=False):
    """ at time k find N_max largest synapses in S and return indices """
    
    m_k = np.array(S.vars['m'][:,k])
    
    idxs_max = m_k.argsort()[-N_max:][::-1]
    
    if plot:
        S.wrap_plt(keys=['w'],dims=idxs_max)
        
    return(idxs_max)
        

# =============================================================================
# experiment with synapse
# =============================================================================

if __name__== "__main__":

    benchmark = False
    eta_search = False
    recover = False
    new = False
    test = False
    ptest = False
    iter_test = False
    syn_test = True
    
    
    if syn_test:
        
        para = 'Peter'
        bayesian = True
        cov0 = None #0.3
        dt = 0.005
        dim = 1000
        #mdims = plt_idxs(plt_num=5,dim=dim)        
        #mdims = [0,4,10,200]
        mdims = None
        opt = 5
        step_max = 48816
        step_max = 50000
        
        init_dict = {}
        for var in ('m','s2','w'):
            init_dict[var] = load_obj('init_'+var,'./../aux/')
                        
        pars_extra = {'PL':{'ON':True,
                            'Sample':True,
                            'alpha':1,
                            'beta':-1,
                            'opt':opt,
                            'b_samp':False},
                      'OU':{'tau':2*10**3,'mu':-0.669},
                      'step_out':1, # down sample
                      'th':-0,
                      'tau_u':0.01,
                      'g0':60,
                      'time_between_output':60,
                      'plot-flags': {'save-output':True},
                      'mu_bounds':None, # works
                      'eps_smooth':False # absolutely not working                      
                      }            
                            
        syn = Synapse(name='', para=para, dim=dim, steps=step_max, dt=dt, mdims=mdims)
    
        # init parameters    
        pars_extra.update({'bayesian':bayesian})        
        syn.set_pars(pars_user=pars_extra)
            

        # init initial conditions    
        if cov0:
            syn.set_vars(init_dict={'s2':cov0})
        elif init_dict:
            syn.set_vars(init_dict=init_dict)
        else:
            syn.set_vars()
        syn.run()
        syn.get_MSE()
        syn.plt('w MSE')
        

    if ptest:
        syn = wrap_run(dim=500,para='Peter',bayesian=False,step_max=25000,dt=0.005,
             pars_extra={'g0':60,'OU':{'tau':2000},
             'PL':{'ON':True,'Sample':True,'alpha':1,'beta':-1,'opt':2}},
             cov0=0.5,hidden=None, gen_hidden=False,prefix='')
                         #'nu':np.array([4.,4.]),                                    
    if iter_test:
        simulate = lambda d,b,cov0: wrap_run(dim=d,para='Peter',
                                     bayesian=b,step_max=250000,dt=0.003,
                                     pars_extra={'g0':60,'OU':{'tau':2000},
             'PL':{'ON':True,'Sample':True,'alpha':1,'beta':-1,'opt':1}},
                                                                 cov0=cov0)
        
                         
        
    if test:
        wrap_run(dim=10,para='logNorm-sigm',bayesian=False,step_max=240000,dt=0.003,
             pars_extra={'g0':60,'OU':{'tau':100},'nu':np.array([4.,4.])},                         
                         cov0=0.1,hidden=None, gen_hidden=False,prefix='')
        
    if new:                
        dim = 100
        para = 'Peter'
        bayesian = True
        dt = 0.001

        nu_input = np.sort(np.exp(0.5*np.log(10)*np.random.randn(dim)))[::-1] #,
                                              #dtype=T.float32),descending=True)
        
        S = Synapse('Peter',para=para,dim=dim,dt=dt,steps=5*10**5) # 240000
        S.set_pars({'PL':{'ON':True,'Sample':True,'alpha':1,'beta':-1},
                    'OU':{'tau':5*10**3,'mu':-0.669}, 
                    'step_out':10, # down sample
                    'th':-0, 
                    'tau_u':0.01,
                    'g0':20,
                    'time_between_output':30, #})
                    'nu' : nu_input})
    
        total_steps = S.pars['xSteps']*S.pars['step_out']
        steps_per_hour = 2*10**6
        
        print('Computation steps:',total_steps)
        print('Output steps:',S.pars['xSteps'])
        print('Expected simulation time [h]:',total_steps/steps_per_hour)
        print('Simulated time [min]:',total_steps*dt/60)
        print('tau OU in [min]:',S.pars['OU']['tau']/60)        
        
            
        S.set_vars()
        S.run()
        res = S.get_MSE()
        
        print('augment dt for plotting')
        S.pars['dt'] *= S.pars['step_out']
        
        plt_maxSyn(S.k,S,N_max=5,plot=True)
#        print('Total fract',res['<P(dw_i < sig_i)>_i'])
        xplt = S.x_axis()
        for i in ([i for i in range(5)] + [20,50,99]):
#            print('Fraction within',res['q'][i])
            p = {}
            p['yplt'] = {'mean': np.array(S.vars['m'][i,:]) , 
                     'sig':np.array(T.sqrt(S.vars['s2'][i,:])), 
                         'hidden':np.array(S.vars['w'][i,:])}            
            plt_filter(xplt,p['yplt'])
            plt.xlabel('time [s]')
            plt.ylabel('log weight')
#            S.plt('wm',i)
#            S.wrap_plt(dims = [i] ,keys=['w'])
            plt.savefig('log-synapse' + str(i) + '.png')
            plt.show()
            
            Ms = np.array(T.exp(S.vars['m'][i,:] + 0.5*S.vars['s2'][i,:]))
            Ss = np.array(T.sqrt(Ms**2*(T.exp(S.vars['s2'][i,:]) - 1)))
            w = np.array(T.exp(S.vars['w'][i,:]))

            p['yplt'] = {'mean': Ms , 'sig': Ss, 'hidden':w}            
            plt_filter(xplt,p['yplt'])
            plt.xlabel('time [s]')
            plt.ylabel('weight')
            plt.savefig('synapse' + str(i) + '.png')
            plt.show()                        
            
#        sig2_u = S.plt('sig2_u')
#        plt.show()
        S.plt('u')
        plt.plot(np.array(S.x_axis()),np.sqrt(np.array(S.out['sig2_u'])),c='orange')
        plt.show()
        eta = S.plt('eta',0,log=False)
        
        
#        plt.xlim([0,2000*dt])
        
#    plt.scatter(np.arange(len(u)),u/np.sqrt(sig2_u),alpha=0.01)
#    plt.scatter(np.arange(len(u)),uu,alpha=0.01)
    
    if recover:
        
        dim = 2
        para = 'logNorm-sigm'
        bayesian = True
        dt = 0.003
        
        S = Synapse('recover-JPP',para=para,dim=dim,dt=dt,steps=240000)
        S.set_pars({'g0':60,'OU':{'tau':100}, 'th':-0.0172594138014})
        S.set_vars()
        S.run()        
        
        S.wrap_plt(dims = [665],keys=['w'])

    
    
    
    if benchmark:
        pred_code = False
        
        dims = [1,2,3,5,10,20,30,50,100,200,500,1000]
        dims = [2]
        res = []    
        for dim in dims:
            print(dim)
            S = Synapse('peter',para='Peter',dim=dim,dt=0.001,steps=50000)
            S.set_pars({'PL':{'ON':pred_code,'Sample':pred_code,
                              'alpha':1,'beta':-1},
                        'g0':20,'OU':{'tau':25}, 'th':-0, 'tau_u':0.01})
            S.set_vars()
            S.run()
            res.append(S.get_MSE()['w MSE'])
    
    #    plt.plot(np.log10(np.array(dims)),res,':o')
    #    plt.xlabel(r'$\log_{10}$ dim')
        plt.plot(np.array(dims),res,':o')
        plt.xlabel('learning rate')
        plt.ylabel('MSE w')
        plt.title('MSE w averaged in time and dim')
        plt.tight_layout()
    #    plt.savefig('MSEwlog.pdf',dpi=400)
    #    plt.savefig('MSEw.pdf',dpi=400)
        S.wrap_plt(dims = [0,1])
        
        

    if eta_search:
        
        # check if traj are truly similar. 
        # check if non-Bayesian learning works at all
        # plot all traj into one plot?
        # check if printing works as expected        

        # run Bayesian
        dim = 100
        para = 'Peter'        
        dt = 0.001
        step_max = 1*10**3
        
        nu_input = np.sort(np.exp(0.5*np.log(10)*np.random.randn(dim)))[::-1] #,        
        #dtype=T.float32),descending=True)            
        pars_extra = {'PL':{'ON':True,'Sample':True,'alpha':1,'beta':-1},
                    'OU':{'tau':5*10**3,'mu':-0.669}, 
                    'step_out':1, # down sample
                    'th':-0, 
                    'tau_u':0.01,
                    'g0':20,
                    'time_between_output':30,
                    'nu' : nu_input, 
                    'plot-flags': {'save-output':True}
                    }

        # Classical synapse test
#        syn = wrap_run(dim,para,bayesian=False,step_max=step_max,dt=dt,
#                 pars_extra=pars_extra,cov0=None,hidden=None,
#                 gen_hidden=False,prefix='_genHidden')

        # remember hidden from Bayesian
        hidden = wrap_run(dim,para,bayesian=True,step_max=step_max,dt=dt,
                 pars_extra=pars_extra,cov0=None,hidden=None,
                 gen_hidden=True,prefix='_genHidden')

        # get error of bayesian synapse
        syn = wrap_run(dim,para,bayesian=True,step_max=step_max,dt=dt,
                 pars_extra=pars_extra,cov0=None,hidden=hidden,
                 gen_hidden=False,prefix='')
        
        BS_err = syn.res['w MSE']
                
        # loop over learning rates
        out_dict = []
        # output dictionary
        covs0 = eps_lin(N=10,eps_min=0.001,symmetric=False,eps_anker=1.0)
                                    
        # run over learning rates

        for cov0 in T.tensor(covs0,dtype=T.float32):
            
            syn = wrap_run(dim,para,bayesian=False,step_max=step_max,dt=dt,
                 pars_extra=pars_extra,cov0=cov0,hidden=hidden,
                 gen_hidden=False,prefix=str(cov0.item())[:6]+' ')
        
            out_dict.append({'cov0':cov0.item(),
                             'Bayesian': BS_err,
                             'Classical': syn.res['w MSE']
                            })
            
        # out dict
        df = pd.DataFrame(out_dict)
        
        df.plot(x='cov0',y=['Bayesian','Classical'],logx=True)
        plt.ylabel('MSE')
        
        

        
        
    
