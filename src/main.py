#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import itertools as it
import os
import matplotlib
matplotlib.use('Agg',warn=False)
from matplotlib import pyplot as plt
import time
from BayesianSynapse.Synapse import wrap_run, Synapse, plt_maxSyn
import torch as T
import pandas as pd
from helpers.helpers import eps_lin, save_obj, load_obj, set_status, get_status,plt_idxs
from my_plotters.my_plotters import plt_filter
import argparse
import shutil

def wrap_run2(dim,para,bayesian,step_max,dt=0.005,pars_extra={},k_samp=None,
             cov0=None,prefix='',plt_num=3,mdims=None,init_dict=None):
    """ create a synapse, run it, plot it OR generate hidden from it """

    # init synapse
    if bayesian:
        name = prefix + para + '_b_'
    else: 
        name = prefix + para + '_c_' + str(cov0)[:6]
                
    syn = Synapse(name=name, para=para, dim=dim, steps=step_max, dt=dt, mdims=mdims)
    #syn.set_pars(pars_user=pars_extra)
    #print(pars_extra)
    # init parameters: sigV_0 such that useful
    #sigV_pi = float(np.sqrt(syn.get_sigV2_pi()))
    
    # update sig0_u parameter
    #print("set sigV_pi and scale th=%f by sigV_pi=%f, yielding th'=%f" 
    #      % (syn.pars['th'],sigV_pi,syn.pars['th']*sigV_pi))
    
    pars_extra.update({'bayesian':bayesian})
                       #'sig0_u':sigV_pi,
                       #'th':syn.pars['th']*sigV_pi 
                           
    syn.set_pars(pars_user=pars_extra)    
    #print(pars_extra)
    
    if k_samp is not None:
        syn.pars['PL']['k_samp'] = k_samp
        print('read in k_samp',k_samp)

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

    # run                            
    syn.run()
    syn.get_MSE()
    
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
            
    
    
def get_par_vec(eps_num,eps_min,eps_anker,k_num,k_min,k_max,b_samp):
    """ create the 'agenda': output = (cov0, k, bayesian_samp flag) """
        
    if eps_num > 1:
        cov_lin = np.round(np.linspace(eps_min,eps_anker,eps_num),5)
    else:
        print('eps_num = ',eps_num,'use eps_anker = ',eps_anker)
        cov_lin = np.array([eps_anker])

    if k_num > 1:
        k_lin = np.round(np.linspace(k_min,k_max,k_num),5)
    else:                
        print('k_num = ',k_num,'use k_max = ',k_max)
        k_lin = np.array([k_max])

    # add pure bayesian step, always        
    par_vec = [(None,None,b_samp)]

    if b_samp == True:
        # only one bayesian run #n_tot = 1 + eps_num*k_num
        
        for cov0,k in it.product(cov_lin,k_lin):
            par_vec.append((cov0,k,False))                
    else:
        # one bayesian run for each k #n_tot = (1 + eps_num)*k_num        
        cov_lin_with_flag = np.hstack([[-1],cov_lin])
        for cov0_with_flag,k in it.product(cov_lin_with_flag,k_lin):
            if cov0_with_flag == -1:
                par_vec.append((None,k,True))
            else:
                # cov0_with_flag is just cov0
                par_vec.append((cov0_with_flag,k,False))
        
    return(par_vec)


class Plot_Cluster():
    """ for plotting data frames after running on euler """

    def __init__(self,path,id_prefix,par_vec):        

        self.pv = pd.DataFrame(par_vec,columns=['cov0','k','b'])
        self.pv = self.pv[['b','cov0','k']] # align to ordering in df
        self.df = self._load_df(path,id_prefix)
        print('|agenda|/|done| = %r/%r' % (len(self.pv),len(self.df)))
        
        if len(self.pv) > len(self.df):
            print('agenda was larger than finished runs.')

            # search for discarded runs
            #for row in range(len(self.df)):
            #    tup1 = self.df.iloc[row][['bayesian','cov0','k_sample']]
            #    tup2 = self.pv['b','cov0','k']

        # get statistics
        self.b_num = sum(self.df['bayesian'])        
        self.k_num = len(set(list(
                    self.df['k_sample'][self.df['k_sample'].isnull()==False])))
        self.cov_num = len(set(list(
                    self.df['cov0'][self.df['cov0'].isnull()==False])))                                    
#                        output = {'bayesian': bayesian, 
#                          'cov0': cov0,
#                          'k_sample': k,        
        
        
    def _load_df(self,path,id_prefix):
        """ read in all the data frames from all runs """            
        ls2df = []
        for file in os.listdir(path):
            if (file.endswith(".pkl") and file.startswith(id_prefix)):            
                ls2df.append(load_obj(file[:-4],path)) # ,path=os.getcwd() + "/output/")
        return(pd.DataFrame(ls2df))

    
    def plt(self,vs_k=False):
        """ wrapper """
        
        if vs_k==False:
            # easy case: absolute error and unique bayesian MSE
            if self.b_num == 1:
                self._plt_one_bayesian()
            elif self.b_num > 1:
                self._plt_many_bayesian()
        else:
            vs_k==True
            self._plt_MSE_vs_k()
                            
        plt.show()
            
            
            

    def _plt_one_bayesian(self):
        """  plt MSE against cov0 """
        print('one bayesians, k_num:',self.k_num)
        
        aux = self.df[['MSE','k_sample','cov0']][self.df['bayesian']==False]
        
        for k in set(list(aux['k_sample'])):            
            xplt,yplt = aux[['cov0','MSE']][aux['k_sample']==k].values.T
            # sort for plotting
            idxs_sorted = xplt.argsort()
            
            plt.plot(xplt[idxs_sorted],yplt[idxs_sorted],label='k_sample = '+str(k))
                              
        yplt = self.df['MSE'][self.df['bayesian']]*np.ones(len(xplt))
        plt.plot(xplt[idxs_sorted],yplt,label='bayesian')
        
        # make pretty
        plt.xlabel(r'learning rate $\eta$')
        plt.ylabel('MSE')        
        plt.gca().legend()        


    def _plt_many_bayesian(self):
        """  plt MSE against cov0, normlize by respective bayesian """
        print('num bayesians:',self.b_num)
        
        aux = self.df[['MSE','k_sample','cov0','bayesian']]
                
        for k in set(list(aux['k_sample'])):           
            xplt,yplt_all,b = aux[['cov0','MSE','bayesian']][aux['k_sample']==k].values.T                                            
            # b = bayesian label
            
            # split along bayesian and classical
            xplt = xplt[b==False]
            # bayesian
            yplt = yplt_all[b==True]*np.ones(len(xplt))
            # classical
            yplt_c = yplt_all[b==False]/yplt
            
            # sort for plotting
            idxs_sorted = xplt.argsort()                        
            plt.plot(xplt[idxs_sorted],yplt_c[idxs_sorted],
                                 label='k_sample = '+str(k))
                                      
        plt.plot(xplt[idxs_sorted],np.ones(len(xplt)),label='bayesian')
        
        # make pretty
        plt.xlabel(r'learning rate $\eta$')
        plt.ylabel('normalized MSE')
        plt.gca().legend()        
        
        

    def _plt_MSE_vs_k(self):
        """ plt best MSE against k """        
        
        # obtain MSE as function of k_sample
        aux = self.df[['MSE','k_sample']][self.df['bayesian']==False]
        # for each k, obtain maximum:
        aux = aux['MSE'].groupby(aux['k_sample']).min()
        
        # read out
        xplt_c = np.array(aux.index)
        yplt_c = aux.values

        if self.b_num == 1:
            # bayesian
            yplt = np.ones(len(xplt_c))*self.df[self.df['bayesian']]['MSE']
                                                                        
        # more than one bayesian:
        elif self.b_num > 1:
            # bayesian
            xplt = self.df[self.df['bayesian']]['k_sample']
            yplt = self.df[self.df['bayesian']]['MSE']
                    
        # now plot
        plt.plot(xplt_c,yplt_c,label='classic')
        plt.plot(xplt,yplt,label='bayesian')

        # make pretty
        plt.xlabel('sampling parameter $k$')
        plt.ylabel('lowest MSE')        
        plt.gca().legend()        

        #plt.show()        
        

        

def get_agenda(N):    
    agenda = []
    agenda.append("-s gen -N %d" % N)
    for i in np.arange(N):
        agenda.append("-s main -N %d -i %d" % (N,i))
    agenda.append("-s post -N %d" % N)
    return(agenda)

wdir='/Users/jannes/Documents/Science/Real-Projects/peter_latham-paper/source-simulations/cluster_runner/src'
filepath = wdir + '/main.py'
agenda = get_agenda(10)
"""
agenda = get_agenda(1)
[ runfile(filepath,wdir=wdir,args=arg) for arg in agenda]
"""

if __name__== "__main__":
    """ i = None: create hidden
        i = 0: bayesian
        i > 0: classical synapse, i is the index in an array
    """

    # pars arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage","-s", help="pre run, main run or post run")
    parser.add_argument("--option","-o", help="special options in the main file to run single synapse")
    #   parser.add_argument("--classic","-c", help="classic (default: bayesian)") # taken out
        
    parser.add_argument("--index","-i", help="index is csv file to set learning rate (default=0 -> Bayesian)",type=int)
    parser.add_argument("--steps","-T", help="number of time steps to run (default: 5e5)",type=int)
    parser.add_argument("--dim","-d", help="number of synapses (default: 100)",type=int)
    parser.add_argument("--parnum","-N", help="number of grid points to loop over",type=int) # NOW: number of runs

    args = parser.parse_args()
    my_args = [(k,v) for k,v in sorted(vars(args).items())]
    print('')
    print('----------')
    print(my_args)

    # my paths
    file_path = os.path.realpath(__file__)
    root = file_path[:-file_path[::-1].find('/')]
    MP = {'root' : root}
    
    for k in ('aux','output','figs'):
        MP[k] = MP['root'] + k  +  '/'
        if not os.path.exists(MP[k]):
            os.makedirs(MP[k])
    # file locations
    MP['nu_input'] = MP['aux'] + 'nu_input'
    MP['hidden'] = MP['aux'] + 'hidden'
    MP['covs0'] = MP['aux'] + 'covs0'

#    """ 
#    
#        WARNING TURN THIS OFF AGAIN OTHERWISE IT KILLS YOUR SIMS
#    
#    """
#    MP['output'] = '/Users/jannes/Documents/Science/Real-Projects/peter_latham-paper/source-simulations/cluster_runner/26-Oct-2018-long/output/'



#    print('use different output dict')

    dim = args.dim if args.dim else 1000
    steps = args.steps if args.steps else 6*10**6 # 5
#    steps = args.steps if args.steps else 1*10**3 # 5

    # parameters to loop over (subtract because of bayesian stuff)

    #bayesian = False if args.classic else True
    # index going from 0 - 10
    
    # General stuff
    #gen_plot_arrays=False # no plotting just arrays w,m,s2
    
    #dim = 100
    #eps_anker = 0.3
    eps_anker = 0.52
    eps_min = 0.03

    k_min = 0
    k_max = 0.5

    # Important: To set up the parameter vector
    eps_num = 26
    k_num = 6
    
    N = args.parnum if args.parnum else 165
    b_samp = False # false = draw from filtering distribution not k*M

    step_max = steps #1*10**3
    para = 'Peter'
    dt = 0.001
    mdims = plt_idxs(plt_num=4,dim=dim)    # TODO change
    opt = 2
    pars_extra = {'PL':{'ON':True,'Sample':True,'alpha':-1,
                        'beta':1,'opt':opt,'b_samp':b_samp},
                        'k_samp': 0.0877,
                  'OU':{'tau':2*10**3,'mu':-0.669},
                  'step_out':1, # down sample
                  'th':-2.1280, # tuned  to g0 60 and E[v] = 0, Sig2 = [1]
                  'tau_u':0.01,
                  'g0':60, # max rate
                  'time_between_output':3600,
                  'plot-flags': {'save-output':True},
                  'mu_bounds':3                  
                  }
        
    N_down_sample = 10
        
    
    if args.stage=='gen':
        """ go to equlibrium distribtuion and create quantities """
        dt_gen = dt*5
        steps = int(pars_extra['OU']['tau']/dt_gen)

        stat = 'fail'
        #set_status(1)
        
        try:
            nu_input = np.sort(np.exp(0.5*np.log(10)*np.random.randn(dim)))[::-1] #,
            save_obj(nu_input,'nu_input', MP['aux'])
                                    
            # set up the parameter vector here:
            par_vec = get_par_vec(eps_num,eps_min,eps_anker,k_num,k_min,k_max,
                                                                        b_samp)            
            # save par_vec
            save_obj(par_vec,'par_vec',MP['aux'])
                        
            # print missing parametes:            
            print('Agenda (cov0,k_samp,Bayesian):',par_vec)
            print('Simulate fraction: %r/%r' % (N,len(par_vec)))     
                                 
            pars_extra.update({'nu' : nu_input})
            stat = 'successfully '

        finally:
            print(stat + 'finished pre, start burn in')
            print('')

        # run    
        syn = wrap_run2(dim,para,bayesian=True,step_max=steps,dt=dt_gen,
                         pars_extra=pars_extra,mdims=None)

        # save weights, filter
        for var in ('m','s2','w'):
            save_obj(syn.vars[var][:,-1],'init_'+var,MP['aux'])
                
        # determine sigV0 --> not needed anymore, now done internally
        #sig0_u = T.sqrt(T.mean(syn.out['sig2_u'])).item()
        #save_obj(sig0_u,'sig0_u',MP['aux'])                 
    

    if args.stage=='main':        
        #if get_status() == 0:
            #print(N+1)
            #set_status(N+1) # the extra bayesian run +1        
            #print('current status',get_status())
        stat = 'fail '
        
        #try:
        if 1:
            # load stuff
            #print(os.getcwd())            
            try:
                nu_input = load_obj('nu_input',MP['aux'])
            except:
                print('main: generate nu input')
                nu_input = np.sort(np.exp(0.5*np.log(10)*np.random.randn(dim)))[::-1] #,        

            try:
                init_vars = {}
                for var in ('m','s2','w'):
                    init_vars[var] = load_obj('init_'+var,MP['aux'])
                print('successfully loaded init cond dict')
            except:
                print('failed to load init cond dict')
                init_vars = None     
                
            #try:
                #sig0_u = load_obj('sig0_u',MP['aux'])
                #pars_extra.update({'sig0_u' : sig0_u})                
                #print('loaded sig0_u, scale up: th -> th * sig0_u ')                                
            #except:
                #print('didnt find sig0_u to load')                                
                
            pars_extra.update({'nu' : nu_input})

            par_vec = load_obj('par_vec',MP['aux'])            

            # start simulations only if parameters are available                
            if len(par_vec) > args.index:
                print('running index',args.index,'with cov0,k,b=',par_vec[args.index])
                cov0,k,bayesian = par_vec[args.index]
                                
                # run                                                    
                syn = wrap_run2(dim,para,bayesian=bayesian,step_max=step_max,dt=dt,
                                k_samp=k,pars_extra=pars_extra,cov0=cov0,
                                prefix=str(args.index),mdims=mdims,
                                init_dict=init_vars)
                        
                # output dict
                output = {'bayesian': bayesian, 
                          'cov0': cov0,
                          'k_sample': k,
                          'MSE': float(syn.res['w MSE']), 
                          'inside': float(syn.res['q'])}                        
                print(output)            
                
                # save entire (downsampled time series)            
                output.update({'k':None, 
                       'MSE series': np.array(syn.mout['w MSE'][::N_down_sample])})
    
                # non bayesians have a fixed k
                if bayesian==True:
                    output.update({'k': np.array(syn.vars['k'])})
                    print('E[k] = ',output['k'].mean())
                    
                # write error to file
                save_obj(output, 'df_id_'+ str(args.index), MP['output'])
                stat = 'successfully '
            else:
                print('no more parameters left')
                stat = 'skipped '
                
        #finally:
            
         #   print(stat + 'finished main: bayesian='+str(bayesian)+' idx=' + str(args.index))
          #  print('')
            # reduce status by one, the last job will make it zero
            #set_status(get_status()-1)                        
        # python ./main.py -s pre -N 3
                
            
    if args.stage=='post':
                
        # what I need: 
        # (1) if b_samp=True: absolute errors against bayesian.
        # (2a) if b_samp=False, then do a plot per k and normalize to b.
        # (2b) do a normalized plot, everything to one. and a classical parabola per k
        # (2c) plot the minima of the parabolas against k and bayesiaen MSE as fct of k. 
        
        copy_from = ('./BayesianSynapse/Synapse.py', './main.py')
        copy_to = (MP['output'] + 'Synapse.py', MP['output'] + 'main.py')
        for cf,ct in zip(copy_from,copy_to):                    
            shutil.copyfile(cf,ct)                     
        
        # runfile('/Users/jannes/Documents/Science/Real-Projects/peter_latham-paper/source-simulations/cluster_runner/src/main.py', wdir='/Users/jannes/Documents/Science/Real-Projects/peter_latham-paper/source-simulations/cluster_runner/src',
        # args="-s post -N 10")
        #ssh jegminat@euler.ethz.ch
    
        # go to interesting path
        #os.chdir(os.getcwd() + '/../output/')
        #print(os.getcwd())

    
        # load all files
        pc = Plot_Cluster(MP['output'],'df_id_',par_vec)
        #df = load_df(MP['output'],id_prefix='df_id_')
        
        if b_samp == True: # no ks for Bayesian
            plt_MSE = True
            plt_MSE_series = False
                    
        elif b_samp == False: # many ks for Bayesian
            plt_minMSE_vs_k = True
            plt_MSE_per_k_normed = True
            plt_MSE_per_k_many_plots = True

                                   
#        if plt_MSE:
#            print('plot absolute MSE one plot')    
            
            # TODO TODO TODO: make a class for plotting all the desired stuff.
            # check out df and how to handle it. I also have par_vec
            
            
            
            # save
#            coln = 'MSE' # short'
#            A = []
#            for i in df.index:
#                A.append(np.mean(df['MSE series'].iloc[i][50000:400000])/dim)
#            df[coln] = np.array(A)
#            df.to_pickle(MP['output'] + 'dataframe')

#        #    df = pd.read_csv("./../rate_vs_mse.csv")
#            df = df.sort_values('cov0')
#            df = df.reset_index(drop=True)
#            print(df)
#            
#            yplt = {}
#            xplt = df.iloc[1:]['cov0'].values
#            yplt['b'] = np.ones(len(df) - 1)*df.iloc[0][coln]
#            yplt['c'] = df.iloc[1:][coln].values
#                                    
##            plt.plot(np.log10(xplt),yplt['b'],':o',label='b')
##            plt.plot(np.log10(xplt),yplt['c'],':o',label='c')            
#            plt.plot(xplt,yplt['b'],':o',label='b')
#            plt.plot(xplt,yplt['c'],':o',label='c')
#            
#            plt.gca().legend()
#            #df.plot(x='cov0',y=['b','c'],logx=True,)
#            plt.ylabel(coln)
#            #plt.xlabel('log10 cov0')
#            plt.xlabel('eta')
#            plt.ylim([0.6,1.0])
#
#            plt.savefig(MP['output'] + coln + ".png")
#            plt.show()            
#
#    
#            #cum_mean = lambda arr,L: np.cumsum(arr[::L])/np.arange(1,1+len(arr))
#            cum_mean = lambda arr,L: arr
#            #cum_mean = lambda arr,L: ndimage.gaussian_filter(arr,L)
#            L = 1
#        
#            if plt_MSE_series:
#                i_0,i_f,i_step = (0,-1,100)
#            #def mse(i_0,i_f,i_step,L=1):
##                for ii in range(len(df)):
#                for ii in [0,2,8]:
#                    lab = df.iloc[ii]['cov0']
#                    yplt = df.iloc[ii]['MSE series'][i_0:i_f:i_step]
#                    xplt = np.arange(len(yplt))*i_step*dt*N_down_sample/3600
#                    #plt.plot(xplt,cum_mean(yplt),label=str(np.round(lab,3)))
#                    plt.plot(xplt,cum_mean(yplt,L),label=str(np.round(lab,3)))
#                    
#                plt.xlabel('Time [h]')
#                plt.ylabel('MSE')
#                plt.title('Cumulative MSE.')
#                plt.gca().legend(ncol=1)
#                plt.savefig('./output/MSE_over_time.pdf')
#                plt.show()
#                                    

    

                            


