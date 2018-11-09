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



def get_agenda(N,use_hidden=True):
    h = ' -o hidden' if use_hidden else ''
    agenda = []
    agenda.append(("-s pre -N %d" % N) + h)
    agenda.append("-s main" + h ) # bayesian
    
    for i in np.arange(N):
        agenda.append(("-s main -N %d -i %d" % (N,i)) + h + " -c 1")

    agenda.append("-s post -N %d" % N)
    return(agenda)

wdir='/Users/jannes/Documents/Science/Real-Projects/peter_latham-paper/source-simulations/cluster_runner/src'
filepath = wdir + '/main.py'
agenda = get_agenda(2,0)
"""
agenda = ["-s pre -N 7 -o hidden",
        "-s main -o hidden",
        "-s main -i 0 -c 1 -N 7 -o hidden",
        "-s main -i 1 -c 1 -N 7 -o hidden",
        "-s main -i 2 -c 1 -N 7 -o hidden",
        "-s main -i 3 -c 1 -N 7 -o hidden",
        "-s main -i 4 -c 1 -N 7 -o hidden",
        "-s main -i 5 -c 1 -N 7 -o hidden",
        "-s main -i 6 -c 1 -N 7 -o hidden",
        "-s post -N 7"]
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
    parser.add_argument("--classic","-c", help="classic (default: bayesian)") # check!!
        
    parser.add_argument("--index","-i", help="index is csv file to set learning rate (default=0 -> Bayesian)",type=int)
    parser.add_argument("--steps","-T", help="number of time steps to run (default: 5e5)",type=int)
    parser.add_argument("--dim","-d", help="number of synapses (default: 100)",type=int)
    parser.add_argument("--parnum","-N", help="number of grid points to loop over",type=int)

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
#    steps = args.steps if args.steps else 6*10**6 # 5
    steps = args.steps if args.steps else 1*10**3 # 5

    # parameters to loop over (subtract because of bayesian stuff)
    N = args.parnum if args.parnum else 20
    bayesian = False if args.classic else True
    # index going from 0 - 10
    
    # General stuff
    gen_plot_arrays=False # no plotting just arrays w,m,s2
    
    #dim = 100
    #eps_anker = 0.3
    eps_anker = 0.52
    eps_min = 0.03
    #eps_min = 0.1

    step_max = steps #1*10**3
    
    para = 'Peter'
    dt = 0.01
    mdims = plt_idxs(plt_num=4,dim=dim)    
    opt = 2
    b_samp = True
    pars_extra = {'PL':{'ON':True,'Sample':True,'alpha':-1,'beta':1,'opt':opt,'b_samp':b_samp},
                  'OU':{'tau':2*10**3,'mu':-0.669},
                  'step_out':1, # down sample
                  'th':-0,
                  'tau_u':0.01,
                  'g0':1,
                  'time_between_output':3600,
                  'plot-flags': {'save-output':True},
                  'mu_bounds':3                  
                  }
        
    N_down_sample = 10
    
    if args.stage=='gen':
        """ go to equlibrium distribtuion """
        steps = pars_extra['OU']['tau']/dt

        stat = 'fail'
        #set_status(1)
        
        try:
            print(os.getcwd())
            nu_input = np.sort(np.exp(0.5*np.log(10)*np.random.randn(dim)))[::-1] #,        
            save_obj(nu_input,'nu_input', MP['aux'])
            
            #covs0 = eps_lin(N=N,eps_min=eps_min,symmetric=False,eps_anker=eps_anker)        
            covs0 = np.linspace(eps_min,eps_anker,N)                
            save_obj(covs0,'covs0',MP['aux'])                    
            pars_extra.update({'nu' : nu_input})
            stat = 'successfully '

        finally:
            print(stat + 'finished pre, start burn in')
            print('')

        pars_extra.update({'nu' : nu_input})  

        # run    
        syn = wrap_run(dim,para,bayesian=True,step_max=steps,dt=dt,
                         pars_extra=pars_extra,mdims=None)

        # save weights, filter
        for var in ('m','s2','w'):
            save_obj(syn.vars[var][:,-1],'init_'+var,MP['aux'])
                
        # determine sigV0
        sig0_u = T.sqrt(T.mean(syn.out['sig2_u'])).item()
        save_obj(sig0_u,'sig0_u',MP['aux'])                    
    
    # create stuff                        
    if args.stage=='pre':
        
        stat = 'fail'
        #set_status(1)
        
        try:
            print(os.getcwd())
            nu_input = np.sort(np.exp(0.5*np.log(10)*np.random.randn(dim)))[::-1] #,        
            save_obj(nu_input,'nu_input', MP['aux'])
            
            #covs0 = eps_lin(N=N,eps_min=eps_min,symmetric=False,eps_anker=eps_anker)        
            covs0 = np.linspace(eps_min,eps_anker,N)                
            save_obj(covs0,'covs0',MP['aux'])
                    
            pars_extra.update({'nu' : nu_input})
                                    
            if args.option=='hidden':
                # remember hidden from Bayesian
                hidden = wrap_run(dim,para,bayesian=bayesian,step_max=step_max,dt=dt,
                         pars_extra=pars_extra,cov0=None,hidden=None,
                         gen_hidden=True,prefix='_genHidden',
                         gen_plot_arrays=gen_plot_arrays, mdims=mdims)            
                ## create hidden
                save_obj(hidden,'hidden',MP['aux'])    
            
            time.sleep(1)
            #set_status(get_status()-1)
            set_status(0)
            stat = 'successfully '
        finally:
            print(stat + 'finished pre')
            print('')
            
    if args.stage=='main':        
        #if get_status() == 0:
            #print(N+1)
            #set_status(N+1) # the extra bayesian run +1        
            #print('current status',get_status())
        stat = 'fail '
        
        try:
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
            try:
                sig0_u = load_obj('sig0_u',MP['aux'])
                pars_extra.update({'sig0_u' : sig0_u})
                print('loaded sig0_u')
            except:
                print('didnt find sig0_u to load')
                
                
            pars_extra.update({'nu' : nu_input})
            hidden = load_obj('hidden',MP['aux']) if args.option=='hidden' else None

            if bayesian==True:
                cov0 = 0                
                cov0_str = str(0)
            else:
                covs0 = load_obj('covs0',MP['aux'])
                cov0 = float(covs0[args.index])
                #cov0 = T.tensor(cov0,dtype=T.float32)
                cov0_str = str(cov0)[:6]
                print('loaded cov0 ',cov0)
                
            syn = wrap_run(dim,para,bayesian=bayesian,step_max=step_max,dt=dt,
                         pars_extra=pars_extra,cov0=cov0,hidden=hidden,
                         gen_hidden=False,prefix=str(args.index),
                         gen_plot_arrays=gen_plot_arrays,mdims=mdims,
                         init_dict=init_vars)
                    
            # output dict
            output = {'bayesian': bayesian, 
                      'cov0': float(cov0),
                      'MSE': float(syn.res['w MSE']), 
                      'inside': float(syn.res['q'])}
            
            print(output)
            
            # save entire (downsampled time series)            
            output.update({'MSE series': 
                np.array(syn.mout['w MSE'][::N_down_sample]), 'k':None})
            # non bayesians have a fixed k
            if bayesian==True:
                output.update({'k': np.array(syn.vars['k'])})
                print('E[k] = ',output['k'].mean())
            
            # write error to file
            save_obj(output,'cov0_'+cov0_str,MP['output'])
            stat = 'successfully '
 
        finally:
            
            print(stat + 'finished main: bayesian='+str(bayesian)+' idx=' + str(args.index))
            print('')
            # reduce status by one, the last job will make it zero
            #set_status(get_status()-1)                        
        # python ./main.py -s pre -N 3
                
            
    if args.stage=='post':
                
        copy_from = ('./BayesianSynapse/Synapse.py', './main.py')
        copy_to = (MP['output'] + 'Synapse.py', MP['output'] + 'main.py')
        for cf,ct in zip(copy_from,copy_to):                    
            shutil.copyfile(cf,ct)                     
        
        # runfile('/Users/jannes/Documents/Science/Real-Projects/peter_latham-paper/source-simulations/cluster_runner/src/main.py', wdir='/Users/jannes/Documents/Science/Real-Projects/peter_latham-paper/source-simulations/cluster_runner/src',
        # args="-s post -N 10")
        #ssh jegminat@euler.ethz.ch
        
        #set_status(1)        
        plt_MSE = True
        plt_MSE_series = False
        plt_traj = False
        
        # go to interesting path
        #os.chdir(os.getcwd() + '/../output/')
        #print(os.getcwd())
        
        # load all files

        if plt_MSE:

            ls2df = []
            for file in os.listdir(MP['output']):
                if (file.endswith(".pkl") and file.startswith("cov0_")):            
                    ls2df.append(load_obj(file[:-4],MP['output'])) # ,path=os.getcwd() + "/output/")                                            
            df = pd.DataFrame(ls2df)
            del(ls2df)

            # save
            coln = 'MSE' # short'
#            A = []
#            for i in df.index:
#                A.append(np.mean(df['MSE series'].iloc[i][50000:400000])/dim)
#            df[coln] = np.array(A)
#            df.to_pickle(MP['output'] + 'dataframe')

        #    df = pd.read_csv("./../rate_vs_mse.csv")
            df = df.sort_values('cov0')
            df = df.reset_index(drop=True)
            print(df)
            
            yplt = {}
            xplt = df.iloc[1:]['cov0'].values
            yplt['b'] = np.ones(len(df) - 1)*df.iloc[0][coln]
            yplt['c'] = df.iloc[1:][coln].values
                                    
#            plt.plot(np.log10(xplt),yplt['b'],':o',label='b')
#            plt.plot(np.log10(xplt),yplt['c'],':o',label='c')            
            plt.plot(xplt,yplt['b'],':o',label='b')
            plt.plot(xplt,yplt['c'],':o',label='c')
            
            plt.gca().legend()
            #df.plot(x='cov0',y=['b','c'],logx=True,)
            plt.ylabel(coln)
            #plt.xlabel('log10 cov0')
            plt.xlabel('eta')
            plt.ylim([0.6,1.0])

            plt.savefig(MP['output'] + coln + ".png")
            plt.show()            

    
            #cum_mean = lambda arr,L: np.cumsum(arr[::L])/np.arange(1,1+len(arr))
            cum_mean = lambda arr,L: arr
            #cum_mean = lambda arr,L: ndimage.gaussian_filter(arr,L)
            L = 1
        
            if plt_MSE_series:
                i_0,i_f,i_step = (0,-1,100)
            #def mse(i_0,i_f,i_step,L=1):
#                for ii in range(len(df)):
                for ii in [0,2,8]:
                    lab = df.iloc[ii]['cov0']
                    yplt = df.iloc[ii]['MSE series'][i_0:i_f:i_step]
                    xplt = np.arange(len(yplt))*i_step*dt*N_down_sample/3600
                    #plt.plot(xplt,cum_mean(yplt),label=str(np.round(lab,3)))
                    plt.plot(xplt,cum_mean(yplt,L),label=str(np.round(lab,3)))
                    
                plt.xlabel('Time [h]')
                plt.ylabel('MSE')
                plt.title('Cumulative MSE.')
                plt.gca().legend(ncol=1)
                plt.savefig('./output/MSE_over_time.pdf')
                plt.show()
        
        # load arrays
        if plt_traj and gen_plot_arrays:
                                    
            plt_dims = (0,)
            print('danger: should I be adding a mdims here?')
            syn = Synapse(name='getx', para=para, dim=dim, steps=step_max, dt=dt)
            
            # load all files that start with arr      
            #big_dic = {}
            for file, flag in it.product(os.listdir(os.getcwd()),('_b_','_c_')):
                if file.startswith("arr_"):                    
                    if flag in file:
                        name = file[file.index(flag):-4]                                                
                        for plt_dim in plt_dims:
                            traj = load_obj(file[:-4])
                            traj['hidden'] = traj.pop('w')[plt_dim]
                            traj['mean'] = traj.pop('m')[plt_dim]
                            traj['sig'] = np.sqrt(traj.pop('s2')[plt_dim])
                            plt_filter(syn.x_axis(),traj)
    
                            plt_name = name + ' dim=' + str(plt_dim)
                            plt.title(plt_name)
                            plt.savefig(MP['figs'] + plt_name + '.png')
                            plt.show()
                            
                            
                        #syn._hidden_readout(traj,keys_readout=('w','m','s2'))
                        # now I should be able to plot and save stuff
                        #kmax = steps - 1 
                        #syn.plt_maxSyn(kmax,syn,N_max=5,plot=True)
                        #syn.plt('wm',dim=0,xlim=None,log=False)
                
            #print(big_dic.keys())

#            covs0 = load_obj('covs0')
#            
#            for cov0 in covs0[args.index]:
#                cov0_str = str(cov0)[:6]
#                path_name ='./output/arr_cov0_' + cov0_str                
#
#                # create synapse with right properties
#                name = para + '_b' if bayesian else para + '_c_' + cov0_str
#                syn = Synapse(name=name, para=para, dim=dim, steps=step_max, dt=dt)                                
#                syn.set_pars(pars_user=pars_extra)
#
#                # load
#                traj = load_obj(path_name)                
#                syn._hidden_readout(traj,keys_readout=('w','m','s2'))
#                   
#                # now I should be able to plot and save stuff
#                #kmax = steps - 1 
#                #syn.plt_maxSyn(kmax,syn,N_max=5,plot=True)
#                syn.plt('wm',dim=0,xlim=None,log=False)                
            



#
#        
#        print('finished post')
#
#
##        if 0:
#        if args.index==None:
#
#            # remember hidden from Bayesian
#            hidden = wrap_run(dim,para,bayesian=True,step_max=step_max,dt=dt,
#                     pars_extra=pars_extra,cov0=None,hidden=None,
#                     gen_hidden=True,prefix='_genHidden',gen_plot_arrays=True)
#
#            ## create hidden
#            save_obj(hidden,"hidden")
#
##        if 1:
#        elif args.index==0:
#            ## Bayesian
#
#            # get error of bayesian synapse
#            syn = wrap_run(dim,para,bayesian=True,step_max=step_max,dt=dt,
#                     pars_extra=pars_extra,cov0=None,prefix='',gen_plot_arrays=True)
#
#            BS_err = float(syn.res['w MSE'])
#
#            # TODO
#            # write error to file
#            save_obj({0:('bayesian',BS_err)} ,"index"+str(args.index))
#
#
#
#        else: # args.index>0:
#
#            ## load value
#            covs0 = eps_lin(N=N,eps_min=0.001,symmetric=False,eps_anker=1.0)
#            cov0 = T.tensor(covs0[args.index-1],dtype=T.float32)
#
#            # run over learning rates
#            syn = wrap_run(dim,para,bayesian=False,step_max=step_max,dt=dt,
#                     pars_extra=pars_extra,cov0=cov0,prefix=str(cov0.item())[:6]+' '
#                     gen_plot_arrays=True)
#
#            CS_err = float(syn.res['w MSE'])
#
#            # TODO
#            save_obj({float(cov0):['classical',CS_err]} ,"index"+str(args.index))
#
#        print('finished process with idx ' + str(args.index))
#
##
##            out_dict.append({'cov0':cov0.item(),
##                             'Bayesian': BS_err,
##                             'Classical': syn.res['w MSE']
##
#

##### DYNAMICAL SYSTEM
#def ff(tau=50,N=1000,gamma=60,dt=0.002):
#    """ tau is oscillation time scale , gamma is damping """
#    
#    #tau = 50
#    #tmax = 1000
#    fac = 1/2.5/tau
#    A = np.array([[-1/gamma,-1./tau],[1./tau,-1/gamma]])
#    xdot = lambda x,s: A.dot(x) + s/dt
#    x = np.zeros([2,N+1])
#    s = np.zeros(2)
#    
#    for t in np.arange(N):
#        x[:,t+1] = x[:,t] + xdot(x[:,t],s*fac)*dt
#        
#        if t == round(N/5):
#            s[0] = 1
#        else:
#            s[0] = 0
#        
#    tspan = np.arange(N+1)*dt
#    plt.scatter(x[0,:],x[1,:],c=np.arange(N+1),cmap=matplotlib.cm.jet,s=10)
#    plt.xlabel('x1')
#    plt.xlabel('x2')
#    plt.show()
#    
#    plt.plot(tspan,x[0,:])
#    plt.plot(tspan,x[1,:])
#    plt.ylabel('x')
#    plt.show()
#    
#    return(x[1,:])
#    
