import numpy as np
from scipy.stats.distributions import norm as npNorm
#from PyPDF2 import PdfFileMerger
import os
#import img2pdf
import pickle



def set_status(number):
    if not os.path.exists('./status.sh'):        
        f = open('status.sh','w+')
    else:
        f = open('status.sh','w')
    #a = input(text)
    f.write('echo ' + str(number))
    f.close()

def get_status():
    if not os.path.exists('./status.sh'):
        set_status(0)            
    text_file = open("./status.sh", "r")
    texts = text_file.readlines()        
    return(int(texts[0][5:])) # cut 'echo ' part in shell

def save_obj(obj, name, path='./'):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, path='./'):
#    print('load obj:',os.getcwd())
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def plt_idxs(plt_num,dim):
    """ create an array of length plt_num that's equidistant in dim """
    skip = int(max(np.floor(dim/plt_num),1))
    dims = (np.arange(0,dim)[::skip])
    dims = dims[:plt_num] if len(dims) > plt_num else dims
    return(dims)



def smooth(x,L=5):
    """ gaussian kernl with length 2L+1"""
    
    kernl = npNorm(0,1).pdf(np.linspace(-2,2,2*L+1))
    kernl /= kernl.sum()
    
    # init output
    out = np.zeros(x.shape)
    
    for i in range(len(x)):
        
        # get min and max indix of window 2L + 1
        idx_0 = max(i-L,0)
        idx_f = min(i+L,len(x)-1)
        
        # slice up input time series
        x_ = x[idx_0:idx_f]
        
        # check if index is near the end
        if len(x_) < len(kernl):
            
            # number of entries to be cut of
            d = len(kernl) - len(x_)
            
            # if true: we're next to beginning of the series
            if i < len(x)/2:
                kernl_ = kernl[d:]
                kernl_ /= kernl_.sum()
            
            # we're next to the ending of the series
            elif i > len(x)/2:
                kernl_ = (kernl[d:])[::-1]
                kernl_ /= kernl_.sum()
            
            out[i] = kernl_.dot(x_)
        # we're in the middle of the series
        else:            
            out[i] = kernl.dot(x_)

    return(out)


def numel(A):
    """ number of elements of tensor """
    return(np.prod(np.array([a for a in A.shape])))

def Tdot(A1,A2):
    """ A1 (N_1, ... N_m dim) dot A2 (dim) for torch tensors """
    if ((numel(A1) == 1) and (numel(A2))==1):
        return(A1*A2)
    else:
        return(A1.mm(A2.unsqueeze(1)).squeeze())


def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue
    
        del globals()[var]
        
        
        
    
def eps_lin(N,eps_min,symmetric=False,eps_anker=1.0):
    """ create line space -- denser close to 0,1 then in the middle     
        N = 101
        eps_min = 0.00000001
        eps = eps_lin(N,eps_min,symmetric=False,eps_anker=0.3)    
    """
    if symmetric:
        # left half space
        n = np.ceil(N/2)
        
        # decay constant ensure that n yields eps_min                        
        gamma = - np.log(2*eps_min)/n 
        
        # left half space
        eps_lin_left = 0.5*np.exp(- gamma*np.arange(n))

        # full space
        eps_lin = np.concatenate((eps_lin_left,1-eps_lin_left))
                        
        return(np.sort(eps_lin))
    else:
        # decay constant ensure that n yields eps_min                        
        gamma = - np.log(eps_min)/(N-1)
        
        # left half space
        eps_lin_left = eps_anker*np.exp(- gamma*np.arange(N))
                            
        return(np.sort(eps_lin_left))
        

def expspace(eps0,epsf,N=20):
    """ use like linspace but exponential    
    """    
    gamma = np.log(eps0)/(N-1)     

    return(np.sort(epsf*np.exp(gamma*np.arange(N))))



class Bool_Dict(dict):
    """ Dict with methods """

    def __init__(self,*arg,**kw):
        # class acts like a dict
        super(Bool_Dict, self).__init__(*arg, **kw)    
        
        
    def contains(self,key):
        """ test if a key exists and if it's true. Use like this:

            opts = Bool_Dict({option1: True})

            if opts.isTrue(option1):
                run()                        
        """
        if key in self:
            if self[key]:
                return(True)
        else:
            return(False)
    
    
        
#def images2pdf(mypath='./',merge=True,output_name='output'):
#    """ turn images png, jpg into pdf """
#    
#    # if don't merge, then remove .pdf suffix
#
#    # create a list of jpg images
#    images = []
#    for f in os.listdir(mypath):
#        if f.endswith('.jpg'):
#            images.append(mypath + f)
#        elif f.endswith('.png'):
#            # rename
#            image_old = mypath + f
#            image_new = mypath + f[:-4] + '.jpg'
#            os.rename(image_old, image_new)            
#            # add to list
#            images.append(image_new)
#                    
#    if merge == False:
#        for i in images:            
#            with open(i[:-4] + ".pdf", "wb") as f:
#                f.write(img2pdf.convert(i))
#                                                                                                                        
#    else: # merge == True
#        with open(mypath + 'converted.pdf', "wb") as f:
#            f.write(img2pdf.convert(images))
#
#        # concatenate with previous pdfs
#        merger = PdfFileMerger()
#        
#        # prepare merger
#        pdf_list = [mypath + f for f in os.listdir(mypath) if f.endswith('.pdf')]
#        for pdf in pdf_list:
#            merger.append(open(pdf, 'rb'))
#    
#        # final output
#        with open(mypath + output_name + '.pdf', 'wb') as fout:
#            merger.write(fout)
#        
#        
#        
#
#        
#def pdf_cat(pdf_list,output_name):
#    """ pdf list contains strings w or w/o pdf extension """
#
#    merger = PdfFileMerger()
#    
#    for pdf in pdf_list:
#        # add extensnion if nedded
#        if pdf[-4:] != '.pdf':
#            pdf = pdf + '.pdf'
#        merger.append(open(pdf, 'rb'))
#    
#    with open(output_name+'.pdf', 'wb') as fout:
#        merger.write(fout)
#
#            
#    
#def RK_wrap(dot_x,t0,x0,t_event,
#            jumpFun=lambda t,x: 1,new=True):
#    if not new: # in each iteration
#        RK = RK45(dot_x,t0,x0,t_event[0])
#    
#    for t_jump in t_event:
#        
#        if new:            
#            RK = RK45(dot_x,t0,x0,t_jump)
#        else:
#            RK.t = t0
#            RK.t_bound = t_jump
#            RK.x = x0
#            RK.status = 'running'
#                            
#        while RK.status is 'running':
#            RK.step()
#            
#        t0 = t_jump
#        x0 = RK.y + jumpFun(RK.t,RK.x) # jump
#
#    return(t0,x0)
#                  