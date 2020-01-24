# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:04:11 2019

@author: giess
"""
import numpy as np
import random

#gaussian = lambda x, A, mu, sig : (A  * np.exp(-(x - mu)**2 / (2 * sig**2)))

def gaussian(x, A, mu, sig):
    return (A  * np.exp(-(x - mu)**2 / (2 * sig**2)))

def response_model(x, mu , sig = 0.1):
    """
    Model across the response values
    """
    return np.exp( (-(x - mu) * (x-mu)) / (2 * (sig * sig)))


def normrnd(no_noise_responses):
    """
    Applies some noise to the responses from the true model
    """
    noise = lambda mu : np.random.normal(loc=mu, scale=0.4, size=None)
    for n in range(0, len(no_noise_responses)):
        no_noise_responses[n] = noise(no_noise_responses[n])
    return no_noise_responses

def exprnd(mu, outsize):
    
#    r = -mu * np.log(np.random.uniform(mu, outsize))
    r = np.random.exponential(mu, outsize)
    return r

def pdf(x, mu, sig = 0.1):
    """Probability density function for the response model"""
    return (1 / (2 * np.pi * (sig * sig)) * np.exp( (-(x - mu) * (x-mu)) / (2 * (sig * sig))) )

def log_pdf(theta, x, y, LB, UB):
    
#    if any(theta) > UB or any(theta) < LB:
    for i in range(len(theta)):
        if theta[i] < LB[i] or theta[i] > UB[i]:
            logL = -np.inf
#            print("out of bounds", theta)
            return logL
        
    A = theta[0]
    mu = theta[1]
    sig = theta[2]
    lam = A*np.exp(-(x-mu)**2 / (2*sig**2))
    
    """
    lam ends up with 0.00e+00 values from which it doesn't recover,
    leading to an exception in the while loop
    """
#    print("lam", lam)
    
    logL = np.multiply(y, np.log(lam)) - np.sum(lam)
    return np.sum(logL)

def inside(th, theta, x, y, LB, UB):
    
    logL = log_pdf(theta, x, y, LB, UB)
#    print(logL)
    return logL > th

def pdf_to_logpdf():
    pass

def sliceSampler(initial, nsamples, **kwargs):
    """
    initial:
        np.array - initial values 
    nsamples:
        int - number of samples to return
    pdf: 
        store as a list the parameters to pass to the pdf function
    logpdf:
        store as a list the parameters to pass to the logpdf function
    burnin:
        # of burnin samples
    width:
        width of the slice
    thin:
        amount of samples to skip between saved samples
    """
    
    param = {'pdf': [], 'logpdf': [], 'burnin':0, 'width':10, 'thin':1}
    param.update(kwargs)
    
#       if pdf provided - convert pdf to log pdf
#        if len(param['logpdf']) < 1:
        
    
    # assign parameter variables
    thin = param['thin']
    burnin = param['burnin']
    width = param['width']
    
    # assign the logpdf variables
    logpdf_xtr = param['logpdf'][0]
    logpdf_ytr = param['logpdf'][1]
    logpdf_LB = param['logpdf'][2]
    logpdf_UB = param['logpdf'][3]
    
    """CHECKS"""
    if len(initial) < 1:
        raise BaseException( "inital values missing" )
    
    if len(param['pdf']) < 1 and len(param['logpdf']) < 1:
        raise BaseException( "no pdf or logpdf function given" )
        
    
    maxiter = 200
    dim = initial.shape[0]
    outclass = initial.dtype
    rnd = np.zeros((nsamples, dim), dtype = outclass)  # inital output samples array
    
    neval = nsamples
    
    e = exprnd(1, [nsamples*thin+burnin])
#    print("e", e)

    RW = np.random.uniform(0.0, 1.0, (nsamples*thin+burnin,dim))  # factors of randomizing the width
    RD = np.random.uniform(0.0, 1.0, (nsamples*thin+burnin,dim))  # uniformly draw the point within the slice
    x0 = initial
    
    for i in range(-burnin, nsamples*thin):

        z = log_pdf(x0, logpdf_xtr, logpdf_ytr, logpdf_LB, logpdf_UB)
        z = z - e[i+burnin]
        
#        print("z", z)
#        sys.exit(0)
    
    
        r = width * RW[i+burnin,:]
        xl = x0 - r
        xr = xl + width
        iter = 0

        """We can skip the step-out procedure as long as we have a unimodal model
        with a width that is not too small."""
        
        
        xp = RD[i+burnin,:] * (xr-xl) + xl
        
        # If xp outside of the slice, shrink the interval
        while not inside(z, xp, logpdf_xtr, logpdf_ytr, logpdf_LB, logpdf_UB) and iter < maxiter:
                rshrink = xp>x0
                xr[rshrink] = xp[rshrink]
                lshrink = np.invert(rshrink)
                xl[lshrink] = xp[lshrink]
                xp = np.multiply(np.random.uniform(0, 1, dim), (xr-xl)) + xl
                iter += 1
        
        if iter >= maxiter:
            print("\n===============\ncurrent xp", xp)
            print("logL", log_pdf(x0, logpdf_xtr, logpdf_ytr, logpdf_LB, logpdf_UB))
            raise BaseException( "error in skrinking slice" )
        
        x0 = xp
        if i > 0 and i % thin == 0:
            rnd[i//thin, :] = x0
        neval += iter
        
        
    neval = neval / (nsamples*thin+burnin)  # mean number of evaluations

    return rnd, neval



A = [1, 12, 1]
mu = [-10, 10, 1]
sig = [.1,10, 1]
stimuli_range = np.arange(-12, 13, 1)
response_range = np.round(np.arange(-0.5, 1.6, 0.1), 1) # range of response values

nx = len(stimuli_range)

A_gen = 9
mu_gen = 5
sig_gen = 5.1


gau_true = gaussian(stimuli_range, A_gen, mu_gen, sig_gen)

theta_true = [A_gen, mu_gen, sig_gen]

ninit = 3
xtr = np.asarray([random.randint(stimuli_range[0], stimuli_range[-1]) for p in range(0, ninit)])
ytr = normrnd(gaussian(xtr, A_gen, mu_gen, sig_gen))

Arnge = np.arange(A[0], A[1]+A[2], A[2])
murnge = np.arange(mu[0], mu[1] + mu[2], mu[2])
sigrnge = np.arange(sig[0], sig[1] + sig[2], sig[2])


LB = np.array([Arnge[0], murnge[0], sigrnge[0]])
UB = np.array([Arnge[-1], murnge[-1], sigrnge[-1]])
print("LB", LB, "\nUB", UB)
prs0 = (LB + UB) / 2 # initial parameters
theta = prs0
print("prs0", prs0)

thetasamps, _ = sliceSampler(theta, 50, logpdf = [xtr, ytr, LB, UB], burnin = 100)


#print(thetasamps)

thetasamps = np.mean(thetasamps, axis = 0)
print("mean initial thetasamps", thetasamps, "\n")

trialNumsToPlot = [1,1,1,1]
nslice = 100
nburnin = 100

for i in range(1, np.max(trialNumsToPlot)+1):
    
    if i == 1:
        thetasamps, _ = sliceSampler(thetasamps, nslice, logpdf = [xtr, ytr, LB, UB], burnin = nburnin)
    else:
        thetasamps, _ = sliceSampler(thetasamps[-1], nslice, logpdf = [xtr, ytr, LB, UB], burnin = nburnin)
    thetamu = np.mean(thetasamps[0], axis = 0)
    
    print("theta samps:", thetasamps.shape)
    
    # compute posterior mean over TC
    TCmu = np.zeros((nx, 1))
    for j in range(0, nslice):
        tc = gaussian(stimuli_range, thetasamps[j][0], thetasamps[j][1], thetasamps[j][2])
        TCmu = np.add(TCmu, (tc[0]/nslice))
    
#    print(TCmu)


    rmax = np.ceil(np.max(TCmu) + np.sqrt(np.max(TCmu)) * 3)
    rr = np.arange(0, rmax+1, 1)
    prr = np.zeros((int(rmax) + 1, nx, nslice))
    print("prr", prr.shape)
    
    for j in range(0, nslice):
        tc = gaussian(stimuli_range, thetasamps[j][0], thetasamps[j][1], thetasamps[j][2])
        print("tc shape", tc.shape)
        temp = np.exp(np.add(np.add(rr*np.log(tc), -tc), np.log(rr+1)))
        print("temp shape", temp.shape)
        prr[:,:,j] = temp

    


#