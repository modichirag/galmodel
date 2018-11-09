import numpy as np

def diracdelta(i, j):
    if i == j:
        return 1
    else:
        return 0

def shear(pm, base):                                                                                                                                          
    '''Takes in a PMesh object in real space. Returns am array of shear'''          
    s2 = pm.create(mode='real', zeros=True)                                                  
    k2 = sum(ki**2 for ki in pm.k)                                                                          
    k2[0,0,0] =  1                                                                  
    for i in range(3):
        for j in range(i, 3):                                                       
            basec = base.r2c()
            basec *= (pm.k[i]*pm.k[j] / k2 - diracdelta(i, j)/3.)              
            baser = basec.c2r()                                                                
            s2[...] += baser**2                                                        
            if i != j:                                                              
                s2[...] += baser**2                                                    
                                                                                    
    return s2  


def derivatives():
    pass

def velocity_dispersion():
    pass

def gauss(pm, R):
    def tf(k):
        k2 = 0
        for ki in k:
            k2 =  k2 + ki ** 2
        wts = np.exp(-0.5*k2*(R**2))
        return wts
    return tf


def fingauss(pm, R):
    kny = np.pi*pm.Nmesh[0]/pm.BoxSize[0]
    def tf(k):
        k2 = sum(((2*kny/np.pi)*np.sin(ki*np.pi/(2*kny)))**2  for ki in k)
        wts = np.exp(-0.5*k2* R**2)
        return wts
    return tf

def tophat(pm, R):
    def tf(k):
        k2 = 0
        for ki in k:
            k2 = k2 + ki**2        
        kr = R * k2**0.5
        kr[0,0] = 1
        wt = 3 * (np.sin(kr)/kr - np.cos(kr))/kr**2
        wt[0,0] = 1        
        return wt
    return tf



def tophatfunction(k, R):
    '''Takes in k, R scalar to return tophat window for the tuple'''
    kr = k*R
    wt = 3 * (np.sin(kr)/kr - np.cos(kr))/kr**2
    if wt is 0:
        wt = 1
    return wt

def gaussfunction(k, R):
    '''Takes in k, R scalar to return gauss window for the tuple'''
    kr = k*R
    wt = np.exp(-0.5*(kr**2))
    return wt


def fingaussfunction(k, kny, R):
    '''Takes in k, R and kny to do Gaussian smoothing corresponding to finite grid with kny'''
    kf = np.sin(k*np.pi/kny/2.)*kny*2/np.pi
    return np.exp(-(kf**2 * R**2) /2.)

def guassdiff(pm, R1, R2):
    pass


def smooth(pm, R, func):
    tfdict = {'fingauss':fingauss, 'gauss':gauss, 'tophat':tophat}
    tf = tfdict[func](pm, R)

    if pm.dtype == 'complex128' or pm.dtype == 'complex64':
        toret = pm.apply(lambda k, v: tf(k)*v).c2r()
    elif pm.dtype == 'float32' or pm.dtype == 'float64':
        toret = pm.r2c().apply(lambda k, v: tf(k)*v).c2r()
    return toret



def decic(pm, n=2):
    def tf(k):
        kny = [np.sinc(k[i]*pm.BoxSize[i]/(2*np.pi*pm.Nmesh[i])) for i in range(3)]
        wts = (kny[0]*kny[1]*kny[2])**(-1*n)
        return wts
        
    if pm.dtype == 'complex128' or pm.dtype == 'complex64':
        toret = pm.apply(lambda k, v: tf(k)*v).c2r()
    elif pm.dtype == 'float32' or pm.dtype == 'float64':
        toret = pm.r2c().apply(lambda k, v: tf(k)*v).c2r()
    return toret
