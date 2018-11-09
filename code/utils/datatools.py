import numpy as np


def read27(base, ind):
    '''
    ind = array of positions of shape (N, 3)
    return an array of 27 points closest to it
    '''

    ar = base.copy()
    ar = periodic(ar)
    if type(ind) is np.ndarray: ind = tuple(ind.T)
    elif type(ind) is tuple: pass


    toret = np.zeros([ind[0].size, 27])

    counter = 0;
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            for z in [-1, 0, 1]:
                rpos = [ind[0] + int(x), ind[1] + int(y), ind[2] +int(z)]
                toret[:, counter] = ar[tuple(rpos)]
                counter +=1
        
    return toret




def balancepts(base, blim, hfinegrid, hlim=0, ratiohigh=1, ratiolow=1, seed=100):
    '''
    Given a base and classgrid, select positions -
    - above hlim on on classgrid, numbered N
    - N*ratiohigh above lim on base but below hlim on hfinegrid
    - N*ratiolow below lim on base but below hlim  on hfinegrid
    '''
    np.random.seed(seed=seed)
    #halomass, halopos = self.halomass, self.halopos
    nc = base.shape[0]
    grid = np.array(np.where(base), dtype='i4').T
    
    #dind selects positions on 'base' grid above 'lim'
    #Idea is to select positions which are above some density threshold
    maskb = base >= blim
    masknob = ~maskb

    #Assign classes
    maskh = (hfinegrid > hlim)
    nh = maskh.sum()
    print(nh)
    maskbnoh = (maskb ^ (maskb & maskh))
    masknobnoh = (masknob ^ (masknob & maskh))

    #All the points that have a halo in the catalog halopos, irrespective of numd
    #ratiofine is taken care of in creating 'alldata' in datasets.py
    hindex = np.array(np.where(maskh)).T 
    bhindex = np.array(np.where(maskbnoh)).T
    blindex = np.array(np.where(masknobnoh)).T

    argsh = np.arange(maskbnoh.sum())
    argsl = np.arange(masknobnoh.sum())
    np.random.shuffle(argsh)
    np.random.shuffle(argsl)
    argsh = argsh[:int(ratiohigh*nh)]
    argsl = argsl[:int(ratiolow*nh)]
    bhindex = bhindex[argsh]
    blindex = blindex[argsl]
    dind = np.concatenate([hindex, bhindex, blindex], axis = 0)
    np.random.shuffle(dind)
    return dind


def subsamplehmass(hmass, mbins, ntot, maxsize):
    '''return index of subsampled halomass while maitaining the ratio of hmf
    in mass-bins where number of halos > maxsize such that total number of indices
    is roughly governed by ntot (can be multiple of ntot due to starred line below)
    '''
    lMin, lMax = np.log10(hmass[-1]), np.log10(hmass[0])
    dlM = (lMax - lMin)/mbins
    mmr = 10**(np.arange(lMin, lMax, dlM))[::-1]
    ranks = hmass.size - np.searchsorted(hmass[::-1], mmr)
    
    index, counts, mmc = [], [], []
    for i, r in enumerate(ranks[:-1]):
        truesize = ranks[i+1] - r
        size = int(ntot * truesize/ranks[-1])

        if truesize:
            if truesize < maxsize:
                index.append(np.random.choice(np.arange(r, ranks[i+1], 1), truesize))
                counts.append(truesize)
                mmc.append(mmr[i])
            else:
                #*+(int(maxsize) increases number from ntot by a lot depending on mbins
                size = int(ntot * truesize/ranks[-1]) + int(maxsize) 
                index.append(np.random.choice(np.arange(r, ranks[i+1], 1), size))
                counts.append(size)
                mmc.append(mmr[i])
    index = np.concatenate(index).astype(int)
    return index, counts, mmc



def subsize_zeros( y, index, ratio, seed=100):
    '''For a given 'y', count the number of non-zero
    entries (N); reshuffle the zero positions and add 
    (N*ratio) number of zero postitions to 'index' and
    return.
    The idea is to create training set with both 0s and
    non-0s position but the 0s need to be subsized since they
    are more in number.
    '''
    np.random.seed(seed=seed)
    mask = y > 0
    nzero = mask.sum()
    size = nzero*ratio
    args = np.arange((~mask).sum())
    np.random.shuffle(args)
    args = args[:size]
    yret = np.concatenate([y[mask], y[args]])
    newindex = np.concatenate([index[mask], index[args]], axis = 0)
    args = np.arange(yret.size)
    np.random.shuffle(args)
    return yret[args], newindex[args]



def norm(ft, meanar = None, stdar = None):
    '''
    If mean and std = None:
        Normalize the features.
        Returns: normlized array, mean-array, std-array
    If mean and std != None:
        Returns: normalized array
    '''
    if meanar is None:
        meanar = ft.mean(axis= 0)
        ft = ft - meanar
        stdar = ft.std(axis = 0)
        ft /= stdar
        return ft, np.array(meanar), np.array(stdar)
    else:
        ft = ft - meanar
        ft /= stdar
        return ft
    
def unnorm(ft, meanar, stdar ):
    '''
    If mean and std = None:
        Normalize the features.
        Returns: normlized array, mean-array, std-array
    If mean and std != None:
        Returns: normalized array
    '''
    ft = ft*stdar
    ft = ft + meanar
    return ft

def minmax(ft, minn = None, maxx = None):
    if minn is None:
        minn = ft.min(axis = 0)
        maxx = ft.max(axis = 0)
        ft -= minn
        ft /= (maxx - minn)
        
        return ft, minn, maxx

def unminmax(ftt, minn, maxx):
    ft = ftt.copy()
    ft *= (maxx - minn)
    ft += minn
    return ft

def nothing3(ft, x=None, y=None):
    if x is None:
        return ft.copy(), x, y
    else:
        return ft.copy()


def periodic(ar1):
    '''Copy the first side in each dim to the last so that
    the array becomes periodic to first side
    Return array of size (N+1)^3
    '''
    nc = ar1.shape[0]
    ar2 = np.zeros(ar1.shape + np.ones(3).astype(int))
    ar2[:nc, :nc, :nc] = ar1.copy()
    ar2[:nc, :nc, -1] = ar1[:, :, 0]
    ar2[:nc, -1, :nc] = ar1[:, 0, :]
    ar2[-1, :nc, :nc] = ar1[0, :, :]
    ar2[:nc, -1, -1] = ar1[:, 0, 0]
    ar2[-1, -1, :nc] = ar1[0, 0, :]
    ar2[-1, :nc, -1] = ar1[0, :, 0]
    ar2[-1, -1, -1] = ar1[0, 0, 0]
    return ar2

