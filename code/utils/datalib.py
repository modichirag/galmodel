import numpy as np
from datatools import *

def balanced27gridpt(ftlist, clsgrid, dind, clim = 0, classify = True, ppx='std', ppy='std', testdata=True):
    '''For classification
    Y = data is the value at the point on halogrid ('clsgrid')
    Training set = 27 neighboring points on ftlist at halo position supplemented with *ratio multiple of points
    above lim on base and *ratiolow multiple of points below lim
    Test = value at the all the grid points    
    '''
 
    if ppx is None:
        ppx = 'none'
    if ppy is None:
        ppy = 'none'

    dind = tuple(dind.T)

    cls = clsgrid[dind]
    if classify:
        print('Classifying')
        cls[cls>clim] = 1
        cls[cls<=clim] = 0

    ndim = len(ftlist)
    print('Number of features is =',ndim)

    #Read features for training and normalize
    ft = np.zeros([dind[0].size, ndim*27])
    for boo in range(ndim):
        ft[:, 27*boo: 27*boo + 27] = read27(ftlist[boo], ind=dind)
    
    #Normalize
    normdict = {'std':norm, 'minmax':minmax, 'none':nothing3}
    ftn, mx, sx = normdict[ppx](ft)
    cls, my, sy =  normdict[ppy](cls)
    if my is None:
        print(' y is not Normalized')

    # Read features over the whole grid to generate data to test performance
    if testdata:
        grid = np.array(np.where(clsgrid*0+1), dtype='i4').T
        ftt = np.zeros([clsgrid.shape[0]**3, 27*ndim])
        for boo in range(ndim):
            ftt[:, 27*boo: 27*boo + 27] = read27(ftlist[boo], ind=grid)
        fttn = normdict[ppx](ftt, mx, sx)

    else:
        ftt, fttn = None, None
    return [ftn, mx, sx], [cls, my, sy], [fttn, dind, ftt]



