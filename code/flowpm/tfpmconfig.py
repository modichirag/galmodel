import numpy as np
import numpy
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
from  astropy.cosmology import Planck15 
from background import MatterDominated
from tfpmfuncs import fftk
##


class Config(dict):
    def __init__(self, bs=100., nc=32, seed=100, stages=None, cosmo=None, pkfile=None, pkinitfile=None):

        self['boxsize'] = np.float32(bs)
        self['shift'] = 0.0
        self['nc'] = int(nc)
        self['ndim'] = 3
        self['seed'] = seed
        self['pm_nc_factor'] = 1
        self['resampler'] = 'cic'
        self['cosmology'] = Planck15
        #self['powerspectrum'] = EHPower(Planck15, 0)
        self['unitary'] = False
        if stages is None: stages = numpy.linspace(0.1, 1.0, 5, endpoint=True)
        self['stages'] = stages
        self['aout'] = [1.0]
        self['perturbation'] = MatterDominated(cosmo=self['cosmology'], a=self['stages'])
        self['kvec'] = fftk(shape=(nc, nc, nc), boxsize=bs, symmetric=False, dtype=np.float32)
        
        if pkfile is None: pkfile = './Planck15_a1p00.txt'
        self['pkfile'] = pkfile
        if pkinitfile is None: pkinitfile = './Planck15_a0p01.txt'
        self['pkfile_ainit'] = pkinitfile

        self.finalize()

    def finalize(self):
        self['aout'] = numpy.array(self['aout'])
        self['klin'] = np.loadtxt(self['pkfile']).T[0]
        self['plin'] = np.loadtxt(self['pkfile']).T[1]
        self['ipklin'] = iuspline(self['klin'], self['plin'])
        
        #self.pm = ParticleMesh(BoxSize=self['boxsize'], Nmesh= [self['nc']] * self['ndim'], resampler=self['resampler'], dtype='f4')
        mask = numpy.array([ a not in self['stages'] for a in self['aout']], dtype='?')
        missing_stages = self['aout'][mask]
        if len(missing_stages):
            raise ValueError('Some stages are requested for output but missing: %s' % str(missing_stages))
