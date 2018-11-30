import numpy as np
import numpy
from nbodykit.lab import BigFileMesh, BigFileCatalog
from pmesh.pm import ParticleMesh

from nbodykit.cosmology import Cosmology, EHPower, Planck15
from background import *
from fpmfuncs import *

##
class Config(dict):
    def __init__(self):

        self['boxsize'] = 100
        self['shift'] = 0.0
        self['nc'] = 32
        self['ndim'] = 3
        self['seed'] = 100
        self['pm_nc_factor'] = 1
        self['resampler'] = 'cic'
        self['cosmology'] = Planck15
        self['powerspectrum'] = EHPower(Planck15, 0)
        self['unitary'] = False
        self['stages'] = numpy.linspace(0.1, 1.0, 5, endpoint=True)
        self['aout'] = [1.0]

        local = {} # these names will be usable in the config file
        local['EHPower'] = EHPower
        local['Cosmology'] = Cosmology
        local['Planck15'] = Planck15
        local['linspace'] = numpy.linspace
#         local['autostages'] = autostages

        import nbodykit.lab as nlab
        local['nlab'] = nlab

        self.finalize()

    def finalize(self):
        self['aout'] = numpy.array(self['aout'])

        self.pm = ParticleMesh(BoxSize=self['boxsize'], Nmesh= [self['nc']] * self['ndim'], resampler=self['resampler'], dtype='f4')
        mask = numpy.array([ a not in self['stages'] for a in self['aout']], dtype='?')
        missing_stages = self['aout'][mask]
        if len(missing_stages):
            raise ValueError('Some stages are requested for output but missing: %s' % str(missing_stages))

###############


###############

class StateVector(object):
    def __init__(self, solver, Q):
        self.solver = solver
        self.pm = solver.pm
        self.Q = Q
        self.csize = solver.pm.comm.allreduce(len(self.Q))
        self.dtype = self.Q.dtype
        self.cosmology = solver.cosmology

        self.H0 = 100. # in km/s / Mpc/h units
        # G * (mass of a particle)
        self.GM0 = self.H0 ** 2 / ( 4 * numpy.pi ) * 1.5 * self.cosmology.Om0 * self.pm.BoxSize.prod() / self.csize

        self.S = numpy.zeros_like(self.Q)
        self.P = numpy.zeros_like(self.Q)
        self.F = numpy.zeros_like(self.Q)
        self.RHO = numpy.zeros_like(self.Q[..., 0])
        self.a = dict(S=None, P=None, F=None)

    @property
    def X(self):
        return self.S + self.Q

    @property
    def V(self):
        a = self.a['P']
        return self.P * (self.H0 / a)

########
class Solver(object):
    def __init__(self, pm, cosmology, B=1, a_linear=1.0):
        """
            a_linear : float
                scaling factor at the time of the linear field.
                The growth function will be calibrated such that at a_linear D1 == 0.

        """
        if not isinstance(cosmology, Cosmology):
            raise TypeError("only nbodykit.cosmology object is supported")

        fpm = ParticleMesh(Nmesh=pm.Nmesh * B, BoxSize=pm.BoxSize, dtype=pm.dtype, comm=pm.comm, resampler=pm.resampler)
        self.pm = pm
        self.fpm = fpm
        self.cosmology = cosmology
        self.a_linear = a_linear

    # override nbodystep in subclasses
    @property
    def nbodystep(self):
        return FastPMStep(self)

    def whitenoise(self, seed, unitary=False):
        return self.pm.generate_whitenoise(seed, type='complex', unitary=unitary)

    def linear(self, whitenoise, Pk):
        return whitenoise.apply(lambda k, v:
                        Pk(sum(ki ** 2 for ki in k)**0.5) ** 0.5 * v / v.BoxSize.prod() ** 0.5)

    def lpt(self, linear, Q, a, order=2):
        """ This computes the 'force' from LPT as well. """
        assert order in (1, 2)

#         from .force.lpt import lpt1, lpt2source

        state = StateVector(self, Q)
        pt = PerturbationGrowth(self.cosmology, a=[a], a_normalize=self.a_linear)
        DX1 = pt.D1(a) * lpt1(linear, Q)

        V1 = a ** 2 * pt.f1(a) * pt.E(a) * DX1
        if order == 2:
            DX2 = pt.D2(a) * lpt1(lpt2source(linear), Q)
            V2 = a ** 2 * pt.f2(a) * pt.E(a) * DX2
            state.S[...] = DX1 + DX2
            state.P[...] = V1 + V2
            state.F[...] = a ** 2 * pt.E(a) * (pt.gf(a) / pt.D1(a) * DX1 + pt.gf2(a) / pt.D2(a) * DX2)
        else:
            state.S[...] = DX1
            state.P[...] = V1
            state.F[...] = a ** 2 * pt.E(a) * (pt.gf(a) / pt.D1(a) * DX1)

        state.a['S'] = a
        state.a['P'] = a

        return state

    def nbody(self, state, stepping, monitor=None):
        step = self.nbodystep
        for action, ai, ac, af in stepping:
            step.run(action, ai, ac, af, state, monitor)

        return state

    



class FastPMStep(object):
    def __init__(self, solver):
        self.cosmology = solver.cosmology
        self.pm = solver.fpm
        self.solver = solver

    def run(self, action, ai, ac, af, state, monitor):
        actions = dict(K=self.Kick, D=self.Drift, F=self.Force)

        event = actions[action](state, ai, ac, af)
        if monitor is not None:
            monitor(action, ai, ac, af, state, event)

    def Kick(self, state, ai, ac, af):
        assert ac == state.a['F']
        pt = PerturbationGrowth(self.cosmology, a=[ai, ac, af], a_normalize=self.solver.a_linear)
        fac = 1 / (ac ** 2 * pt.E(ac)) * (pt.Gf(af) - pt.Gf(ai)) / pt.gf(ac)
        state.P[...] = state.P[...] + fac * state.F[...]
        state.a['P'] = af

    def Drift(self, state, ai, ac, af):
        assert ac == state.a['P']
        pt = PerturbationGrowth(self.cosmology, a=[ai, ac, af], a_normalize=self.solver.a_linear)
        fac = 1 / (ac ** 3 * pt.E(ac)) * (pt.Gp(af) - pt.Gp(ai)) / pt.gp(ac)
        state.S[...] = state.S[...] + fac * state.P[...]
        state.a['S'] = af

    def prepare_force(self, state, smoothing):
        nbar = 1.0 * state.csize / self.pm.Nmesh.prod()
        X = state.X
        layout = self.pm.decompose(X, smoothing)
        X1 = layout.exchange(X)
        rho = self.pm.paint(X1)
        rho /= nbar # 1 + delta
        return layout, X1, rho

    def Force(self, state, ai, ac, af):

        assert ac == state.a['S']
        # use the default PM support
        layout, X1, rho = self.prepare_force(state, smoothing=None)
        state.RHO[...] = layout.gather(rho.readout(X1))
        delta_k = rho.r2c(out=Ellipsis)
        state.F[...] = layout.gather(longrange(X1, delta_k, split=0, factor=1.5 * self.cosmology.Om0))
        state.a['F'] = af
        return dict(delta_k=delta_k)



def fastpm(lptinit=False):
    config = Config()

    solver = Solver(config.pm, cosmology=config['cosmology'], B=config['pm_nc_factor'])
    whitenoise = solver.whitenoise(seed=config['seed'], unitary=config['unitary'])
    dlin = solver.linear(whitenoise, Pk=lambda k : config['powerspectrum'](k))

    Q = config.pm.generate_uniform_particle_grid(shift=config['shift'])

    state = solver.lpt(dlin, Q=Q, a=config['stages'][0], order=2)
    if lptinit: return state

    def monitor(action, ai, ac, af, state, event):
        if config.pm.comm.rank == 0:
            print('Step %s %06.4f - (%06.4f) -> %06.4f' %( action, ai, ac, af),
                  'S %(S)06.4f P %(P)06.4f F %(F)06.4f' % (state.a))


    solver.nbody(state, stepping=leapfrog(config['stages']), monitor=monitor)
    return state
