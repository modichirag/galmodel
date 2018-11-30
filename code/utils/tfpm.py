# Module storing a few tensorflow function to implement FastPM
import numpy as np
import numpy
import tensorflow as tf
from tfpmfuncs import *
from background import *

def tflpt1(dlin_k, pos, gdict):
    """ Run first order LPT on linear density field, returns displacements of particles
        reading out at q. The result has the same dtype as q.
    """
    bs, nc = gdict['bs'], gdict['nc']    
    #ones = tf.ones_like(dlin_k)
    lap = tflaplace(gdict)
    
    displacement = tf.zeros_like(pos)
    displacement = []
    for d in range(3):
        #kweight = tf.multiply(tfgradient(gdict, d), lap)
        kweight = tfgradient(gdict, d) * lap
        dispc = tf.multiply(kweight, dlin_k)
        disp = tf.multiply(tf.spectral.irfft3d(dispc), nc**3)
        displacement.append(cic_readout(disp, pos))

    return tf.stack(displacement, axis=1)



def tflpt2source(dlin_k, gdict):
    """ Generate the second order LPT source term.  """

    bs, nc = gdict['bs'], gdict['nc']
    source = tf.zeros((nc, nc, nc))
    
    D1 = [1, 2, 0]
    D2 = [2, 0, 1]

    phi_ii = []

    # diagnoal terms
    lap = tflaplace(gdict)
    
    for d in range(3):
        grad = tfgradient(gdict, d) 
        kweight = grad * grad * lap
        phic = tf.multiply(kweight, dlin_k)
        phi_ii.append(tf.multiply(tf.spectral.irfft3d(phic), nc**3))

    for d in range(3):
        source = tf.add(source, tf.multiply(phi_ii[D1[d]], phi_ii[D2[d]]))
    
#     return source
    # free memory
    phi_ii = []

    # off-diag terms
    for d in range(3):
        gradi = tfgradient(gdict, D1[d])
        gradj = tfgradient(gdict, D2[d])
        kweight = gradi * gradj * lap
        phic = tf.multiply(kweight, dlin_k)
        phi = tf.multiply(tf.spectral.irfft3d(phic), nc**3)
        source = tf.subtract(source, tf.multiply(phi, phi))

    source = tf.multiply(source, 3.0/7.)
    return tf.multiply(tf.spectral.rfft3d(source), 1/nc**3)


def tflptz0(lineark, Q, gdict, a=1, order=2):
    
    bs, nc = gdict['bs'], gdict['nc']
    pos = Q*nc/bs
    
    DX1 = 1 * tflpt1(lineark, pos, gdict)
    if order == 2: DX2 = 1 * tflpt1(tflpt2source(lineark, gdict), pos, gdict)
    else: DX2 = 0
    return tf.add(DX1 , DX2)
    
    
###############################################################################################


########


def tflptinit(lineark, Q, a, gdict, config, order=2):
        """ This computes the 'force' from LPT as well. """
        assert order in (1, 2)

        bs, nc = gdict['bs'], gdict['nc']
        pos = Q * nc/bs
        dtype = np.float32

        pt = PerturbationGrowth(config['cosmology'], a=[a], a_normalize=1.0)
        DX = tf.multiply(dtype(pt.D1(a)) , tflpt1(lineark, pos, gdict)) 
        P = tf.multiply(dtype(a ** 2 * pt.f1(a) * pt.E(a)) , DX)
        F = tf.multiply(dtype(a ** 2 * pt.E(a) * pt.gf(a) / pt.D1(a)) , DX)
        if order == 2:
            DX2 = tf.multiply(dtype(pt.D2(a)) , tflpt1(tflpt2source(lineark, gdict), pos, gdict))
            P2 = tf.multiply(dtype(a ** 2 * pt.f2(a) * pt.E(a)) , DX2)
            F2 = tf.multiply(dtype(a ** 2 * pt.E(a) * pt.gf2(a) / pt.D2(a)) , DX2)
            DX = tf.add(DX, DX2)
            P = tf.add(P, P2)
            F = tf.add(F, F2)

        return tf.stack((DX, P, F), axis=0)



def tfKick(state, ai, ac, af, gdict, config, dtype=np.float32):
    var = tf.Variable(0, dtype=tf.float32)
    var = tf.assign(var, state, validate_shape=False)
    pt = PerturbationGrowth(config['cosmology'], a=[ai, ac, af], a_normalize=1.0)
    fac = 1 / (ac ** 2 * pt.E(ac)) * (pt.Gf(af) - pt.Gf(ai)) / pt.gf(ac)
    update = tf.multiply(dtype(fac), state[2])
    var = tf.scatter_add(var, 0, update)
    return var

def tfDrift(state, ai, ac, af, gdict, config, dtype=np.float32):
    var = tf.Variable(0, dtype=tf.float32)
    var = tf.assign(var, state, validate_shape=False)
    pt = PerturbationGrowth(config['cosmology'], a=[ai, ac, af], a_normalize=1.0)
    fac = 1 / (ac ** 3 * pt.E(ac)) * (pt.Gp(af) - pt.Gp(ai)) / pt.gp(ac)
    update = tf.multiply(dtype(fac), state[1])
    var = tf.scatter_add(var, 0, update)
    return var

    

def tfForce(state, ai, ac, af, gdict, config, dtype=np.float32):
    var = tf.Variable(0, dtype=tf.float32)
    var = tf.assign(var, state, validate_shape=False)

    bs, nc = gdict['bs'], gdict['nc']
    rho = tf.Variable(0, dtype=tf.float32)
    rho = tf.assign(var, tf.zeros((nc, nc, nc), dtype=tf.float32) , validate_shape=False)
    wts = tf.ones(nc**3)
    nbar = nc**3/bs**3

    rho = cic_paint(rho, tf.multiply(state[0], nc/bs), wts)
    rho = tf.multiply(rho, 1/nbar)

    delta_k = tf.multiply(tf.spectral.rfft3d(rho), 1/nc**3)
    fac = dtype(1.5 * config['cosmology'].Om0)
    update = tflongrange(gdict, state[0], delta_k, split=0, factor=fac)

    var = tf.scatter_add(var, 0, update)
    return var

    #print(var, update)
    #update1 = tf.zeros((nc**3, 3))
    #update = tf.add(update, update1)
    #return tf.add(update, update1) 
    #print(var, update)
    #print(var, update1)
    #var = tf.scatter_add(var, 0, update)
    #return var


#
#    def nbody(self, state, stepping, monitor=None):
#        step = self.nbodystep
#        for action, ai, ac, af in stepping:
#            step.run(action, ai, ac, af, state, monitor)
#
#        return state
#
#
#    def run(self, action, ai, ac, af, state, monitor):
#        actions = dict(K=self.Kick, D=self.Drift, F=self.Force)
#
#        event = actions[action](state, ai, ac, af)
#        if monitor is not None:
#            monitor(action, ai, ac, af, state, event)
#
#    def prepare_force(self, state, smoothing):
#        nbar = 1.0 * state.csize / self.pm.Nmesh.prod()
#        X = state.X
#        layout = self.pm.decompose(X, smoothing)
#        X1 = layout.exchange(X)
#        rho = self.pm.paint(X1)
#        rho /= nbar # 1 + delta
#        return layout, X1, rho
#
#    def Force(self, state, ai, ac, af):
#
#        assert ac == state.a['S']
#        # use the default PM support
#        layout, X1, rho = self.prepare_force(state, smoothing=None)
#        state.RHO[...] = layout.gather(rho.readout(X1))
#        delta_k = rho.r2c(out=Ellipsis)
#        state.F[...] = layout.gather(longrange(X1, delta_k, split=0, factor=1.5 * self.cosmology.Om0))
#        state.a['F'] = af
#        return dict(delta_k=delta_k)
#
#
#
