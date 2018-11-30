# Module storing a few tensorflow function to implement FastPM
import numpy as np
import numpy
import tensorflow as tf


def cic_paint(mesh, part, weight=None, cube_size=None):
    """
        - mesh is a cube
        - part is a list of particles (:, 3), positions in mesh units
        - weight is a list of weights (:)
        - cube_size is the size of the cube in mesh units
    """

    # Create a variable to store the input mesh
    var = tf.Variable(0, dtype=tf.float32)
    var = tf.assign(var, mesh, validate_shape=False)
    if weight is None: weight = np.ones(part.shape[0], dtype=part.dtype)
    if cube_size is None: cube_size = mesh.shape[0].value
    
    # Extract the indices of all the mesh points affected by each particles
    i000 = tf.cast(tf.floor(part), dtype=tf.int32)
    i100 = i000 + tf.constant([1, 0, 0])
    i010 = i000 + tf.constant([0, 1, 0])
    i001 = i000 + tf.constant([0, 0, 1])
    i110 = i000 + tf.constant([1, 1, 0])
    i101 = i000 + tf.constant([1, 0, 1])
    i011 = i000 + tf.constant([0, 1, 1])
    i111 = i000 + tf.constant([1, 1, 1])
    neighboor_coords = tf.stack([i000, i100, i010, i001,
                                 i110, i101, i011, i111], axis=1)
    kernel = 1. - tf.abs(tf.expand_dims(part, axis=1) - tf.cast(neighboor_coords, tf.float32))
    kernel = tf.reduce_prod(kernel, axis=-1, keepdims=False)
    kernel = tf.expand_dims(weight, axis=1) * kernel
        
#     if cube_size is not None:
    neighboor_coords = neighboor_coords % cube_size

    updated_mesh = tf.scatter_nd_add(var, tf.reshape(neighboor_coords, (-1, 3)),
                                     tf.reshape(kernel, (-1,)))
    return updated_mesh




def cic_readout(mesh, part, cube_size=None):
    """
        - mesh is a cube
        - part is a list of particles (:, 3), positions in mesh units
        - cube_size is the size of the cube in mesh units
    """

    if cube_size is None: cube_size = mesh.shape[0].value
    # Extract the indices of all the mesh points affected by each particles
    i000 = tf.cast(tf.floor(part), dtype=tf.int32)
    i100 = i000 + tf.constant([1, 0, 0])
    i010 = i000 + tf.constant([0, 1, 0])
    i001 = i000 + tf.constant([0, 0, 1])
    i110 = i000 + tf.constant([1, 1, 0])
    i101 = i000 + tf.constant([1, 0, 1])
    i011 = i000 + tf.constant([0, 1, 1])
    i111 = i000 + tf.constant([1, 1, 1])
    neighboor_coords = tf.stack([i000, i100, i010, i001,
                                 i110, i101, i011, i111], axis=1)
    kernel = 1. - tf.abs(tf.expand_dims(part, axis=1) - tf.cast(neighboor_coords, tf.float32))
    kernel = tf.reduce_prod(kernel, axis=-1, keepdims=False)
    
#     if cube_size is not None:
    neighboor_coords = neighboor_coords % cube_size

    meshvals = tf.gather_nd(mesh, neighboor_coords)
    weightedvals = tf.multiply(meshvals, kernel)
    value = tf.reduce_sum(weightedvals, axis=1)
    return value


def tflaplace(gdict):
    kvec = gdict['kvec']
    kk = sum(ki**2 for ki in kvec)
    mask = (kk == 0).nonzero()
    kk[mask] = 1
    wts = 1/kk
    imask = (~(kk==0)).astype(int)
    wts *= imask
    return wts
#     b = tf.multiply(v, 1/kk)
#     b = tf.multiply(b, imask)
#     return b 


def tfgradient(gdict, dir):
    kvec = gdict['kvec']
    bs, nc = gdict['bs'], gdict['nc']
    cellsize = bs/nc
    w = kvec[dir] * cellsize
    a = 1 / (6.0 * cellsize) * (8 * numpy.sin(w) - numpy.sin(2 * w))
    wts = a*1j
    return wts
    # a is already zero at the nyquist to ensure field is real
#     return tf.multiply(v , ac)
    


def tffknlongrange(gdict, r_split):
    kk = sum(ki ** 2 for ki in gdict['kvec'])
    if r_split != 0:
        return numpy.exp(-kk * r_split**2)
    else:
        return np.ones_like(kk)

def tflongrange(gdict, x, delta_k, split=0, factor=1):
    """ like long range, but x is a list of positions """
    # use the four point kernel to suppresse artificial growth of noise like terms

    ndim = 3
    lap = tflaplace(gdict)
    fknlrange = tffknlongrange(gdict, split)
    kweight = lap * fknlrange    
    pot_k = tf.multiply(delta_k, kweight)

    var = tf.Variable(0, dtype=tf.float32)
    f = []
    for d in range(ndim):
        force_dc = tf.multiply(pot_k, tfgradient(gdict, d))
        forced = tf.multiply(tf.spectral.irfft3d(force_dc), gdict['nc']**3)
        f.append(cic_readout(forced, x))
    
    f = tf.stack(f, axis=1)
    f = tf.multiply(f, factor)
    return f






    
#
#def tflaplace(v, gdict):
#    kvec = gdict['kvec']
#    kk = sum(ki**2 for ki in kvec)
#    mask = (kk == 0).nonzero()
#    kk[mask] = 1
#    imask = (~(kk==0)).astype(int)
#    b = tf.multiply(v, 1/kk)
#    b = tf.multiply(b, imask)
#    return b 
#
#
#def tfgradient(v, dir, gdict):
#    kvec = gdict['kvec']
#    bs, nc = gdict['bs'], gdict['nc']
#    cellsize = bs/nc
#    w = kvec[dir] * cellsize
#    a = 1 / (6.0 * cellsize) * (8 * numpy.sin(w) - numpy.sin(2 * w))
#    ac = a*1j
#    # a is already zero at the nyquist to ensure field is real
#    return tf.multiply(v , ac)
#    

##
##def cic_paint(mesh, part, weight, cube_size=None):
##    """
##        - mesh is a cube
##        - part is a list of particles (:, 3)
##        - weight is a list of weights (:)
##    """
##
##    # Create a variable to store the input mesh
##    var = tf.Variable(0, dtype=tf.float32)
##    var = tf.assign(var, mesh, validate_shape=False)
##
##    # Extract the indices of all the mesh points affected by each particles
##    i000 = tf.cast(tf.floor(part), dtype=tf.int32)
##    i100 = i000 + tf.constant([1, 0, 0])
##    i010 = i000 + tf.constant([0, 1, 0])
##    i001 = i000 + tf.constant([0, 0, 1])
##    i110 = i000 + tf.constant([1, 1, 0])
##    i101 = i000 + tf.constant([1, 0, 1])
##    i011 = i000 + tf.constant([0, 1, 1])
##    i111 = i000 + tf.constant([1, 1, 1])
##    neighboor_coords = tf.stack([i000, i100, i010, i001,
##                                 i110, i101, i011, i111], axis=1)
##
##    kernel = 1. - tf.abs(tf.expand_dims(part, axis=1) - tf.cast(neighboor_coords, tf.float32))
##    kernel = tf.reduce_prod(kernel, axis=-1, keepdims=False)
##    kernel = tf.expand_dims(weight, axis=1) * kernel
##
##    if cube_size is not None:
##        neighboor_coords = neighboor_coords % cube_size
##
##    updated_mesh = tf.scatter_nd_add(var, tf.reshape(neighboor_coords, (-1, 3)),
##                                     tf.reshape(kernel, (-1,)))
##    return updated_mesh
##


#
#def checkpaint():
#    from pmesh.pm import ParticleMesh
#    bs = 50
#    nc = 16
#    pm = ParticleMesh(BoxSize=bs, Nmesh = [nc, nc, nc], dtype='f4')
#
#    nparticle = 100
#    pos = bs*np.random.random(3*nparticle).reshape(-1, 3).astype(np.float32)
#    wts = np.random.random(nparticle).astype(np.float32)
#    
#    pmmesh = pm.paint(pos, mass=wts)
#    
#    tfmesh = tf.zeros((nc, nc, nc), dtype=tf.float32)
#    with tf.Session() as sess:
#        sess.run(tf.global_variables_initializer())
#        tfmesh = sess.run(cic_paint(tfmesh, pos*nc/bs, weight=wts))
#    
#    print(abs(pmmesh[...] - tfmesh).sum())
#
#
#def checkreadout():
#    from pmesh.pm import ParticleMesh
#    bs = 50
#    nc = 16
#    pm = ParticleMesh(BoxSize=bs, Nmesh = [nc, nc, nc], dtype='f4')
#
#    nparticle = 100
#    pos = bs*np.random.random(3*nparticle).reshape(-1, 3).astype(np.float32)
#    base = 100*np.random.random(nc**3).reshape(nc, nc, nc).astype(np.float32)
#    
#    pmmesh = pm.create(mode='real', value=base)    
#    pmread = pmmesh.readout(pos)
#    
#    tfmesh = tf.constant(base, dtype=tf.float32)
#    with tf.Session() as sess:
#        sess.run(tf.global_variables_initializer())
#        tfread = sess.run(cic_readout(tfmesh, pos*nc/bs))
#    
#    print(abs((pmread[...] - tfread)/pmread).sum())
##     print(abs(pmread[...] - tfread).sum()
#


