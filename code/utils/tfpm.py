# Module storing a few tensorflow function to implement FastPM
import tensorflow as tf
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


def checkpaint():
    bs = 50
    nc = 16
    pm = ParticleMesh(BoxSize=bs, Nmesh = [nc, nc, nc], dtype='f4')

    nparticle = 100
    pos = bs*np.random.random(3*nparticle).reshape(-1, 3).astype(np.float32)
    wts = np.random.random(nparticle).astype(np.float32)
    
    pmmesh = pm.paint(pos, mass=wts)
    
    tfmesh = tf.zeros((nc, nc, nc), dtype=tf.float32)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tfmesh = sess.run(cic_paint(tfmesh, pos*nc/bs, weight=wts))
    
    print(abs(pmmesh[...] - tfmesh).sum())


def checkreadout():
    bs = 50
    nc = 16
    pm = ParticleMesh(BoxSize=bs, Nmesh = [nc, nc, nc], dtype='f4')

    nparticle = 100
    pos = bs*np.random.random(3*nparticle).reshape(-1, 3).astype(np.float32)
    base = 100*np.random.random(nc**3).reshape(nc, nc, nc).astype(np.float32)
    
    pmmesh = pm.create(mode='real', value=base)    
    pmread = pmmesh.readout(pos)
    
    tfmesh = tf.constant(base, dtype=tf.float32)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tfread = sess.run(cic_readout(tfmesh, pos*nc/bs))
    
    print(abs((pmread[...] - tfread)/pmread).sum())
#     print(abs(pmread[...] - tfread).sum()
