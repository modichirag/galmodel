# Module storing a few tensorflow function to implement FastPM
import tensorflow as tf

def cic_paint(mesh, part, weight, cube_size=None):
    """
        - mesh is a cube
        - part is a list of particles (:, 3)
        - weight is a list of weights (:)
    """

    # Create a variable to store the input mesh
    var = tf.Variable(0, dtype=tf.float32)
    var = tf.assign(var, mesh, validate_shape=False)

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

    if cube_size is not None:
        neighboor_coords = neighboor_coords % cube_size

    updated_mesh = tf.scatter_nd_add(var, tf.reshape(neighboor_coords, (-1, 3)),
                                     tf.reshape(kernel, (-1,)))
    return updated_mesh
