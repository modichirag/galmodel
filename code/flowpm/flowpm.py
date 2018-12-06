import numpy as np
import numpy
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from time import time
import matplotlib.pyplot as plt

import tensorflow as tf

from background import *
import tfpm 
import tfpmfuncs as tfpf
from tfpmconfig import Config


pkfile = './Planck15_a1p00.txt'
config = Config(bs=100, nc=32, seed=100, pkfile=pkfile)
bs, nc = config['boxsize'], config['nc']
grid = bs/nc*np.indices((nc, nc, nc)).reshape(3, -1).T.astype(np.float32)
config['grid'] = grid

tf.reset_default_graph()

#Example 1
def pm(config):
     g = tf.Graph()
     bs, nc = config['boxsize'], config['nc']
     with g.as_default():
         linear = tf.get_variable('linear', shape=(nc, nc, nc), dtype=tf.float32, 
                                  initializer=tf.zeros_initializer())
         final = tf.get_variable('final', shape=(nc, nc, nc), dtype=tf.float32, 
                                 initializer=tf.zeros_initializer())
         state = tf.get_variable('state', shape=(3, nc**3, 3), dtype=tf.float32, 
                                 initializer=tf.zeros_initializer())

         getlinear = linear.assign(tfpm.linfield(config), name='getlinear')
         with tf.control_dependencies([getlinear]):
              getinit = state.assign(tfpm.lptinit(linear, grid, config),name='getinit')
         with tf.control_dependencies([getlinear, getinit]):
              donbody = state.assign(tfpm.nbody(state, config, verbose=True), name='donbody')
         with tf.control_dependencies([getlinear, getinit, donbody]):
              getfinal = final.assign(tfpf.cic_paint(final, state[0], boxsize=bs), name='getfinal')
     return g
pmgraph = pm(config)
with tf.Session(graph=pmgraph) as sess:
     sess.run(tf.global_variables_initializer())
     #sess.run('getlinear')
     #sess.run('getinit')
     #sess.run('donbody')
     sess.run('getfinal')
     linmesh, mesh, fstate = sess.run(['linear:0' , 'final:0', 'state:0'])

    
#Example 2
def pm(config):
    g = tf.Graph()
    bs, nc = config['boxsize'], config['nc']
    with g.as_default():
        
        linear = tfpm.linfield(config)
        state = tfpm.lptinit(linear, grid, config)
        state = tfpm.nbody(state, config, verbose=True)
        final = tf.zeros_like(linear)
        final = tfpf.cic_paint(final, state[0], boxsize=bs)
#         tf.add_to_collections('mesh', linear, final)
    return g, linear, final, state


pmgraph, linear, final, state = pm(config)
with tf.Session(graph=pmgraph) as sess:
    sess.run(tf.global_variables_initializer())
    linmesh2, mesh2, fstate2 = sess.run([linear, final, state])


print(fstate, fstate2)
print(linmesh- linmesh2)
print(mesh- mesh2)
