import numpy as np
import numpy
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf

sys.path.append('../flowpm/')
from background import *
import tfpm 
import tfpmfuncs as tfpf
from tfpmconfig import Config


#Generate DATA



def pm(config, verbose=True):
    g = tf.Graph()
    with g.as_default():
        linear = tfpm.linfield(config, name='linear')
        icstate = tfpm.lptinit(linear, config, name='icstate')
        fnstate = tfpm.nbody(icstate, config, verbose=verbose, name='fnstate')
        final = tf.zeros_like(linear)
        final = tfpf.cic_paint(final, fnstate[0], boxsize=config['boxsize'], name='final')
        tf.add_to_collection('pm', [linear, icstate, fnstate, final])
    return g


def gendata(config, ofolder=None):

    #Generate Data
    g = pm(config)

    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        linear = g.get_tensor_by_name('linear:0')
        final = g.get_tensor_by_name('final:0')
        fnstate = g.get_tensor_by_name('fnstate:0')
        icstate = g.get_tensor_by_name('icstate:0')
        truth, data = sess.run([linear, final])


    if ofolder is not None:
        np.save(ofolder+'final.f4', data)
        np.save(ofolder+'linear.f4', truth)

    return truth, data


if __name__ == "__main__":

    bs, nc = 400, 128
    seed = 100
    ofolder = './saved/L%04d_N%04d_S%04d/'%(bs, nc, seed)
    try: os.makedirs(ofolder)
    except: pass
    
    pkfile = '../code/flowpm/Planck15_a1p00.txt'
    config = Config(bs=bs, nc=nc, seed=seed, pkfile=pkfile)

    lin, data = gendata(config)
