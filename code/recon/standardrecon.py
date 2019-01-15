import numpy as np
import sys
sys.path.append('../flowpm/')
sys.path.append('../utils/')

import tools
import tfpmfuncs, tfpm
import tfpmconfig 

import tensorflow as tf




def standardrecon(config, base, pos, bias, R=8):
    bs, nc = config['boxsize'], config['nc']
    basesm = tools.gauss(base, tools.fftk((nc, nc, nc), bs), R)

    g = tf.Graph()
    with g.as_default():
        mesh = tf.constant(basesm.astype(np.float32))
        meshk = tfpmfuncs.r2c3d(mesh, norm=nc**3)

        DX = tfpm.lpt1(meshk, pos, config)
        DX = tf.multiply(DX, -1/bias)
        pos = tf.add(pos, DX)
        displaced = tf.zeros_like(mesh)
        displaced = tfpm.cic_paint(displaced, pos, boxsize=bs, name='displaced')

        DXrandom = tfpm.lpt1(meshk, config['grid'], config)
        DXrandom = tf.multiply(DXrandom, -1/bias)
        posrandom = tf.add(config['grid'], DXrandom)
        random = tf.zeros_like(mesh)
        random = tfpm.cic_paint(random, posrandom, boxsize=bs, name='random')
        tf.add_to_collection('recon', [displaced, random])
    return g

    

if __name__=="__main__":

    bs, nc, step = 400, 128, 5
    ncf, stepf = 512, 40
    seed = 100
    config = tfpmconfig.Config(bs=bs, nc=nc, seed=seed)
    #
    path = '../../data/z00/'
    ftype = 'L%04d_N%04d_S%04d_%02dstep/'
    numd = 1e-3
    num = int(numd*bs**3)
    #
    mesh = {}
    partp = tools.readbigfile(path + ftype%(bs, nc, seed, step) + 'dynamic/1/Position/')
    mesh['cic'] = tools.paintcic(partp, bs, nc)
    mesh['s'] = tools.readbigfile(path + ftype%(bs, nc, seed, step) + 'mesh/s/')

    hmesh = {}
    hposall = tools.readbigfile(path + ftype%(bs, ncf, seed, stepf) + 'FOF/PeakPosition/')[1:]
    hposd = hposall[:num].copy()
    hmesh['pcic'] = tools.paintcic(hposd, bs, nc)
    hmesh['pnn'] = tools.paintnn(hposd, bs, nc)


    ##
    base = hmesh['pcic']
    base = (base - base.mean())/base.mean()
    pfin = tools.power(mesh['cic'], boxsize=bs)[1]
    ph = tools.power(1+base, boxsize=bs)[1]
    bias = ((ph[1:5]/pfin[1:5])**0.5).mean()
    print(bias)

    g = standardrecon(config, base, hposd, bias, R=8)

    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        tfdisplaced = g.get_tensor_by_name('displaced:0')
        tfrandom = g.get_tensor_by_name('random:0')

        displaced, random = sess.run([tfdisplaced, tfrandom])

    print(displaced)
    print(random)
