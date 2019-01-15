import numpy as np



def randomvoxels(ftlist, targetlist, num_cubes, max_offset, cube_size=32, cube_sizeft=32, seed=100, rprob=0.0):
    '''Generate 'num_cubes' training voxels of 'cube_size' for target and 'cube_sizeft' for features
    from the meshes in 'ftlist' for features and targets in 'target'.
    Rotate voxels with probability 'rprob'
    '''
    np.random.seed(seed)
    rand = np.random.rand

    nchannels = len(ftlist)
    if type(targetlist) == list:
        pass
    else:
        targetlist = [targetlist]
    ntarget = len(targetlist)
    print('Length of targets = ', ntarget)

    cube_features = []
    cube_target = []

    nrotated = 0 
    for it in range(num_cubes):
        #print(it)
        # Extract random cubes from the sim
        offset_x = int(round(rand()*max_offset))
        offset_y = int(round(rand()*max_offset))
        offset_z = int(round(rand()*max_offset))
        x1, x2, x2p = offset_x, offset_x+cube_size, offset_x+cube_sizeft
        y1, y2, y2p = offset_y, offset_y+cube_size, offset_y+cube_sizeft
        z1, z2, z2p = offset_z, offset_z+cube_size, offset_z+cube_sizeft

        features = []
        for i in range(nchannels): features.append(ftlist[i][x1:x2p, y1:y2p, z1:z2p])
        cube_features.append(np.stack(features, axis=-1))
        #
        targets = []
        for i in range(ntarget): targets.append(targetlist[i][x1:x2, y1:y2, z1:z2])
        cube_target.append(np.stack(targets, axis=-1))
        
        rotate = False
        rotation = []
        while (np.random.random() < rprob) & (len(rotation) <= 3):
            rotate = True
            nrot, ax0, ax1 = np.random.randint(0, 3), *np.random.permutation((0, 1, 2))[:2]
            rotation.append([nrot, ax0, ax1])

        def do_rotation(ar):
            for j in rotation:
                ar = np.rot90(ar, k=j[0], axes=(j[1], j[2]))
            return ar
        
        if rotate:
            nrotated +=1
            #Do for features
            features = []
            for i in range(nchannels):
                features.append(do_rotation(ftlist[i][x1:x2p, y1:y2p, z1:z2p]))#.copy()
            cube_features.append(np.stack(features, axis=-1))

            #Do for targets
            targets = []
            for i in range(ntarget):
                targets.append(do_rotation(targetlist[i][x1:x2, y1:y2, z1:z2]))#.copy()
            cube_target.append(np.stack(targets, axis=-1))
            
    print('Supplemented by rotation : ', nrotated)
    return cube_features, cube_target

    
def splitvoxels(ftlist, cube_size, shift=None, ncube=None):
    '''Split the meshes in ftlist in voxels of 'cube_size' in a regular fashion by
    shifting with 'shift' over the range of (0, ncp) on the mesh
    '''
    if type(ftlist) is not list: ftlist = [ftlist]
    ncp = ftlist[0].shape[0]
    if shift is None: shift = cube_size
    if ncube is None: ncube = int(ncp/shift)
    
    inp = []
    for i in range(ncube):
        for j in range(ncube):
            for k in range(ncube):
                x1, y1, z1 = i*shift, j*shift, k*shift
                x2, y2, z2 = x1+cube_size, y1+cube_size, z1+cube_size
                fts = np.stack([ar[x1:x2, y1:y2, z1:z2] for ar in ftlist], axis=-1)
                inp.append(fts)

    inp = np.stack(inp, axis=0)
    return inp


#    predict = np.zeros_like(mesh)
#    counter = 0 
#    for i in range(nshift):
#        for j in range(nshift):
#            for k in range(nshift):
#                x1, y1, z1 = i*shift, j*shift, k*shift
#                x2, y2, z2 = x1+shift, y1+shift, z1+shift
#                predict[x1:x2,y1:y2, z1:z2] = recp[counter, shift//2:-shift//2, shift//2:-shift//2, shift//2:-shift//2, 0]
#                counter +=1
#



def readperiodic(ar, coords):
    '''
    '''
    def roll(ar, l1, l2, l0, axis):
        if l1<0 and l2>l0: 
            print('Inconsistency along axis %d'%axis)
            return None
        if l1<0: 
            ar=np.roll(ar, -l1, axis=axis)
            l1, l2 = 0, l2-l1
        elif l2>l0: 
            ar=np.roll(ar, l0-l2, axis=axis)
            l1, l2 = l1+l0-l2, l0
        return ar, l1, l2,

    if len(ar.shape) != len(coords): 
        print('dimensions inconsistent')
        return None
    ndim = len(coords)
    newcoords = []
    for i in range(ndim):
        ar, l1, l2 = roll(ar, coords[i][0], coords[i][1], ar.shape[i], i)
        newcoords.append([l1, l2])
    sl = []
    for i in range(ndim):
        sl.append(slice(*newcoords[i]))
    return ar[tuple(sl)]



def cubify(arr, newshape):
    oldshape = np.array(arr.shape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.column_stack([repeats, newshape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    # newshape must divide oldshape evenly or else ValueError will be raised
    return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)

def uncubify(arr, oldshape):
    N, newshape = arr.shape[0], arr.shape[1:]
    oldshape = np.array(oldshape)    
    print(newshape, oldshape)
    repeats = (oldshape // newshape).astype(int)
    tmpshape = np.concatenate([repeats, newshape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
    return arr.reshape(tmpshape).transpose(order).reshape(oldshape)



###num_cubes= 500
###cube_sizes = np.array([8, 16, 32, 64]).astype(int)
###nsizes = len(cube_sizes)
###pad = int(2)
###cube_sizesft = (cube_sizes + 2*pad).astype(int)
###max_offset = ncp - cube_sizes
###ftname = ['cic']
###tgname = ['pnn']
###
###def gentrainingdata(path, bs, nc, ftname, tgname, seeds, cube_sizes, num_cubes, pad, gal=True):
###
###    meshes = {}
###    cube_features, cube_target = [[] for i in range(len(cube_sizes))], [[] for i in range(len(cube_sizes))]
###
###    for seed in seeds:
###        mesh = {}
###        partp = tools.readbigfile(path + ftype%(bs, nc, seed, step) + 'dynamic/1/Position/')
###        mesh['cic'] = tools.paintcic(partp, bs, ncp)
###
###        hmesh = {}
###        if gal:
###            hpath = path + ftype%(bs, ncf, seed, stepf) + 'galaxies_n05/galcat/'
###            hposd = tools.readbigfile(hpath + 'Position/')
###            massd = tools.readbigfile(hpath + 'Mass/').reshape(-1)*1e10
###            galtype = tools.readbigfile(hpath + 'gal_type/').reshape(-1).astype(bool)
###            hmesh['pnn'] = tools.paintnn(hposd, bs, ncp)
###            #hmesh['mnn'] = tools.paintnn(hposd, bs, ncp, massd)
###            hmesh['pnnsat'] = tools.paintnn(hposd[galtype], bs, ncp)
###            hmesh['pnncen'] = tools.paintnn(hposd[~galtype], bs, ncp)
###        else:
###            hposall = tools.readbigfile(path + ftype%(bs, ncf, seed, stepf) + 'FOF/PeakPosition/')[1:]    
###            hposd = hposall[:num].copy()
###            massd = massall[:num].copy()
###            hmesh['pnn'] = tools.paintnn(hposd, bs, ncp)
###            
###        meshes[seed] = [mesh, hmesh]
###
###        print('All the mesh have been generated for seed = %d'%seed)
###
###        #Create training voxels
###        ftlist = [mesh[i].copy() for i in ftname]
###        ftlistpad = [np.pad(i, pad, 'wrap') for i in ftlist]
###    #     targetmesh = hmesh['pnn']
###        targetmesh = [hmesh['pnncen'], hmesh['pnnsat']]
###        ntarget = len(targetmesh)
###
###        for i, size in enumerate(cube_sizes):
###            numcubes = int(num_cubes/size*4)
###            features, target = dtools.randomvoxels(ftlistpad, targetmesh, numcubes, max_offset[i], 
###                                                size, cube_sizesft[i], seed=seed, rprob=0)
###            cube_features[i] = cube_features[i] + features
###            cube_target[i] = cube_target[i] + target
###
###    # #
###    for i in range(cube_sizes.size):
###        cube_target[i] = np.stack(cube_target[i],axis=0)
###        cube_features[i] = np.stack(cube_features[i],axis=0)
###        print(cube_features[i].shape, cube_target[i].shape)
###
###
