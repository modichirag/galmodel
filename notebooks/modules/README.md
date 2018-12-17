# Description of trained modules

  - **1544168303/likelihood**:
  This module was trained in [resnet_galaxies_estimator.ipynb](https://github.com/modichirag/galmodel/blob/master/notebooks/resnet_galaxies_estimator.ipynb)
  to predict the likelihood of central and satellite galaxies here is an example of how to use it:
    ```python
    module = hub.Module('modules/1544168303/likelihood')
    xx = tf.placeholder(tf.float32, shape=[None, cube_sizeft, cube_sizeft, cube_sizeft, nchannels], name='input')
    yy = tf.placeholder(tf.float32, shape=[None, cube_size, cube_size, cube_size, 2], name='labels')
    likelihood = module(dict(features=xx, labels=yy), as_dict=True)['loglikelihood']
    # The output log likelihood has the same shape as yy, to compute the full likelihood, for use in loss function:
    loss = - tf.reduce_sum(likelihood)
    ```
