from setuptools import setup

setup(name='galmodel',
      version='0.1',
      description='Modeling galaxies in simulations with TensorFlow',
      url='https://github.com/modichirag/galmodel',
      author='Chirag Modi',
      author_email='modichirag@berkeley.edu',
      license='MIT',
      packages=['galmodel'],
      install_requires=['flowpm', 'tensorflow'])
