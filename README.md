# galmodel

Modeling halo and galaxy position and mass fields with deep networks, as well as reconstructing initial conditions with them.
All architectures are mixture density networks and their variants to learn the likelihood of the data. Given the nature of our data, all networks are implemented with 3 spatial dimensions. 

- pixel CNN (with masked convolutions)
- unet
- revenet
- glow
- resnets with spectral normed weight
- resnet

For reconstruction of initial conditions, we combine these networks with PM simulations in tensorflow developed in flowpm repository. 

