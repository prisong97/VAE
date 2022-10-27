# Implementation of the Conditional Variational Autoencoder (CVAE)

The CVAE is a generative model that allows us to sample from the data distribution after training, conditioned on some covariates.
We implement the model using the MNIST dataset, where our conditional variables are:
- The digit (from 0-9),
- The angle of rotation (from -45 to 45 degrees).


![image info](./figures/images/reconstructed_images.pdf)

To add:
- config file
- GPU support
