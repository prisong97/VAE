import numpy as numpy
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F

from helper import OHE

class CVAE(nn.Module):
    
    """
    Implementation of the conditional-VAE model
    """
    
    def __init__(self, h_dims, y_dim, conditional_vector_dim):
        super(CVAE, self).__init__()
        self.h_dims = h_dims
        self.y_dim = y_dim
        self.conditional_vector_dim = conditional_vector_dim            
        

    def get_layers(self):
        """
        prepare the layers of the encoder and decoder -- input to decoder network has additional
        nodes to accomodate conditional variable
        
        conditional_vector encodes the conditional variable 
        """
        
        y_full_shape = self.y_dim + self.conditional_vector_dim
        self.h_dims = torch.cat([torch.tensor([y_full_shape]), self.h_dims])  
        
        # reverse and modify the input layer of the decoder network
        self.h_rev_dims = torch.flip(self.h_dims, dims=[0])
        self.h_rev_dims[0] = self.h_rev_dims[0] + self.conditional_vector_dim
        self.h_rev_dims[-1] = self.y_dim
             

    def build_encoder(self):
        """
        constructs the data for the forward pass (encoder)
        """
        
        self.input = nn.ModuleList() 

        for i in range(len(self.h_dims)-2):
            input_size = self.h_dims[i]
            output_size = self.h_dims[i+1]
            linear = nn.Sequential(nn.Linear(in_features=input_size, out_features=output_size), nn.ReLU())
            self.input.append(linear)

        # for final layer, output layer should be x2 -- one for mu, one for log_sigma
        input_size = self.h_dims[-2]
        output_size = self.h_dims[-1]
        linear = nn.Sequential(nn.Linear(in_features=input_size, out_features=output_size*2), nn.ReLU())
        self.input.append(linear)
       
    
    def forward(self,y, nn_sequential_list):
        """
        builds the encoder/decoder
        """
        encoder = nn.Sequential(*nn_sequential_list)(y)
 
        return encoder 

    
    def build_decoder(self):
        """
        constructs the data for the forward pass (decoder)
        """
        
        self.output = nn.ModuleList() 

        for i in range(len(self.h_rev_dims)-1):
            input_size = self.h_rev_dims[i]
            output_size = self.h_rev_dims[i+1]
            linear = nn.Sequential(nn.Linear(in_features=input_size, out_features=output_size), nn.ReLU())
            self.output.append(linear)
      
    
    def get_param(self, mu_logsigma, param = 'mu'):
        """
        this method queries the columns of interest (corresponding to mu or log_sigma, as specified)
        
        dim of latent_layer mu_logsigma is (N x (z_size+size)) where:
        
            columns 1:z_size represent mu, and 
            columns z_size+1:z_size represent log_sigma
        """
        
        batch_size = mu_logsigma.shape[0]
        
        z_size = self.h_dims[-1]
        param_size = (z_size).type(torch.int)
        
        if param == 'mu':
            cols =  torch.tensor([i for i in range(param_size)])
            
        elif param == 'log_sigma':
            cols = torch.tensor([i for i in range(param_size, param_size + param_size)])
        
        else:
            raise ValueError('Please specify a valid param name: mu or log_sigma')
        
        # gather the torches
        all_columns = cols.repeat(batch_size, 1)
        param_batch = torch.gather(mu_logsigma, 1, all_columns).flatten(start_dim=1)
        
        return param_batch
        
        
 
    def z_sample(self, mu_batch, logsigma_batch, epsilon):
        """
        constructs the z sample given mu, log_sigma and epsilon, 
        through the reparameterisation trick
        """

        z_sample = mu_batch + torch.mul(torch.exp(logsigma_batch), epsilon)

        return z_sample
    
    
    def construct_model(self):
        """
        constructs the model which comprises of the encoder and decoder
        """
        
        # initialise the encoder/decoder layers
        self.get_layers()
        
        # build encoder
        self.build_encoder()
        
        # build decoder
        self.build_decoder()     

    
    def loss(self, y, conditional_vector):
        """
        computes the mean loss on each pass
        """

        batch_size = y.shape[0]
        
        z_size = self.h_dims[-1]
        sigma_size = (z_size).type(torch.int)
        epsilon = torch.randn(batch_size, sigma_size)

        encoder_input = torch.concat([y, conditional_vector], axis=1)
        latent_layer = self.forward(encoder_input, self.input)
        
        mu_batch = self.get_param(latent_layer, param = 'mu')
        logsigma_batch = self.get_param(latent_layer, param = 'log_sigma')
        z_sample = self.z_sample(mu_batch, logsigma_batch, epsilon)
        
        logqzx = -torch.mean(torch.sum(logsigma_batch + 0.5*epsilon**2, axis=1), axis=0) #- z_size*0.5*torch.log(torch.tensor(2)*torch.pi)
        logpz =  -0.5*torch.mean(torch.sum(epsilon**2, axis=1),axis=0) #-z_size*0.5*torch.log(torch.tensor(2)*torch.pi)
        
        # assume std normal dist about the output
        decoder_input = torch.concat([z_sample, conditional_vector], axis=1)
        output = self.forward(decoder_input, self.output)
        logpxz =  -0.5*torch.mean(torch.sum(torch.square(y-output), axis=1), axis=0) # -z_size*0.5*torch.log(torch.tensor(2)*torch.pi)
        
        loss = -(logqzx + logpz + logpxz) 

        return loss
    
    
    def initialise_model(self):
        """
        This function initialises the model and optimisation parameters
        """
        
        self.construct_model()
        self.optimizer = optim.SGD(self.parameters(), lr=0.0001, momentum=0.9)
        self.clip_grad_norm = 0.02


    def train(self, n_iter, train_loader, clip_gradient=False):
        """
        This is the main function to run for training to take place
        """
        
        minibatch_len = len(train_loader)
        
        self.batch_loss = []
        
        for i in range(n_iter):
            examples = enumerate(train_loader)
            for j in range(minibatch_len):
                batch_idx, (example_data, example_targets) = next(examples)
                example_data = example_data.flatten(start_dim=1)
                conditional_vector = torch.tensor(OHE(example_targets.numpy()))
                conditional_vector = conditional_vector.type(torch.float)
                
                self.optimizer.zero_grad()
                loss = self.loss(example_data, conditional_vector)
                loss.backward()
                
             # clip grad norm and perform optimization step
                if clip_gradient:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
                    
                self.optimizer.step()           
                self.batch_loss.append(loss.detach().numpy().item())

    
    def predict(self, y, conditional_vector):
        """
        Given Y and the conditional vector, we reconstruct the output Y'
        """
        
        batch_size = y.shape[0]
        
        z_size = self.h_dims[-1]
        sigma_size = (z_size).type(torch.int)
        
        epsilon = torch.randn(batch_size, sigma_size)
        
        encoder_input = torch.concat([y, conditional_vector], axis=1)
        latent_layer = self.forward(encoder_input, self.input)
        
        mu_batch = self.get_param(latent_layer, param = 'mu')
        logsigma_batch = self.get_param(latent_layer, param = 'log_sigma')
        
        z_sample = self.z_sample(mu_batch, logsigma_batch, epsilon)
        
        decoder_input = torch.concat([z_sample, conditional_vector], axis=1)
        output = self.forward(decoder_input, self.output)
        
        return output
    
    
    def generate_samples(self, conditional_vector):
        """
        This function allows us to sample from the underlying data distribution (latent space)
        given the conditional vector
        """
        
        batch_size = conditional_vector.shape[0]
        
        z_size = self.h_dims[-1]
        sigma_size = (z_size).type(torch.int)
        
        epsilon = torch.randn(batch_size, sigma_size)
        
        decoder_input = torch.concat([epsilon, conditional_vector], axis=1)
        output = self.forward(decoder_input, self.output)
        
        return output
        