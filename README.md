# ImageSynth
Various attempts to create images using different deep generative models. This project was mainly to learn the basics of creating APIs, UIs, and setting up monitoring for machine learning models. GANs seemed like a really cool (and challenging) architecture to work with so I thought seeing the results would be interesting.

## Approaches
### GAN
Uses an architecture similar to DC-GAN but applied on MNIST (for now). Discriminator is a stack of 5 convolutional layers and the generator is a stack of 5 convtranspose (or deconvolutional) layers. The discriminator attempts to classify data as fake (generated by the generative network) or real while the discriminator attempts to create data that matches the distributions in the real dataset used to train it. This occurs in sequence (D -> G -> D -> ...) and the hope is that one does not dominate the other.

### VAE
This approach uses an encoder-decoder architecture to try and create new data samples that model the type of data it has already encountered. The VAE will encounter data samples during training and try to produce latent vectors that follow a normal distribution. The encoder is essentially a series of convolutional transforms applied to set of matrices that represent images. The encoder will create data samples that follow a Gaussian distribution. The decoder is fed the z-values from the encoder and apply deconvolutional (convtranspose) transforms to try and create data samples similar to the input images.

## To run (WIP)
### Gain access to the code.
This project requires python 3 (python 2 will lose support January 2020).
1. Clone the repo using `git clone https://github.com/sidg54/imagesynth.git`.
2. Change directory to the newly cloned repo `cd imagesynth`.
3. Create a virtual environment, for example, `virtualenv .env` and activate with `source .env/bin/activate`.
4. Download all required dependencies from the requirements.txt file using pip (or pip3 if you have multiple python versions installed) using `pip install -r requirements.txt` or `pip3 install -r requirements.txt`.
5. Find a configuration file that has the parameters and models you would like to use. For example, the gan_mnist.yml file contains the base configuration for the DC-GAN-like architecture being run on the MNIST data.
6. Run the project using `python main.py <name_of_selected_config_file>` or `python3 main.py <name_of_selected_config_file>`.

### To extend (add your own work) to this project.
There are a few things that will need to be done to run an entirely new set of models, configurations, datasets, etc. This guide will show you how to go through and extend the classes to ensure you can run your own agent with its own models and data.

#### New Dataset
To create a new dataset, simply follow the steps below.
1. Create a new file in the ./dataloaders/ directory with the name of the .
2. Create a class with a relevant name concatenated with "DataLoader" at the end (see MNISTDataLoader). This class should extend the BaseDataLoader class which can be found in ./dataloaders/dataloader.py.
3. Override the methods in this and load the data similar to in [./dataloaders/mnist.py](https://github.com/sidg54/imagesynth/blob/master/dataloaders/mnist.py)
4. Ensure the new dataloader's load_data method returns both a train_loader and a test_loader.

#### New Model
To create a new model, just do the following.
1. Create a new file (or files if you're creating a multi-model architecture like a GAN or seq2seq) in the ./models/gan/ directory (if it is a network that will be used in a GAN). Or, create a new directory in ./models/ with a name fitting the new architecture.
2. Create a class (or classes) that extends nn.Module from Pytorch and implement the relevant class and create your architecture.
3. If you created a new directory, create an __init__.py file and copy-paste the contents of [this file](https://github.com/sidg54/imagesynth/blob/master/utils/__init__.py) into it.
4. If you created a new directory, create the architecture in another file called main.py in your new directory. Create a class similar to [this one for GANs](https://github.com/sidg54/imagesynth/blob/master/models/GAN/main.py).


## Future Work
1. New models: VAE, flow models
2. More datasets: ImageNet, Celeb-A
3. New optimizers (mainly just for checking to see how they stack up against existing ones).
4. Set the website up so users can interact with a UI to see how the models are generating things.
5. Set up monitoring (Kubeflow maybe?)
6. Use namedtensors.