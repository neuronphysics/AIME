# AIME
## Implementation of the new model

To train the model and the agent, run `python main.py --env cheetah-run --use-regular-vae` for using regular VAE.
Run `python main.py --env cheetah-run` for using infinite GMM VAE. Override the default hyperparameters on the command line using for example `--state-size 100`.

# For running the infinite GMM variational autoencoder and using visdom use this line 
ssh -L 8097:127.0.0.1:8097 username@hostname
 
## For tracking the results using tensorboard, you should first connect to the cluster using this command line
ssh -L localhost:16010:localhost:6010 username@hostname

## Once you used slurm to run the script in another terminal you can run the following command
tensorboard --logdir="~/scalar" --port=6010

## Then you can open the browser and copy the following web address
http://localhost:16010 

# Reference
1. https://github.com/zhenwendai/RGP
2. https://github.com/EmbodiedVision/dlgpd
3. https://pyro.ai/examples/index.html
4. https://gpytorch.ai/
5. https://github.com/Kaixhin/PlaNet
6. https://github.com/ku2482/slac.pytorch
7. https://github.com/Olloxan/Pytorch-A2C
8. https://github.com/higgsfield/Imagination-Augmented-Agents
9. https://github.com/pranz24/pytorch-soft-actor-critic
