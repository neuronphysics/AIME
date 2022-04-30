# AIME
## Implementation of the new model
Note that the current work is being done in branch `implement-planner`

# For running the infinite GMM variational autoencoder and using visdom use this line 
ssh -L 8097:127.0.0.1:8097 username@hostname
 
#For tracking the results using tensorboard, you should first connect to the cluster using this command line
ssh -L localhost:16010:localhost:6010 username@hostname

#Once you used slurm to run the script in another terminal you can run the following command
tensorboard --logdir="~/scalar" --port=6010

#Then you can open the browser and copy the following web address
http://localhost:16010 

# Reference
1. https://github.com/zhenwendai/RGP
2. https://github.com/EmbodiedVision/dlgpd
3. https://pyro.ai/examples/index.html
4. https://gpytorch.ai/
5. https://github.com/Kaixhin/PlaNet
6. https://github.com/ku2482/slac.pytorch
