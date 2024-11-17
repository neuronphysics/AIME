import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from pathlib import Path
import numpy as np
from Hierarchical_StickBreaking_GMMVAE import InfGaussMMVAE, VAECritic, gradient_penalty  # Assuming InfGaussMMVAE class is saved separately
from utilities import AverageMeter
import copy
import time
import matplotlib
import os
from torch.utils.tensorboard import SummaryWriter
import re
import glob
import logging
import subprocess
import tarfile
from torch.utils.data import random_split, Subset

import torchvision

import datasets
from PIL import Image


class Chairs(torchvision.datasets.ImageFolder):
    """Chairs Dataset with background color control"""
    urls = {"train": "https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar"}
    files = {"train": "chairs_64"}
    img_size = (1, 64, 64)
    background_color = (255, 255, 255)  # White background

    def __init__(self, root='data/chairs', logger=logging.getLogger(__name__)):
        self.root = root
        self.logger = logger
        self.train_data = os.path.join(root, type(self).files["train"])
        
        self.transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.Lambda(self.set_background),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        if not os.path.isdir(root):
            self.logger.info("Downloading {} ...".format(str(type(self))))
            self.download()
            self.logger.info("Finished Downloading.")

        super().__init__(self.train_data, transform=self.transforms)

    def set_background(self, image):
        """Set the background color for transparent images"""
        if image.mode in ('RGBA', 'LA') or \
           (image.mode == 'P' and 'transparency' in image.info):
            # Convert image to RGBA if it isn't already
            alpha = image.convert('RGBA').split()[-1]
            bg = Image.new("RGBA", image.size, self.background_color)
            bg.paste(image, mask=alpha)
            return bg.convert('RGB')
        return image

    def preprocess_image(self, image_path):
        """Preprocess a single image"""
        img = Image.open(image_path)
        img = self.set_background(img)
        img = img.resize(self.img_size[1:], Image.LANCZOS)
        return img

    def download(self):
        """Download and extract dataset"""
        save_path = os.path.join(self.root, 'chairs.tar')
        os.makedirs(self.root, exist_ok=True)
        
        self.logger.info("Downloading chairs dataset...")
        subprocess.check_call(["curl", self.urls["train"],
                             "--output", save_path])

        self.logger.info("Extracting chairs...")
        with tarfile.open(save_path) as tar:
            tar.extractall(self.root)
        
        os.rename(os.path.join(self.root, 'rendered_chairs'), 
                 self.train_data)
        os.remove(save_path)

        # Preprocess all images
        self.logger.info("Preprocessing images...")
        for root, _, files in os.walk(self.train_data):
            for file in files:
                if file.endswith('.png'):
                    image_path = os.path.join(root, file)
                    img = self.preprocess_image(image_path)
                    img.save(image_path)

# Hyperparameters and configuration
hyperParams = {
    "batch_size": 100,
    "input_d": 1,
    "prior_alpha": 7.0,
    "prior_beta": 1.0,
    "K": 25,
    "hidden_d": 500,
    "latent_d": 200,
    "latent_w": 150,
    "LAMBDA_GP": 10,
    "LEARNING_RATE": 1e-4,
    "CRITIC_ITERATIONS": 5
}

def create_train_test_splits(dataset, train_ratio=0.8, seed=42):
    """
    Create train/test splits using indices.
    """
    # Set random seed
    np.random.seed(seed)
    
    # Generate indices
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    
    # Split indices
    split = int(np.floor(train_ratio * len(dataset)))
    train_indices, test_indices = indices[:split], indices[split:]
    
    # Create Subsets
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, test_dataset



def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def train(epoch):
    global iters
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    train_loss_avg = []
    net.train()
    discriminator.train()

    train_loss = 0
    train_loss_avg.append(0)
    num_batches = 0

    start = end = time.time()
    for batch_idx, (X, classes) in enumerate(train_loader):
        #measure time
        data_time.update(time.time()-end)

        X = X.to(device=device, dtype=torch.float)
        #print(f'size of input image {X.size()}')
        #optimizer.zero_grad()
        for _ in range(hyperParams["CRITIC_ITERATIONS"]):
            z_real, z_x_mean, z_x_sigma, c_posterior, w_x_mean, w_x_sigma, gmm_dist, z_wc_mean_prior, z_wc_logvar_prior, x_reconstructed = net(X)

            z_fake = gmm_dist.sample()


            critic_real = discriminator(z_real).reshape(-1)
            critic_fake = discriminator(z_fake).reshape(-1)
            gp = gradient_penalty(discriminator, z_real, z_fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + hyperParams["LAMBDA_GP"] * gp
            )
            dis_optim.zero_grad()
            loss_critic.backward(retain_graph=True)
            dis_optim.step()
        gen_fake = discriminator(z_fake).reshape(-1)


        outputs = net.get_ELBO(X)
        outputs.loss_dict["wasserstein_loss"] =  -torch.mean(gen_fake)
        vae_optim.zero_grad()

        outputs.loss_dict["WAE_GP"]=outputs.loss_dict["loss"]+outputs.loss_dict["wasserstein_loss"]


        outputs.loss_dict["WAE_GP"].backward()
        train_loss += outputs.loss_dict['loss'].item()

        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        #optimizer.step()
        vae_optim.step()

        train_loss_avg[-1] += outputs.loss_dict["WAE_GP"].item()
        num_batches += 1
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 10 == 0:

            print('epoch {} --- iteration {}: '
                      ', Date = {:0.3f}s '
                      ', Time ={:0.3f}s '
                      ', Gamma KL = {:.6f} '
                      ', kumar2beta KL = {:.6f} '
                      ', Z latent space KL = {:.6f} '
                      ', reconstruction loss = {:.6f} '
                      ', W context latent space KL = {:.6f}'
                      ', Wasserstein gradient penalty = {:.6f}'.format(epoch, batch_idx, data_time.avg, batch_time.avg, outputs.loss_dict['gamma_params_kld'].item(), outputs.loss_dict['kumar2beta_kld'].item(), outputs.loss_dict['z_latent_space_kld'].item(), outputs.loss_dict['recon'].item(),outputs.loss_dict['w_context_kld'].item(),outputs.loss_dict["wasserstein_loss"].item()))

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(X), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                outputs.loss_dict["WAE_GP"].item() / len(X)))
        if ((epoch > 1) and (epoch % 50 == 0)) :
            #plot low dimensional embedding

            reconst_images_concatenate    = outputs.x_reconst

            model_tsne = TSNE(n_components=2, random_state=0)
            #print(f'size of the latent dimension {z_states.shape}')
            z_embed = model_tsne.fit_transform(outputs.z_posterior.detach().cpu().numpy())
            #print(f'size of the embedding dimension {z_embed.shape}')
            classes = classes.detach().cpu().numpy()
            order_classes=set()
            ls =[x for x in classes if x not in order_classes and  order_classes.add(x) is None]
            matplotlib.use("Agg")
            fig = plt.figure()
            for ic in range(len(ls)):

                ind_class = classes == ic
                #print(f"index of classes {ind_class}")
                color = plt.cm.Set1(ic)
                plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, edgecolors=color, facecolors='none')
                plt.title("Latent Variable T-SNE per Class")
                fig.savefig(str(Path().absolute())+"/results/Hierarchical_StickBreaking_GMMVAE_Embedding_" + str(ic) + "_epoch_"+str(epoch)+".png")
            fig.savefig(str(Path().absolute())+"/results/Hierarchical_StickBreaking_GMMVAE_Embedding_epoch_"+str(epoch)+".png")
            # create original and reconstructed image-grids for
            d = X.shape[1]
            w = X.shape[2]
            h = X.shape[3]
            orig_img = X.view(len(X), d, w, h)
            #50 x 3 x 96 x 96
            recons_img = reconst_images_concatenate.view(len(X), d, w, h)

            canvas = torch.cat([orig_img, recons_img], 0)
            canvas = canvas.cpu().data
            # save_image(orig_img, f'{directory}orig_grid_sample_{b}.png')
            canvas_grid = make_grid(
                canvas, nrow=hyperParams["batch_size"], range=(-1, 1), normalize=True
            )
            save_image(canvas, str(Path().absolute())+"/results/Hierarchical_StickBreaking_GMMVAE_Reconstruct_GridSample_Prior_Epoch_"+str(epoch)+".png")

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    train_loss_avg[-1] /= num_batches
    print(f' average training loss : {train_loss_avg}')
    print ("Calculating the marginal likelihood...")

    return train_loss_avg, x_reconstructed, outputs.loss_dict['kumar2beta_kld'].item(), outputs.loss_dict["wasserstein_loss"].item(), outputs.loss_dict['z_latent_space_kld'].item()

def test(epoch):

    net.eval()
    test_loss = 0
    test_loss_reconstruction = 0
    with torch.no_grad():
        for i, (image_batch, _) in enumerate(test_loader):
            image_batch = image_batch.to(device=device,dtype=torch.float)
            out = net.get_ELBO(image_batch)

            if i == 0 and epoch % 20 == 0:
                # VAE Reconstruction
                # reconstruction error


                image_numpy = image_batch[-1].cpu().float().numpy()


                ###
                real_reco_numpy  = out.x_reconst[-1].cpu().float().numpy()
                print(image_numpy.shape)
                writer.add_image('Real images', image_numpy, epoch)

                writer.add_image('Reconstructed images from posterior', real_reco_numpy, epoch)
                """
                plotter.add_image(
                        image_numpy.transpose([2, 0, 1]),
                        win='Real',
                        opts={"caption": "test image"},
                )
                plotter.add_image(
                        reco_numpy.transpose([2, 0, 1]),
                        win='Fake',
                        opts={"caption": "reconstructed image"},
                )
                """

            test_loss += out.loss_dict['loss'].item()
            test_loss_reconstruction += out.loss_dict['recon'].item()

        test_loss /= len(test_loader.dataset)
        test_loss_reconstruction /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        print('====> Test set reconstuction loss: {:.4f}'.format(test_loss_reconstruction))

        writer.add_scalar('Test reconstruction loss', out.loss_dict['recon'].item(), epoch)
        writer.add_scalar( 'Test KL loss of Z', out.loss_dict['z_latent_space_kld'].item(), epoch)

def show_image(img, filename):
    img = img.clamp(0, 1)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(filename)

def visualise_output(images, model,filename):

    with torch.no_grad():
        model.eval()
        images = images.to(device=device)

        _, _, _, _, _, _, _, _, _, out = model(images)
        img =torchvision.utils.make_grid(out.detach().cpu())
        img = 0.5*(img + 1)
        npimg = np.transpose(img.numpy(),(1,2,0))
        fig = plt.figure(dpi=700)
        plt.imshow(npimg)
    plt.savefig(filename)

if __name__ == "__main__":

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(log_dir='scalar', comment='runs/Oxford_Pet_experiment')


    # Data transformation and loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    print("Loading Chairs dataset...")
    # Define transformations for the images
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    ])

    # Load the training dataset
    train_dataset = torchvision.datasets.OxfordIIITPet(
    root='./data',
    split='trainval',
    target_types='category',
    transform=transform,
    download=True
    )

    # Load the test dataset
    test_dataset = torchvision.datasets.OxfordIIITPet(
    root='./data',
    split='test',
    target_types='category',
    transform=transform,
    download=True
    )



    classes = [
    "Abyssinian",
    "American Bulldog",
    "American Pit Bull Terrier",
    "Basset Hound",
    "Beagle",
    "Bengal",
    "Birman",
    "Bombay",
    "Boxer",
    "British Shorthair",
    "Chihuahua",
    "Egyptian Mau",
    "English Cocker Spaniel",
    "English Setter",
    "German Shorthaired",
    "Great Pyrenees",
    "Havanese",
    "Japanese Chin",
    "Keeshond",
    "Leonberger",
    "Maine Coon",
    "Miniature Pinscher",
    "Newfoundland",
    "Persian",
    "Pomeranian",
    "Pug",
    "Ragdoll",
    "Russian Blue",
    "Saint Bernard",
    "Samoyed",
    "Scottish Terrier",
    "Shiba Inu",
    "Siamese",
    "Sphynx",
    "Staffordshire Bull Terrier",
    "Wheaten Terrier",
    "Yorkshire Terrier"
    ]


    print("Done!")
    ##########
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=hyperParams["batch_size"], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=hyperParams["batch_size"], shuffle=True, num_workers=0)
    # Get a batch of images and labels
    data, labels = next(iter(train_loader))

    # Visualize the images in a grid
    img_grid = torchvision.utils.make_grid(data)
    matplotlib_imshow(img_grid)

    # Print the corresponding class names
    print(f"Labels: {[classes[label] for label in labels]}")


    img_grid = torchvision.utils.make_grid(data)

    # show images
    matplotlib_imshow(img_grid, one_channel=True)

    # write to tensorboard
    writer.add_image('Oxford-IIIT Pet Dataset_images', img_grid)
    writer.close()
    ###*********************************************************###

    test_data, test_label  = next(iter(test_loader))
    img_width = test_data.shape[3]
    print('image size in test data ', img_width)

    raw_img_width = data.shape[3]
    print('images size in the training data: ', raw_img_width)


    net = InfGaussMMVAE(hyperParams, 25, 3, 200, 150, 500, device, raw_img_width, hyperParams["batch_size"],include_elbo2=True)

    params = list(net.parameters())
    #if torch.cuda.device_count() > 1:
    #    net = nn.DataParallel(net)
    net = net.to(device=device)
    for name, param in net.named_parameters():
        if param.device.type != 'cuda':
            print('param {}, not on GPU'.format(name))
    ###*********************************************************###

    #net.cuda()
    discriminator = VAECritic(net.z_dim)
    
    trainable_params = net.get_trainable_parameters()
    vae_optim = torch.optim.Adam(trainable_params, lr = hyperParams["LEARNING_RATE"], betas=(0.5, 0.9))
    dis_optim = torch.optim.Adam(discriminator.parameters(), lr = 0.5 * hyperParams["LEARNING_RATE"], betas=(0.5, 0.9))

    vae_scheduler = torch.optim.lr_scheduler.StepLR(vae_optim, step_size = 30, gamma = 0.5)
    dis_scheduler = torch.optim.lr_scheduler.StepLR(dis_optim, step_size = 30, gamma = 0.5)
    #optimizer = optim.Adam(net.parameters(), lr=hyperParams["LEARNING_RATE"])
    cwd = os.getcwd()
    dirpath = os.path.join(cwd, "results")
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    else:
        pass
    ##optimizer = AdaBound(net.parameters(), lr=0.0001)

    img_list = []
    num_epochs= 2001
    grad_clip = 1.0
    iters=0

    avg_train_loss=[]
    best_loss = np.finfo(np.float64).max # Random big number (bigger than the initial loss)
    best_epoch = -1
    regex = re.compile(r'\d+')
    start_epoch = 0
    for file in os.listdir("./results/"):
        if file.startswith("model_Hierarchical_StickBreaking_GMMVAE_") and file.endswith(".pth"):
            print(file)
            list_of_files = glob.glob(str(Path().absolute())+"/results/model_Hierarchical_StickBreaking_GMMVAE*.pth") # * means all if need specific format then *.csv
            latest_file = max(list_of_files, key=os.path.getctime)
            print("latest saved model: ")
            print(latest_file)
            checkpoint = torch.load(latest_file)
            net.load_state_dict(checkpoint['model_state_dict'])
            vae_optim.load_state_dict(checkpoint['vae_optimizer_state_dict'])
            dis_optim.load_state_dict(checkpoint['disc_optimizer_state_dict'])
            discriminator.load_state_dict(checkpoint['citic_state_dict'])
            best_loss   = checkpoint['best_loss']
            start_epoch = int(regex.findall(latest_file)[-1])
            print(f"start of epoch: {start_epoch}")


    ####Training

    #global plotter
    #plotter = VisdomLinePlotter(env_name='HIERARCHICAL_STICKBREAKING_GMMVAE_PLOTS')

    #scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=2)
    for epoch in range(start_epoch, num_epochs):

            average_epoch_loss, out, elbo2, wasserstein_loss, latent_dimension_kld = train(epoch)
            if (epoch % 10 == 0):
                test(epoch)
            avg_train_loss.extend(average_epoch_loss)
            if (epoch % 50 == 0) and (epoch > 0):

                img = make_grid(out.detach().cpu())
                img = 0.5*(img + 1)
                npimg = np.transpose(img.numpy(),(1,2,0))
                fig = plt.figure(dpi=600)
                plt.imshow(npimg)
                plt.imsave(str(Path().absolute())+"/results/reconst_images_Hierarchical_StickBreaking_GMMVAE_"+str(epoch)+"_epochs.png", npimg)
            # plot beta-kumaraswamy loss
            #plotter.plot('KL of beta-kumaraswamy distributions', 'val', 'Class Loss', epoch, elbo2)

            # plot loss
            #print(average_epoch_loss)
            #print(epoch)
            #plotter.plot('Total loss', 'train', 'Class Loss', epoch, average_epoch_loss[0])
            net.eval()
            discriminator.eval()
            if (epoch % 500 == 0) and (epoch//500> 0):
                torch.save({
                        'best_epoch': epoch,
                        'model_state_dict': copy.deepcopy(net.state_dict()),
                        'vae_optimizer_state_dict':copy.deepcopy(vae_optim.state_dict()),
                        'disc_optimizer_state_dict':copy.deepcopy(dis_optim.state_dict()),
                        'citic_state_dict':copy.deepcopy(discriminator.state_dict()),
                        #'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                        'best_loss': average_epoch_loss[-1],
                }, str(Path().absolute())+"/results/model_Hierarchical_StickBreaking_GMMVAE_"+str(epoch)+".pth")
                print("Saved Best Model avg. Loss : {:.4f}".format(average_epoch_loss[-1]))
            #scheduler.step(average_epoch_loss[-1])
            writer.add_scalar('training average loss', average_epoch_loss[-1], epoch )
            writer.add_scalar('training loss_kld_Kumar_beta', elbo2, epoch )
            writer.add_scalar('training loss_wasserstein', wasserstein_loss, epoch )
            writer.add_scalar('training loss_kld_z',latent_dimension_kld, epoch )

    writer.flush()

    print('Finished Trainning')

    #plot the loss values of trained model
    fig = plt.figure()
    plt.plot(avg_train_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(str(Path().absolute())+"/results/Loss_Hierarchical_StickBreakingGMM_VAE.png")

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(make_grid(data[0].to(device=device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    ###plot original versus fake images
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.savefig(str(Path().absolute())+"/results/real_vs_fake_images.jpg")

    #plotter.viz.save([plotter.env])###???

    images, labels = iter(test_loader).next()
    show_image(torchvision.utils.make_grid(images[1:50],10,5), str(Path().absolute())+"/results/Original_Images_Hierarchical_StickBreaking_GMM_VAE.png")
    visualise_output(images, net,str(Path().absolute())+"/results/Reconstructed_Images_Hierarchical_StickBreaking_VAE.png")
