import math
import os
import torch
import numpy as np
from ml_collections.config_dict import config_dict
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset
import VRNN.perceiver.perceiver as perceiver
import VRNN.perceiver.perceiver_helpers as perceiver_helpers
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from VRNN.perceiver.Utils import random_mask_image_grid, random_mask_image_dot, imshow, random_mask_image_group
from VRNN.perceiver.data_set_loader import *

SAMPLE_LABEL_MODALITIES = ('audioset_label', 'coco_labels', 'imagenet_label',
                           'jft_label', 'multi_nli_labels', 'objects365_labels', 'label')

DEFAULT_MODEL_KWARGS = config_dict.ConfigDict({
    # Canonical models:
    'PerceiverIO': {
        # The size of the raw ('latent') position encodings.
        # If != the embedding size, will be projected.
        'num_position_encoding_channels': 512,
        'activation_name': 'sq_relu',
        'dropout_prob': 0.0,
        'drop_path_rate': 0.0,
    },
    'HiP': {
        # The size of the raw ('latent') position encodings.
        # If != the embedding size, will be projected.
        'num_position_encoding_channels': 512,
        'regroup_type': 'reshape',
        'activation_name': 'sq_relu',
        'dropout_prob': 0.0,
        'drop_path_rate': 0.0,
        # Optional index dimension overrides:
    },
    'HiPClassBottleneck': {
        # The size of the raw ('latent') position encodings.
        # If != the embedding size, will be projected.
        'num_position_encoding_channels': 512,
        'regroup_type': 'reshape',
        'activation_name': 'sq_relu',
        'dropout_prob': 0.3,
        'drop_path_rate': 0.0,
        'label_modalities': SAMPLE_LABEL_MODALITIES,
    },
})


def generate_model(model_base_name, model_variant_name, mock_data):
    return perceiver.build_perceiver(
        input_data=mock_data,
        model_base_name=model_base_name,
        model_variant_name=model_variant_name,
        model_kwargs=DEFAULT_MODEL_KWARGS[model_base_name])


class PerceiverClassificationTrainer:
    def __init__(self, train_set, test_set, pre_train_set, image_size, num_channel, classes, batch_size, store_path,
                 writer, mask_percent=0.5, grid_size=2, mask_type="g", worker=2, enable_plot=False, print_fre=50,
                 setting='Mini', use_weight=False):
        self.train_set = train_set
        self.test_set = test_set
        self.image_size = image_size
        self.image_side_length = int(math.sqrt(image_size))
        self.num_channel = num_channel
        self.classes = classes
        self.num_classes = len(self.classes)
        self.batch_size = batch_size
        self.store_path = store_path
        self.print_fre = print_fre
        self.mask_percent = mask_percent
        self.grid_size = grid_size
        self.enable_plot = enable_plot
        self.setting = setting
        self.mask_type = mask_type
        self.circle_mask_percent = None

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.store_path = store_path
        self.out_keys = perceiver_helpers.ModelOutputKeys
        self.writer = writer

        self.model = generate_model('HiPClassBottleneck', setting, self.generate_mock_input())
        self.model.to(self.device)

        self.split_point = int(len(train_set) * 0.8)
        train_sampler = Subset(self.train_set, np.arange(0, self.split_point))
        self.train_loader = torch.utils.data.DataLoader(train_sampler, batch_size=self.batch_size,
                                                        shuffle=False, num_workers=worker)

        valid_sampler = Subset(self.train_set, np.arange(self.split_point, len(train_set)))
        self.vali_loader = torch.utils.data.DataLoader(valid_sampler, batch_size=self.batch_size,
                                                       shuffle=False, num_workers=worker)

        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size,
                                                       shuffle=False, num_workers=worker)

        self.pre_train_loader = torch.utils.data.DataLoader(pre_train_set, batch_size=self.batch_size,
                                                            shuffle=False, num_workers=worker)

        if use_weight:
            weight = torch.tensor(self.count_sample_in_each_class(), dtype=torch.float32) / self.split_point
        else:
            weight = None
        self.criterion = nn.CrossEntropyLoss(weight=weight)
        self.reconstruction_loss = nn.MSELoss()

        if self.mask_type == 'grid':
            print(f'Info: Use grid mask, with mask rate {self.mask_percent}')
        elif self.mask_type == 'random_dot':
            self.circle_mask_percent = self.mask_percent / self.grid_size / self.grid_size / 3
            print(f"Info: Use random dot mask, with actual mask rate {self.circle_mask_percent}")
        elif self.mask_type == 'group':
            print(f"Info: Use group mask, with mask rate {self.mask_percent}, group size {self.grid_size}")
            if self.image_side_length / self.grid_size != 16:
                print("Warning: Use group mask but group number is not recommended number 16")

    
    def generate_mock_input(self):
        return {
            'image': torch.from_numpy(
                np.random.random((self.batch_size, self.image_size//self.num_channel, self.num_channel)).astype(np.float32)).to(self.device),

            'label': torch.from_numpy(np.random.randint(low=0, high=1,
                                              size=(self.batch_size, self.num_classes, 1)).astype(np.float32)).to(self.device),
        }

    def preprocess_data(self, data, labels):
        batch = data.shape[0]
        one_hot = torch.zeros((batch, len(self.classes), 1))
        for i, l in enumerate(labels):
            one_hot[i][l][0] = 1
        return {
            'image': data.transpose(1, -1).reshape(batch, self.image_size, self.num_channel).to(self.device),
            'label': one_hot.to(self.device)
        }

    def display_test_image(self, train_loader):
        dataiter = iter(train_loader)
        images, labels = next(dataiter)

        # show images
        imshow(torchvision.utils.make_grid(images))
        # print labels
        print(' '.join(f'{self.classes[labels[j]]:5s}' for j in range(self.batch_size)))

    def preprocess_pre_train_data(self, data, labels):
        batch = data.shape[0]
        one_hot = torch.zeros((batch, len(self.classes), 1))
        for i, l in enumerate(labels):
            one_hot[i][l][0] = 1

        original_image = data.transpose(1, -1)
        # batch, width, height, channel
        if self.mask_type == 'grid':
            masked_image = random_mask_image_grid(original_image, self.grid_size, self.mask_percent, self.num_channel)
        elif self.mask_type == 'random_dot':
            masked_image = random_mask_image_dot(original_image, self.grid_size, self.circle_mask_percent,
                                                 self.num_channel)
        elif self.mask_type == 'group':
            masked_image = random_mask_image_group(original_image, self.grid_size, self.mask_percent, self.num_channel)
        else:
            masked_image = original_image
        original_image = original_image.reshape(batch, self.image_size//self.num_channel, self.num_channel)
        return {
                   'image': masked_image.to(self.device),
                   'label': one_hot.to(self.device)
               }, original_image

    def pre_train(self, epochs, use_exist_model=False):
        if use_exist_model:
            self.model.load_state_dict(torch.load(self.store_path))

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # loop over the dataset multiple times
        counter = 0
        for ep in range(epochs):
            for i, data in enumerate(self.pre_train_loader, 0):
                inputs, labels = data
                model_input, original_image = self.preprocess_pre_train_data(inputs, labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(model_input, is_training=True)
                reconstruct_loss = self.reconstruction_loss(outputs[self.out_keys.INPUT_RECONSTRUCTION]['image'],
                                                            original_image.to(self.device))
                reconstruct_loss.backward()
                clip_grad_norm_(self.model.parameters(), 0.1)
                optimizer.step()

                if i % self.print_fre == self.print_fre - 1:
                    self.writer.add_scalar("Reconstruction loss (sampled by print freq)", reconstruct_loss,
                                           global_step=counter)

                    self.plot_sample(original_image[0], "original", counter)
                    self.plot_sample(model_input['image'][0], "masked", counter)
                    self.plot_sample(outputs[self.out_keys.INPUT_RECONSTRUCTION]['image'][0], "reconstruction",
                                     counter)

                    counter += 1

            # save model every episode
            print(f'Ep {ep} done')
            torch.save(self.model.state_dict(), self.store_path)
        print('Finished Pre-Training')

    def count_sample_in_each_class(self):
        sample_count = [0] * self.num_classes
        for data in self.train_loader:
            _, labels = data
            for ind in labels:
                sample_count[ind] += 1
        return sample_count

    def train(self, epochs, use_exist_model=False):
        if use_exist_model:
            self.model.load_state_dict(torch.load(self.store_path))

        optimizer = optim.Adam(self.model.parameters(), lr=0.00005, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, 'min')

        # loop over the dataset multiple times
        counter = 0
        for ep in range(epochs):
            for i, data in enumerate(self.train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                model_input = self.preprocess_data(inputs, labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(model_input, is_training=True)
                loss = self.criterion(outputs[self.out_keys.OUTPUT], model_input['label'])

                loss.backward()
                clip_grad_norm_(self.model.parameters(), 0.1)
                optimizer.step()

                # print statistics
                if i % self.print_fre == self.print_fre - 1:
                    # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_fre:.3f}')
                    self.writer.add_scalar("Mini batch loss (sampled by print freq)", loss, global_step=counter)

                    self.plot_sample(model_input['image'][0], "original", counter)
                    self.plot_sample(outputs[self.out_keys.INPUT_RECONSTRUCTION]['image'][0], "reconstruction",
                                     counter)

                    # record accuracy
                    _, predicted = torch.max(outputs[self.out_keys.OUTPUT], 1)
                    accuracy = (torch.squeeze(predicted).to('cpu') == labels).sum().item() / labels.size(0)
                    self.writer.add_scalar("Accuracy (sampled by print freq)", accuracy, global_step=counter)
                    counter += 1

            vali_loss = self.validate()
            self.writer.add_scalar("Per epoch Validation Loss", vali_loss, global_step=ep)
            scheduler.step(vali_loss)

            # save model every episode
            torch.save(self.model.state_dict(), self.store_path)
        print('Finished Training')

    def validate(self):
        with torch.no_grad():
            loss = 0
            for data in self.vali_loader:
                images, labels = data
                model_input = self.preprocess_data(images, labels)

                # calculate outputs by running images through the network
                outputs = self.model(model_input, is_training=False)
                # the class with the highest energy is what we choose as prediction
                loss += self.criterion(outputs[self.out_keys.OUTPUT], model_input['label'])
            return loss

    def test(self):
        self.model.load_state_dict(torch.load(self.store_path))

        correct = 0
        total = 0
        step_counter = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                model_input = self.preprocess_data(images, labels)

                # calculate outputs by running images through the network
                outputs = self.model(model_input, is_training=False)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs[self.out_keys.OUTPUT], 1)
                total += labels.size(0)
                correct += (torch.squeeze(predicted).to('cpu') == labels).sum().item()

                self.plot_sample(model_input['image'][0], "original", step_counter)
                self.plot_sample(outputs[self.out_keys.INPUT_RECONSTRUCTION]['image'][0], "reconstruction",
                                 step_counter)
                step_counter += 1

        # print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    def plot_sample(self, img_tensor, tag, step):
        if not self.enable_plot:
            return
        img = torch.reshape(img_tensor, (self.image_side_length, self.image_side_length, self.num_channel)).cpu()
        self.writer.add_image(tag=tag, img_tensor=img, dataformats='WHC', global_step=step)


if __name__ == '__main__':
    log_dir = "runs"
    model_path = './VRNN/perceiver/model/'
    model_name = "Mini_ImageNet_400_mae_pre_train_patch_6"
    # model_name = "mini_mnist_loca_test"
    model_extend = ".pth"
    model_store = model_path + model_name + model_extend
     
    summary_writer = SummaryWriter(log_dir=log_dir)
    os.makedirs(model_path, exist_ok=True)  # Creates the model directory if it doesn't exist
    os.makedirs('./VRNN/perceiver/data', exist_ok=True)  # Creates directory if not exists
    #Load ImageNet100
    imagenet100_path='/home/zsheikhb/transfer/downloads/ImageNet100/'  # Replace with actual path
    
    
    train_data, test_data, pre_train_data, label_classes, flat_size, image_channel = imagenet100(
        data_path=imagenet100_path, 
        image_size=96  # Smaller size for memory efficiency
    )
    #train_data, test_data, pre_train_data, label_classes, flat_size, image_channel = stl10(download=True)

    model = PerceiverClassificationTrainer(train_data, test_data, pre_train_data, flat_size, image_channel,
                                           label_classes, batch_size=55,
                                           store_path=model_store, worker=1, grid_size=6, mask_percent=0.8,
                                           print_fre=2, mask_type='group', setting='Mini', use_weight=False,
                                           enable_plot=False, writer=summary_writer)
    model.pre_train(400, use_exist_model=False)
    # model.train(200, use_exist_model=False)
    # model.test()