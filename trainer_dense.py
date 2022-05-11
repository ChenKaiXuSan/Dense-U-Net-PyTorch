# %% 
"""

"""
from collections import defaultdict
import os
import time
import torch

import torchvision
from torch.optim import lr_scheduler

from models.DenseUNet import UNet
from utils.utils import *
from utils.helper import *

import copy

# %%
class Trainer_dense_unet(object):
    def __init__(self, train_dataloader, val_dataloader, test_dataloader, config):
        super(Trainer_dense_unet, self).__init__()

        # data loader 
        self.train_data_loader = train_dataloader
        self.val_data_loader = val_dataloader
        self.test_data_loader = test_dataloader

        # exact model and loss 
        self.model = config.model

        # model hyper-parameters
        self.imsize = config.img_size 
        self.channels = config.channels

        self.epochs = config.epochs + 1
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers 
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.dataset = config.dataset 
        self.use_tensorboard = config.use_tensorboard

        # path
        self.image_path = config.dataroot 
        self.log_path = config.log_path
        self.sample_path = config.sample_path
        self.version = config.version

        # step 
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # path with version
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        if self.use_tensorboard:
            self.build_tensorboard()

        self.build_model()

        # start with trained model 
        # if self.pretrained_model:
        #     self.load_pretrained_model()

    def train(self):
        '''
        Training
        '''

        best_model_wts = copy.deepcopy(self.unet.state_dict())
        best_loss = 1e10

        for epoch in range(self.epochs):
            # start time
            start_time = time.time()

            print('Epoch {}/{}'.format(epoch, self.epochs - 1))
            print('-' * 30)

            # each epoch has a training and validation phase 
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.exp_lr_scheduler.step()
                    for param_group in self.unet_optimizer.param_groups:
                        print('LR', param_group['lr'])

                    self.unet.train()

                    data_loader = self.train_data_loader
                else:
                    self.unet.eval()

                    data_loader = self.val_data_loader

                metrics = defaultdict(float)
                epoch_samples = 0

                for i, (data, target) in enumerate(data_loader):

                    # configure input 
                    inputs = tensor2var(data)
                    labels = tensor2var(target)

                    # zero the parameter gradients 
                    self.unet_optimizer.zero_grad()

                    # forward 
                    # track history if only in train 
                    with torch.set_grad_enabled(phase=='train'):
                        outputs = self.unet(inputs)
                        loss = calc_loss(outputs, labels, metrics)

                        # backward + optimzer only if in training phase 
                        if phase == 'train':
                            loss.backward()
                            self.unet_optimizer.step()

                    # statistics 
                    epoch_samples += inputs.size(0)

                self.print_metrics(metrics, epoch_samples, phase)
                epoch_loss = metrics['loss'] / epoch_samples

                # deep copy the model 
                if phase == 'val' and epoch_loss < best_loss:
                    print("saving best model")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(self.unet.state_dict())

                    # save best model checkpoint 
                    torch.save({
                        'epoch': epoch,
                        'state_dict': best_model_wts,
                        'loss': epoch_loss,
                    },
                    os.path.join(self.model_save_path, '{}.pth.tar'.format(epoch))
                    )
                    
                    print('Best val loss: {:4f}'.format(best_loss))
        # end one epoch

            elapsed = time.time() - start_time
            print('Time: {:.0f}m {:0f}s'.format(elapsed // 60, elapsed % 60))

            # log to the tensorboard
            # self.logger.add_scalar('unet_loss', best_loss.item(), epoch)

            # load best model weights 
            self.unet.load_state_dict(best_model_wts)
            
            test_iou = compute_iou(self.unet, self.test_data_loader)
            print(self.model, f"""\tMean IoU of the test images - {np.around(test_iou, 2)*100}%""")

    def build_model(self):

        # self.unet = UNet(n_class=1).cuda()
        self.unet = UNet().cuda()

        # optimizer 
        self.unet_optimizer = torch.optim.Adamax(self.unet.parameters(), lr=self.lr, betas=[self.beta1, self.beta2])

        # lr scheduler 
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.unet_optimizer, step_size=25, gamma=0.1, verbose=True)

        # for orignal gan loss function
        # self.adversarial_loss_sigmoid = nn.BCEWithLogitsLoss()

        # print networks
        print(self.unet)

    def build_tensorboard(self):
        from torch.utils.tensorboard import SummaryWriter
        self.logger = SummaryWriter(self.log_path)

    def save_image_tensorboard(self, images, text, step):
        if step % 100 == 0:
            img_grid = torchvision.utils.make_grid(images, nrow=8)

            self.logger.add_image(text + str(step), img_grid, step)
            self.logger.close()

   
    # metrics
    def print_metrics(self, metrics, epoch_samples, phase):
        outputs = []
        for k in metrics.keys():
            outputs.append(" {}: {:4f}".format(k, metrics[k] / epoch_samples))

        print("{}: {} ".format(phase, ",".join(outputs)))

