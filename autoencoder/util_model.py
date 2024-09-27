import torch
import torch.nn as nn

# import lightning as L
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.utils.data as data
# import torchvision
# from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms
# from torchvision.datasets import CIFAR10
# from torch.utils.data import TensorDataset, DataLoader
# from torch.utils.data import random_split
# import random
# from util_model import ConvAutoencoder, WeightedBinaryCrossEntropyLoss

class ConvAutoencoder(nn.Module):
    def __init__(self, bottle, inputs = 1):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(inputs, 16, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 8, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(8, 8, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(8, bottle, kernel_size=5, stride=2, padding=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bottle, 8, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Conv2d(8, 8, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output, latent

# Define biased loss function
class WeightedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, pos_weight=1.0, neg_weight=1.0):
        super(WeightedBinaryCrossEntropyLoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, outputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(outputs, targets)
        
        # Apply weights
        pos_weight = torch.where(targets == 1, self.pos_weight, torch.tensor(1.0, device=outputs.device))
        neg_weight = torch.where(targets == 0, self.neg_weight, torch.tensor(1.0, device=outputs.device))

        # Recalculate loss
        weighted_loss = bce_loss * (pos_weight + neg_weight)
        return weighted_loss.mean()
