import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
#from matplotlib import pyplot as plt
import os

from dataset.dataset_class import PreprocessedVidDataSet
from dataset.video_extraction_conversion import *
from loss.loss_discriminator import *
from loss.loss_generator import *
from network.blocks import *
from network.model import *


# Constants
K = 16  # K-shot
path_to_chkpt = './model_weights.tar'
path_to_backup = './backup_model_weights.tar'

device = torch.device("cuda:0")
cpu = torch.device("cpu")

# Data
dataset = PreprocessedVidDataSet(K=K,
                                 path_to_data='../../data/voxceleb2/dev-32',
                                 device=device)
dataLoader = DataLoader(dataset, batch_size=2, shuffle=True)

G = Generator(224).to(device)
E = Embedder(224).to(device)
D = Discriminator(dataset.__len__()).to(device)

G.train()
E.train()
D.train()

optimizerG = optim.Adam(params = G.parameters(), lr=5e-5)
optimizerE = optim.Adam(params = E.parameters(), lr=5e-5)
optimizerD = optim.Adam(params = D.parameters(), lr=2e-4)


# Training init
epochCurrent = epoch = i_batch = 0
lossesG = []
lossesD = []
i_batch_current = 0

num_epochs = 750

# initiate checkpoint if doesn't exist
if not os.path.isfile(path_to_chkpt):
  print('Initiating new checkpoint...')
  torch.save({
          'epoch': epoch,
          'lossesG': lossesG,
          'lossesD': lossesD,
          'E_state_dict': E.state_dict(),
          'G_state_dict': G.state_dict(),
          'D_state_dict': D.state_dict(),
          'optimizerG_state_dict': optimizerG.state_dict(),
          'optimizerE_state_dict': optimizerE.state_dict(),
          'optimizerD_state_dict': optimizerD.state_dict(),
          'num_vid': dataset.__len__(),
          'i_batch': i_batch
          }, path_to_chkpt)
  print('...Done')
