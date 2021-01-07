import os
import time
import h5py
import json
from PIL import Image

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

import torch.optim
import torch.nn.functional as F

from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from tqdm.notebook import tqdm
import matplotlib.cm as cm

import torch.backends.cudnn as cudnn
import torch.utils.data

import cv2
import skimage.transform
from scipy.misc import imread, imresize

from nltk.translate.bleu_score import corpus_bleu

from utils import *

# sets device for model and PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")