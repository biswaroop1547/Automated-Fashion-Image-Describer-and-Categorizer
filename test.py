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
from model import *
from inference import *

if __name__ == '__main__':

    # sets device for model and PyTorch tensors
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # text_df = pd.read_excel("dataset.xlsx")
    # print(text_df.tail())


    import wget
    # Load model

    checkpoint = torch.load("BEST_checkpoint_fashion_captioning_1_cap_per_img_2_min_word_freq.pth.tar", map_location="cpu")
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix) {"word1":0, "word2":1}
    with open("WORDMAP_fashion_captioning_1_cap_per_img_2_min_word_freq.json", 'r') as j:
        word_map = json.load(j)

    ### idx to word {0 : "word1", 1:"word2"}
    rev_word_map = {v: k for k, v in word_map.items()}
    IMG_URL = "https://assets.ajio.com/medias/sys_master/root/20200820/cBfP/5f3d8a26f997dd2277a1e849/rozveh_maroon_printed_gown_dress.jpg"
    # image = "images/pic_474.jpg"
    image = wget.download(IMG_URL, out="testing_image.jpg")


    ### alphas is nothing but attention weights
    seq, alphas = caption_image(encoder, decoder, image, word_map, 5)
    alphas = torch.FloatTensor(alphas)

    pred_caption = []

    for word in seq:
        pred_caption.append(rev_word_map[word])

    print(pred_caption)

    # image = Image.open(image)
    # image = image.resize([14 * 14, 14 * 14], Image.LANCZOS)

    # fig = plt.figure()
    # plt.imshow(image)



    # Visualize caption and attention of best sequence
    # visualize_att(image,#original image
    #               seq, #generated sequence
    #               alphas, #attention weights for every time steps
    #               rev_word_map # idx to word mapping
    #              )