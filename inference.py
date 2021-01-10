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


import skimage.transform
from scipy.misc import imread, imresize


device = torch.device("cpu")

def caption_image(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.
    
    Input:
    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map(word to index mapping)
    :param beam_size: number of sequences to consider at each decode-step
    
    Output:
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    ## Read image and process
    img = imread(image_path)
    
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
        
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)
    
    
    # Encode
    # (1, 3, 256, 256)
    image = image.unsqueeze(0)
    
    #(1, enc_image_size, enc_image_size, encoder_dim)
    #(1, 14, 14, 2048)
    encoder_out = encoder(image)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    # (1, num_pixels, encoder_dim)
    # (1, 196, 2048)
    encoder_out = encoder_out.view(1, -1, encoder_dim)  
    num_pixels = encoder_out.size(1)
    
    # We'll treat the problem as having a batch size of k
    # (k, num_pixels, encoder_dim)
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    # (k, 1)
    seqs = k_prev_words

    # Tensor to store top k sequences scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences alphas; now they're just 1s
    # (k, 1, enc_image_size, enc_image_size)
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, 
    # because sequences are removed from this process once they hit <end>
    while True:
        
        # (s, embed_dim)
        embeddings = decoder.embedding(k_prev_words).squeeze(1)
        
        # (s, encoder_dim), (s, num_pixels)
        awe, alpha = decoder.attention(encoder_out, h)  
        
        # (s, enc_image_size, enc_image_size)
        alpha = alpha.view(-1, enc_image_size, enc_image_size)
        
        # gating scalar, (s, encoder_dim)
        gate = decoder.sigmoid(decoder.f_beta(h))
        awe = gate * awe
        
        # (s, decoder_dim)
        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        
        # (s, vocab_size)
        scores = decoder.fc(h)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # print(top_k_words)
        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)
        # print(seqs[prev_word_inds])
        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        
        ## will be empty if none of them have reached <end>
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        
        ### updating h's and c's for incomplete sequences
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        
        
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 40:
            break
        
        step += 1
    
    # print(complete_seqs)
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas



# def visualize_att(image_path, seq, alphas, rev_word_map, smooth=False):
#     """
#     Visualizes caption with weights at every word.
#     Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

#     :param image_path: path to image
#     :param seq: generated caption
#     :param alphas: attention weights for every time steps
#     :param rev_word_map: reverse word mapping, i.e. ix2word
#     :param smooth: smooth weights?
#     """
#     image = Image.open(image_path)
#     image = image.resize([14 * 14, 14 * 14], Image.LANCZOS)

#     words = [rev_word_map[ind] for ind in seq]
    
    
#     figures = []
    
    
#     for t in range(len(words)):
        
#         fig = plt.figure()
        
#         if t > 50:
#             break
        
#         #plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)
#         fig.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
#         plt.imshow(image)
        
#         current_alpha = alphas[t, :]
        
#         if smooth:
#             alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=14, sigma=8)
#         else:
#             alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 14, 14 * 14])
        
#         if t == 0:
#             plt.imshow(alpha, alpha=0)
#         else:
#             plt.imshow(alpha, alpha=0.8)
        
#         plt.set_cmap(cm.Greys_r)
#         plt.axis('off')
        
#         figures.append(fig)
#         #plt.savefig("horse_riding/"+words[t]+ str(t)+'.png', bbox_inches = 'tight', pad_inches = 0)
        
        
    
#     plt.show()
    