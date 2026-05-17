import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from numpy.random import randint
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda:0")



def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
 
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def generate_missing_modalities(mask, text_features, audio_features, video_features):
    # "Mask should have shape [batchsize, 3]"
    assert mask.shape[1] == 3 
    assert text_features.shape[0] == mask.shape[0]
    assert audio_features.shape[0] == mask.shape[0]
    assert video_features.shape[0] == mask.shape[0]
    zero_vector_text = torch.zeros_like(text_features)
    zero_vector_audio = torch.zeros_like(audio_features)
    zero_vector_video = torch.zeros_like(video_features)
    mask = mask.bool()
    
    masked_text_features = torch.where(mask[:, 0].unsqueeze(1).unsqueeze(2), text_features, zero_vector_text)
    masked_audio_features = torch.where(mask[:, 1].unsqueeze(1).unsqueeze(2), audio_features, zero_vector_audio)
    masked_video_features = torch.where(mask[:, 2].unsqueeze(1).unsqueeze(2), video_features, zero_vector_video)
    return masked_text_features, masked_audio_features, masked_video_features



def generate_fixed_missing_matrix(M, fixed_protocol):
    """
    Generate a mask matrix with shape [M, 3], based on the fixed_protocol.
    fixed_protocol is a list of length 3, indicating the presence of text, audio, and video modalities.
    """
    # Ensure the fixed_protocol is of length 3
    assert fixed_protocol in ['0', '1', '2', '3', '4', '5', '6']
    # if len(fixed_protocol) != 3:
    #     raise ValueError("fixed_protocol must be a list of length 3")
    if fixed_protocol == '0':
        modalities = [1,0,0]
    elif fixed_protocol == '1':
        modalities = [0,1,0]
    elif fixed_protocol == '2':
        modalities = [0,0,1]
    elif fixed_protocol == '3':
        modalities = [1,1,0]
    elif fixed_protocol == '4':
        modalities = [1,0,1]
    elif fixed_protocol == '5':
        modalities = [0,1,1]
    else:
        modalities = [1,1,1]
    
    # Create the matrix by repeating the fixed_protocol M times
    matrix = np.tile(modalities, (M, 1))
    return matrix
    

def generate_missing_matrix(M, view_num=3, missing_rate=0.5):
    """
    generate a mask matrix whith shape [M, view_num], and keep that at least one modality exist in each sample
    missing rate is the missing modalities in all modalities
    """
    one_rate = 1 - missing_rate
    
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(M, 1))).toarray()
        return view_preserve
    
    if one_rate == 1: 
        matrix = randint(1, 2, size=(M, view_num))
        return matrix
    
    alldata_len = max(M, 32)
    error = 1
    while error >= 0.005:
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        
        one_num = view_num * alldata_len * one_rate - alldata_len
        ratio = one_num / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int32)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int32))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int32)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int32)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)
    
    matrix = matrix[:M, :]
    return matrix





def modality_drop(modal1, modal2, modal3, p, args):
    # The code is adapted from MMANet: https://github.com/shicaiwei123/MMANet-CVPR2023/blob/main/classification/lib/model_arch.py
    modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    index_list = [x for x in range(7)]
    # for training with uncertain conditions
    # save ps for uncertain modalities reweighting
    if p == [0, 0, 0]:
        p = []
        prob = np.array((1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7))
        for i in range(modal1.shape[0]):# bsz
            index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
            p.append(modality_combination[index])# store each missing conditions
            
        p = np.array(p)
        p = torch.from_numpy(p)# adds the missing conditions into the list
        p1 = torch.unsqueeze(p, 2)
        p1 = torch.unsqueeze(p1, 3)
    else:
        p = p
        p = [p * modal1.shape[0]]
        p = np.array(p).reshape(modal1.shape[0], 3)# for 3 modalities
        p = torch.from_numpy(p)
        p1 = torch.unsqueeze(p, 2)
        p1 = torch.unsqueeze(p1, 3)
    p1 = p1.float().to(modal1.device)#.to(DEVICE)
    p = p.float().to(modal1.device)#.to(DEVICE)

    modal1 = modal1 * p1[:, 0].to(modal1.device)
    modal2 = modal2 * p1[:, 1].to(modal2.device)
    modal3 = modal3 * p1[:, 2].to(modal3.device)

    return modal1, modal2, modal3, p


class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2).to(DEVICE)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0).to(DEVICE)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]).to(DEVICE))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY      


