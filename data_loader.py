import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import image
from build_vocab import Vocabulary
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):
    """torch.utils.data.DataLoader와 호환되는 Custom COCO Dataset"""
    pass


def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """custom coco dataset을 위한 torch.utils.data.DataLOader를 반환한다."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                        json=json,
                        vocab=vocab,
                        transform=transform)

    # COCO dataset을 위한 데이터셋
    # (images, captions, lengths)를 매 반복마다 반환한다.
    # imgaes: a tensor of shape (batch_size, 3, 224, 224)
    # lengths: 각 caption마다 valid length를 나타내는 list. 길이는 (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers,
                                                collate_fn=collate_fn)
    
    return data_loader