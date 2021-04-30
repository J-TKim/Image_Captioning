import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from konlpy.tag import Okt  
okt=Okt()

from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):
    """torch.utils.data.DataLoader와 호환되는 Custom COCO Dataset"""
    def __init__(self, root, json, vocab, en2ko, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image 경로
            json: coco annotation 파일 경로
            vocab: vocabulary warpper
            transform: image transformer
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.en2ko = en2ko
        self.transform = transform

    def __getitem__(self, index):
        """data 쌍을 반환한다. (image and caption)"""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        if self.en2ko:
            caption = self.en2ko[caption]
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Caption(String)을 word ids로 변환한다
        if self.en2ko:
            tokens = okt.morphs(caption)
        else:
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)



def collate_fn(data):
    """ (image, caption) 형식이 모여있는 list로부터 mini-batch tensor들을 생성해낸다.

    기본적으로 caption을 병합하는것은 지원하지 않으므로
    만들어져있는 collate_fn을 쓰는 것 대신 custom해서 만든 collate_fn을 만들어 사용해야 한다.

    Args:
        data: tuple의 list (image, caption)
            - image: torch tensor of shape (3, 256, 256)
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256)
        targets: torch tensor of shape (batch_size, padded_length)
        lengths: list; valid length for each padded caption
    """
    # caption의 길이에 따라 list 정렬 (reverse=True : 내림차순 정렬)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # 이미지 병합 (3D tensor의 tuple에서 4D tensor의 tuple로 변환)
    images = torch.stack(images, 0)

    # captions 병합 (1D tensor의 tuple에서 2D tensor의 tuple로 변환)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers, en2ko):
    """custom coco dataset을 위한 torch.utils.data.DataLOader를 반환한다."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                        json=json,
                        vocab=vocab,
                        en2ko=en2ko,
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