import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
from torchvision import transforms
from build_vocab import Vovabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image


# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image
    
def main(args):
    # 이미지 전처리
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # 단어 사전을 불러온다.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # 모델을 빌드한다.
    encoder = EncoderCNN(args.embed_size).eval() # eval mode로 설정한다.
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab) args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # 훈련된 모델의 가중치를 불러온다.
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # 이미지를 불러온다.
    image = load_image(args.image, transform)
    image_tensor = image.to(device)

    # 이미지에서 캡션을 생성해낸다.
    feature = encodeR(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy() # (1, max_seq_length) -> (max_seq_length)    

    # 단어의 인덱스를 단어로 변환한다.
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    # 이미지와 생성된 설명을 출력한다.
    print(sentence)
    image = Image.open(args.image)
    plt.imshow(np.asarray(image))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='models/encoder-5-3000.pkl', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-5-3000.pkl', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)