import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import json

from data_loader import get_loader
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    if args.ko:
        with open(args.ko, 'r') as f:
            en2ko = json.load(f)
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))
    ])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab,
                            transform, args.batch_size,
                            shuffle=True, num_workers=args.num_workers,
                            en2ko=en2ko)

    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the model
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):

            # mini batch dataset을 만든다.
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # 모델 학습
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # 훈련 상황을 출력한다.
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexirt: {:5.4f}'.format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))

            # model의 checkpoint를 저장한다.
            if (i+1) % args.save_step == 0:
                torch.save(deocder.state_dict(), os.path.join(
                    args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)
                ))
                torch.save(enocder.state_dict(), os.path.join(
                    args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)
                ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='훈련된 모델을 저장할 경로')
    parser.add_argument('--crop_size', type=int, default=224, help='랜덤으로 crop할 이미지의 사이즈')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='vacabulary warpper의 경로')
    parser.add_argument('--image_dir', type=str, default='data/resized2014', help='resized images의 경로')
    parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json', help='train annotation json file의 경로')
    parser.add_argument('--ko', type=str, default=None, help='ko dataset 경로')
    parser.add_argument('--log_step', type=int, default=10, help='log info를 출력하기 위한 step size')
    parser.add_argument('--save_step', type=int, default=10, help='trained models을 저장하기 위한 step size')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='word embedding vectors 차원')
    parser.add_argument('--hidden_size', type=int, default=512, help='LSTM의 hidden state의 차원')
    parser.add_argument('--num_layers', type=int, default=1, help='LSTM의 layer 수')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)