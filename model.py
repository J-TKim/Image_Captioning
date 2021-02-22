import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        '''pretrained resnet152를 가져오고, 마지막 fc layer를 바꾼다.'''
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1] # 마지막 layer 빼고 불러온다.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """인풋 이미지로부터 특징을 담은 벡터들을 추출한다."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """하이퍼파라미터를 설정하고, layer를 만든다."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.max_seq_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """EncoderCNN에서 만든 feature vectors를 decode하여 captions를 만든다."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embedding), 1)
        packed = pack_padded_sequence(embedding, lengths, batch_first=True)
        hidden, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sampled(self, features, states=None):
        """greedy 서치를 이용해서 주어진 image features로 captions를 만든다."""
        sampeld_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hidden.squeeze(1))
            _, predicted = output.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = input.unsqueeze(1)
        sampled_ids = torch.stack(sampeld_ids, 1)
        return sampled_ids