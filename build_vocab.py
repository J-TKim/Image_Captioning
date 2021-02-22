import nltk
import pickle
import argparse

from collections import Counter
from pycocotools.coco import COCO


class Vovabulary(object):
    """Simple vocabulary warpper"""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold):
    """간단한 vocabulary warpper를 만든다."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print('[{}/{}] Tokenized the captions.'.format(i+1, len(ids)))

    # 단어가 threshold보다 적게 나오면, 그 단어는 버려진다.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens
    vocab = Vovabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # 단어를 vocabulary에 추가한다.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(args):
    vocab = build_vocab(json=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, default='data/annotaions/captions_train2014.json', help='path for train annotataion file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, help='minimun word count threshold')
    args = parser.parse_args()
    main(args)