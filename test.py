# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
import numpy as np
import random
from torch.nn.utils.rnn import pad_sequence
from conformer.model import Conformer


char_map_strs = """
 ' 0
 <SPACE> 1
 a 2
 b 3
 c 4
 d 5
 e 6
 f 7
 g 8
 h 9
 i 10
 j 11
 k 12
 l 13
 m 14
 n 15
 o 16
 p 17
 q 18
 r 19
 s 20
 t 21
 u 22
 v 23
 w 24
 x 25
 y 26
 z 27
 """

class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = char_map_strs
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('', ' ')


class ClassificationDataSet(Dataset):
    def __init__(self, data: Dataset):
        """
        convert data sentences into indices b/c torch can't handle strings as input, and
        order word_ids and labels in respectively
        :param data: training, validation, or test data
        :param word_embeddings: get embeddings for pretrained words
        :return:
        """
        self.texts = np.array([None] * len(data))
        self.audios = np.array([None] * len(data))
        text_transform = TextTransform()
        self.data_len = len(data)

        for i in range(self.data_len):
          
          self.texts[i] = np.array(text_transform.text_to_int(data[i]["text"].lower()))
          self.audios[i] = np.array(data[i]["audio"]["array"])

    def __len__(self):
        return self.data_len

    def __getitem__(self, item: int):
        """
        Given an index, return an example from the position.
        :param item:
        :return: `Dict[str, float or int]`: Dictionary of inputs that are used to feed to a model.
        """
        return {'audio': self.audios[item], 'text': self.texts[item]}

def collate_batch(batch):
    # sentence_batch = np.array([row_data["text"] for row_data in batch])
    # target_batch = np.array([row_data["label"] for row_data in batch])

    sentence_batch = [torch.from_numpy(row_data["text"]) for row_data in batch]
    sentence_batch = pad_sequence(sentence_batch, batch_first=True, padding_value=0)

    audio_batch = [torch.from_numpy(row_data["audio"]) for row_data in batch]
    audio_batch = pad_sequence(audio_batch, batch_first=True, padding_value=0)
    audio_batch = audio_batch.unsqueeze(2)
    
    audio_length_batch = torch.Tensor([len(row_data["audio"]) for row_data in batch]).long()

    # NOTE: each audio array len is varied!!!
    # sentence_batch = torch.from_numpy(sentence_batch)
    # target_batch = torch.from_numpy(target_batch)

    return audio_batch, audio_length_batch, sentence_batch


from datasets import load_dataset, load_dataset_builder
ds_builder = load_dataset_builder("librispeech_asr", "clean", "train.clean.100")
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

train_dataset = ClassificationDataSet(ds)
train_dataloader = DataLoader(train_dataset, batch_size=20, collate_fn=collate_batch, shuffle=True)
num_epochs = 10
for epoch in range(num_epochs):

    # training
    for audio_batch, audio_length_batch, sentence_batch in train_dataloader:  # shuffling automatically starts
        print("audio_batch:", audio_batch)
        print("audio_batch size:", audio_batch.size())
        print("sentence_batch:", sentence_batch)
        print("sentence_batch size:", sentence_batch.size())
        Conformer(audio_batch, audio_length_batch)
        break
    
    break