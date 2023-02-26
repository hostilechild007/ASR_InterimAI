# -*- coding: utf-8 -*-
import os
import pdb
import torch
import torch.optim as optim
import torchaudio
from model import *
from comet_ml import Experiment
from torch.utils import data
import torchvision.transforms as T

# train_dataset = torchaudio.datasets.LIBRISPEECH("./", url="train-clean-100", download=True)
# test_dataset = torchaudio.datasets.LIBRISPEECH("./", url="test-clean", download=True)


#torchaudio.transforms.FrequencyMasking()
#torchaudio.transforms.TimeMasking()

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


train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35)
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

text_transform = TextTransform()

def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        
        # spec_len, spec_dim = spec.size()
        # image_transform = T.Resize((400, spec_dim))
        # print("spec:", spec)
        # print("spec size:", spec.size())
        # spectrograms.append(image_transform(spec))
        
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1)#.transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, Tensor(input_lengths).long(), label_lengths

def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes, targets

# initialize experiment object
#experiment = Experiment(api_key="key", project_name="ASR_group_prj")
#experiment.set_name("Exp_01")

class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment):
    model.train()
    data_len = len(train_loader.dataset)
    with experiment.train():
        for batch_idx, _data in enumerate(train_loader):
            spectrograms, labels, input_lengths, label_lengths = _data 
            print("spectrograms size:", spectrograms.size())
            print("spectrograms:", spectrograms)
            print("labels size:", labels.size())
            print("labels:", labels)
            print("input_lengths size:", input_lengths.size())
            print("input_lengths:", Tensor(input_lengths))
            print("label_lengths size:", len(label_lengths))
            print("label_lengths:", label_lengths)
            print()
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(spectrograms, input_lengths)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            loss.backward()

            #experiment.log_metric('loss', loss.item(), step=iter_meter.get())
            #experiment.log_metric('learning_rate', scheduler.get_lr(), step=iter_meter.get())

            optimizer.step()
            scheduler.step()
            iter_meter.step()
            if batch_idx % 100 == 0 or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(spectrograms), data_len,
                    100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, criterion, epoch, iter_meter, experiment):
    print('\n evaluating…')
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with experiment.test():
        with torch.no_grad():
            for I, _data in enumerate(test_loader):
                spectrograms, labels, input_lengths, label_lengths = _data 

                spectrograms, labels = spectrograms.to(device), labels.to(device)

                output = model(spectrograms)  # (batch, time, n_class)
                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1) # (time, batch, n_class)

                loss = criterion(output, labels, input_lengths, label_lengths)
                test_loss += loss.item() / len(test_loader)

                decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
                for j in range(len(decoded_preds)):
                    test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                    test_wer.append(wer(decoded_targets[j], decoded_preds[j]))


    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)
    #experiment.log_metric('test_loss', test_loss, step=iter_meter.get())
    #experiment.log_metric('cer', avg_cer, step=iter_meter.get())
    #experiment.log_metric('wer', avg_wer, step=iter_meter.get())

    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))

def main(learning_rate=5e-4, batch_size=5, epochs=10,
        train_url="train-clean-100", test_url="test-clean",
        experiment=Experiment(api_key='dummy_key', disabled=True), train_mode=True):

    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 29,
        "n_feats": 128,
        "stride": 2,
        "dropout": 0.1,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }

    experiment.log_parameters(hparams)

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    if not os.path.isdir("./data"):
        os.makedirs("./data")

    train_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=train_url, download=True)
    # test_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=test_url, download=True)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=True,
                                collate_fn=lambda x: data_processing(x, 'train'),
                                **kwargs)
    # test_loader = data.DataLoader(dataset=test_dataset,
    #                             batch_size=hparams['batch_size'],
    #                             shuffle=False,
    #                             collate_fn=lambda x: data_processing(x, 'valid'),
    #                             **kwargs)

    ###TODO
    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
        ).to(device)

    print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=28).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'], 
                                            steps_per_epoch=int(len(train_loader)),
                                            epochs=hparams['epochs'],
                                            anneal_strategy='linear')

    iter_meter = IterMeter()
    
    if train_mode:
        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment)
    else:    
        # test(model, device, test_loader, criterion, epoch, iter_meter, experiment)  # ORIGINAL
        test(model, device, test_loader, criterion, 0, iter_meter, experiment)

if __name__=="__main__":
    main(learning_rate=5e-4, batch_size=50, epochs=20,
        train_url="train-clean-100", test_url="test-clean",
        experiment=Experiment(api_key='dummy_key', disabled=True), train_mode=True)
    main(learning_rate=5e-4, batch_size=50, epochs=20,
        train_url="train-clean-100", test_url="test-clean",
        experiment=Experiment(api_key='dummy_key', disabled=True), train_mode=False)