# coding: utf-8
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import HTML

import torchtext
import pandas as pd
import datetime

from utils.dataloader import Preprocessing
from utils.transformer import TransformerClassification
from sklearn.metrics import roc_auc_score

preprocessing = Preprocessing()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs, label, device="cpu"):

    print("using device: ", device)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     net = nn.DataParallel(net)

    net.to(device)

    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            for batch in (dataloaders_dict[phase]):
                inputs = batch.Text[0].to(device)
                labels = getattr(batch, label)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    input_pad = 1
                    input_mask = (inputs != input_pad)

                    outputs, _, _ = net(inputs, input_mask)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # validation mode
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += roc_auc_score(labels.data, preds)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase])

            print('Epoch {}/{} | {:^5} |  Loss: {:.4f} ROC_AUC: {:.4f}'.format(epoch + 1, num_epochs,
                                                                           phase, epoch_loss, epoch_acc))

    return net


def main():
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    path = "./data/"
    train_file = "train.csv"
    test_file = "test.csv"
    vector_list = "./data/wiki-news-300d-1M.vec"
    max_sequence_length = 900
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    train_dl, val_dl, test_dl, TEXT = preprocessing.get_data(path=path, train_file=train_file, test_file=test_file,
                                                             vectors=vector_list, max_length=max_sequence_length,
                                                             batch_size=3000)

    dataloaders_dict = {"train": train_dl, "val": val_dl}

    # define output dataframe
    sample = pd.read_csv("./data/sample_submission.csv")

    for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        net = TransformerClassification(
            text_embedding_vectors=TEXT.vocab.vectors, d_model=300, max_seq_len=max_sequence_length, output_dim=2, device=device)

        net.train()

        net.net3_1.apply(weights_init)
        net.net3_2.apply(weights_init)

        print('done setup network with {}'.format(label))

        criterion = nn.CrossEntropyLoss()

        learning_rate = 2e-5
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

        num_epochs = 15
        net_trained = train_model(net, dataloaders_dict,
                                  criterion, optimizer, num_epochs=num_epochs, label=label, device=device)

        # load net if weight avaiable
        # net_trained = torch.load("net_trained.weights", map_location=device)

        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     net_trained = nn.DataParallel(net_trained)

        # net_trainedを保存
        torch.save(net_trained, "net_trained_{}.weights".format(label))

        net_trained.eval()
        net_trained.to(device)

        epoch_corrects = 0

        predicts = []

        for batch in (test_dl):

            inputs = batch.Text[0].to(device)

            with torch.set_grad_enabled(False):
                input_pad = 1
                input_mask = (inputs != input_pad)

                outputs, _, _ = net_trained(inputs, input_mask)
                _, preds = torch.max(outputs, 1)

                preds = preds.cpu()
                preds = preds.numpy().tolist()

                predicts += preds

        sample[label] = predicts

    # save predictions
    sample.to_csv("submission_{}.csv".format(
        datetime.datetime.now().date()), index=False)


if __name__ == '__main__':
    main()
