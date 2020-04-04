# coding: utf-8
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score

import torchtext
import pandas as pd
import datetime
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), './utils')))

from EarlyStopping import EarlyStopping
from transformer import TransformerClassification
from dataloader import Preprocessing


preprocessing = Preprocessing()


def main(train_mode=True, load_trained=False, early_stop=False):
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    path = "./data/"
    train_file = "train.csv"
    test_file = "test.csv"
    vector_list = "./data/wiki-news-300d-1M.vec"
    max_sequence_length = 512
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dl, val_dl, test_dl, TEXT = preprocessing.get_data(path=path, train_file=train_file, test_file=test_file,
                                                             vectors=vector_list, max_length=max_sequence_length,
                                                             batch_size=1500)

    dataloaders_dict = {"train": train_dl, "val": val_dl}

    # define output dataframe
    sample = pd.read_csv("./data/sample_submission.csv")

    for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:

        if load_trained is True:
            net = torch.load("net_trained_transformer_{}.weights".format(label),
                         map_location=device)
        else:
            net = TransformerClassification(
            text_embedding_vectors=TEXT.vocab.vectors, d_model=300, max_seq_len=max_sequence_length, output_dim=2, device=device)

       
        net.train()

        net.net3_1.apply(weights_init)
        net.net3_2.apply(weights_init)

        print('done setup network with {}'.format(label))

        criterion = nn.CrossEntropyLoss()

        learning_rate = 2e-5
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

        num_epochs = 10
        if train_mode:
            net_trained = train_model(net, dataloaders_dict,
                                  criterion, optimizer, num_epochs=num_epochs, label=label, device=device, early_stop=early_stop)

            # net_trainedを保存
            torch.save(net_trained, "net_trained_transformer_{}.weights".format(label))

        else:
            net_trained = net
            

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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def roc_auc_score_FIXED(y_true, y_pred):
    try:
        score = roc_auc_score(y_true, y_pred)
    except ValueError:
        score = accuracy_score(y_true, np.rint(y_pred))
    return score



def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs, label, device="cpu", early_stop=False):

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
            epoch_metrics = 0

            for batch in (dataloaders_dict[phase]):
                inputs = batch.Text[0].to(device)
                labels = getattr(batch, label)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    input_pad = 1
                    input_mask = (inputs != input_pad)

                    outputs, _, _ = net(inputs, input_mask)
                    # print("outputs size : {}".format(outputs.size()))
                    # print("labels size : {}".format(labels.size()))
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    preds = preds.cpu()
                    y_true = labels.cpu()
                    epoch_metrics += roc_auc_score_FIXED(y_true, preds)
                    epoch_loss += loss.item() * inputs.size(0)
            

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_eval = epoch_metrics / len(dataloaders_dict[phase])

            print('Label: {} Epoch {}/{} | {:^5} |  Loss: {:.4f} ROC_AUC: {:.4f}'.format(label, epoch + 1, num_epochs,
                                                                               phase, epoch_loss, epoch_eval))

    return net


if __name__ == '__main__':
    main()