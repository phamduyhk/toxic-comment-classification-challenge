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

from dataloader import Preprocessing
from transformer import TransformerClassification
from EarlyStopping import EarlyStopping


preprocessing = Preprocessing()
es = EarlyStopping(patience=20)


def main(train_mode, load_trained=True):
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    path = "./data/"
    train_file = "train.csv"
    test_file = "test.csv"
    vector_list = "./data/wiki-news-300d-1M.vec"
    max_sequence_length = 900
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dl, val_dl, test_dl, TEXT = preprocessing.get_data(path=path, train_file=train_file, test_file=test_file,
                                                             vectors=vector_list, max_length=max_sequence_length,
                                                             batch_size=128)

    dataloaders_dict = {"train": train_dl, "val": val_dl}

    # define output dataframe
    sample = pd.read_csv("./data/sample_submission.csv")

    label_cols = ['toxic', 'severe_toxic', 'obscene',
                  'threat', 'insult', 'identity_hate']

    num_labels = len(label_cols)

    if load_trained is True:
        net = torch.load("net_trained_transformer.weights",
                         map_location=device)
    else:
        net = TransformerClassification(
            text_embedding_vectors=TEXT.vocab.vectors, d_model=300, max_seq_len=max_sequence_length,
            output_dim=num_labels,
            device=device)

    net.train()

    net.net3_1.apply(weights_init)
    net.net3_2.apply(weights_init)

    print('done setup network')

    print("running mode: {}".format("training" if train_mode else "predict"))

    if train_mode:
        # Define loss function
        criterion = nn.BCEWithLogitsLoss()

        """or"""
        #criterion = nn.MultiLabelSoftMarginLoss()

        learning_rate = 2e-3
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

        num_epochs = 50
        net_trained = train_model(net, dataloaders_dict,
                                criterion, optimizer, num_epochs=num_epochs, label_cols=label_cols, device=device)

        # net_trainedを保存
        torch.save(net_trained, "net_trained_transformer.weights")

    else:
        net_trained = net

    net_trained.eval()
    net_trained.to(device)

    pred_probs = np.array([]).reshape(0, num_labels)

    for batch in (test_dl):
        inputs = batch.Text[0].to(device)

        with torch.set_grad_enabled(False):
            input_pad = 1
            input_mask = (inputs != input_pad)

            outputs, _, _ = net_trained(inputs, input_mask)
            preds = (outputs.sigmoid() > 0.5) * 1
            preds = preds.cpu()
            pred_probs = np.vstack([pred_probs, preds])

    print(pred_probs)
    predicts = np.round(pred_probs)
    df = pd.DataFrame(predicts)
    df.to_csv("dummy_predict_transformer.csv", index=False)
    predicts = predicts.reshape(predicts.shape[1], predicts.shape[0])
    for index, label in enumerate(label_cols):
        sample[label] = predicts[index]

    # save predictions
    if not os.path.exists("./submission"):
        os.mkdir("./submission")
    sample.to_csv("./submission/submission_Transformer_{}_{}ep.csv".format(
        datetime.datetime.now().date(), num_epochs), index=False)


def roc_auc_score_FIXED(y_true, y_pred):
    try:
        score = roc_auc_score(y_true, y_pred)
    except ValueError:
        score = accuracy_score(y_true, np.rint(y_pred))
    return score


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs, label_cols, device="cpu"):
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
                y_true = torch.cat([getattr(batch, feat).unsqueeze(1)
                                    for feat in label_cols], dim=1).float()
                y_true = y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    input_pad = 1
                    input_mask = (inputs != input_pad)

                    outputs, _, _ = net(inputs, input_mask)
                    loss = criterion(outputs, y_true)

                    preds = (outputs.sigmoid() > 0.5) * 1

                    # training mode
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # validation mode
                    epoch_loss += loss.item() * inputs.size(0)
                    y_true = y_true.data.cpu()
                    preds = preds.cpu()
                    # print("y_true {}, y_pred {}".format(y_true, preds))
                    epoch_metrics += roc_auc_score_FIXED(y_true, preds)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_eval = epoch_metrics / len(dataloaders_dict[phase])

            print('Epoch {}/{} | {:^5} |  Loss: {:.4f} ROC_AUC: {:.4f}'.format(epoch + 1, num_epochs,
                                                                               phase, epoch_loss, epoch_eval))

        if es.step(torch.tensor(epoch_eval)):
            print("Early stoped at epoch: {}".format(num_epochs))
            break  # early stop criterion is met, we can stop now

    return net


if __name__ == '__main__':
    main(train_mode=False)
