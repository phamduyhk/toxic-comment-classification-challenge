import datetime

import pandas as pd
import os
import math

import torch
from torch.backends import cudnn
from torch.nn import BCEWithLogitsLoss, BCELoss, MultiLabelSoftMarginLoss,CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, XLNetTokenizer, XLNetModel, XLNetLMHeadModel, XLNetConfig
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import sys


def main():
    print("GPU Available: {}".format(torch.cuda.is_available()))
    n_gpu = torch.cuda.device_count()
    print("Number of GPU Available: {}".format(n_gpu))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device: {}".format(device))

    num_embeddings = 256
    # Select a batch size for training
    batch_size = 64

    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")

    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    if len(sys.argv)<2:
        print("Example: python3 predict_xlnet.py <label>")
        sys.exit()

    label = sys.argv[1]

    if not os.path.exists("./submission"):
        os.mkdir("./submission")

    sample = pd.read_csv("./data/sample_submission.csv")

    if label:
        print("Label: {}".format(label))
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
        train_text_list = train["comment_text"].values
        test_text_list = test["comment_text"].values

        train_input_ids = tokenize_inputs(train_text_list, tokenizer, num_embeddings=num_embeddings)
        test_input_ids = tokenize_inputs(test_text_list, tokenizer, num_embeddings=num_embeddings)

        train_attention_masks = create_attn_masks(train_input_ids)
        test_attention_masks = create_attn_masks(test_input_ids)

        # add input ids and attention masks to the dataframe
        train["features"] = train_input_ids.tolist()
        train["masks"] = train_attention_masks

        test["features"] = test_input_ids.tolist()
        test["masks"] = test_attention_masks


        X_train = train["features"].values.tolist()

        Y_train = y_split(train, label)
        print("y true: ",Y_train[:5])

        num_labels = 2

        num_epochs = 2

        # load model: xlnet_label_3ep_weight.bin (trained on 2.4.2020 | 4label score: 0.84)
        model_save_path = "xlnet_{}_{}ep_weights.bin".format(label, 3)

        model, epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist = load_model(model_save_path)
        print(model)

        # validation
        train_predicts = generate_predictions(model, train, num_labels, device=device, batch_size=batch_size)
        score = roc_auc_score_FIXED(Y_train, train_predicts)
        print("Label: {}, ROC_AUC: {}".format(label, score))

        predicts = generate_predictions(model, test, num_labels, device=device, batch_size=batch_size)
        
        sample[label] = predicts
        output_filename = "submission_XLNET_{}_{}_{}ep.csv".format(datetime.datetime.now().date(), label, num_epochs)
        sample.to_csv(output_filename, index=False)
        print("Label: {}, Output: {}".format(label, output_filename))


def y_split(data, label):
    y = data[label]
    y = np.array(y)
    # y = y.reshape(y.shape[0], 1)
    return y


def tokenize_inputs(text_list, tokenizer, num_embeddings=512):
    """
    Tokenizes the input text input into ids. Appends the appropriate special
    characters to the end of the text to denote end of sentence. Truncate or pad
    the appropriate sequence length.
    """
    # tokenize the text, then truncate sequence to the desired length minus 2 for
    # the 2 special characters
    tokenized_texts = list(map(lambda t: tokenizer.tokenize(t)[:num_embeddings - 2], text_list))
    # convert tokenized text into numeric ids for the appropriate LM
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # append special token "<s>" and </s> to end of sentence
    input_ids = [tokenizer.build_inputs_with_special_tokens(x) for x in input_ids]
    # pad sequences
    input_ids = pad_sequences(input_ids, maxlen=num_embeddings, dtype="long", truncating="post", padding="post")
    return input_ids


def create_attn_masks(input_ids):
    """
    Create attention masks to tell model whether attention should be applied to
    the input id tokens. Do not want to perform attention on padding tokens.
    """
    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks


def roc_auc_score_FIXED(y_true, y_pred):
    try:
        score = roc_auc_score(y_true, y_pred)
    except ValueError:
        score = accuracy_score(y_true, np.rint(y_pred))
    return score


def load_model(save_path):
    """
    Load the model from the path directory provided
    """
    checkpoint = torch.load(save_path)
    model_state_dict = checkpoint['state_dict']
    model = XLNetForMultiLabelSequenceClassification(num_labels=model_state_dict["classifier.weight"].size()[0])
    model.load_state_dict(model_state_dict)

    epochs = checkpoint["epochs"]
    lowest_eval_loss = checkpoint["lowest_eval_loss"]
    train_loss_hist = checkpoint["train_loss_hist"]
    valid_loss_hist = checkpoint["valid_loss_hist"]

    return model, epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist


def generate_predictions(model, df, num_labels, device="cpu", batch_size=32):
    num_iter = math.ceil(df.shape[0] / batch_size)

    predicts =[]

    model.to(device)
    model.eval()

    for i in range(num_iter):
        df_subset = df.iloc[i * batch_size:(i + 1) * batch_size, :]
        X = df_subset["features"].values.tolist()
        masks = df_subset["masks"].values.tolist()
        X = torch.tensor(X)
        masks = torch.tensor(masks, dtype=torch.long)
        X = X.to(device)
        masks = masks.to(device)
        with torch.no_grad():
            logits = model(input_ids=X, attention_mask=masks)
            _, preds = torch.max(logits, 1)
            preds = preds.cpu()
            preds = preds.numpy().tolist()

            predicts += preds

    return predicts


if __name__ == "__main__":
    main()
