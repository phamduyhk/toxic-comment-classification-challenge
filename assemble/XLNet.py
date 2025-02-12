import datetime

import pandas as pd
import os
import sys
import math

import torch
from torch.backends import cudnn
from torch.nn import BCEWithLogitsLoss, BCELoss, MultiLabelSoftMarginLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, XLNetTokenizer, XLNetModel, XLNetLMHeadModel, XLNetConfig
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score


sigmoid = torch.nn.Sigmoid()



def main():
    print("GPU Available: {}".format(torch.cuda.is_available()))
    n_gpu = torch.cuda.device_count()
    print("Number of GPU Available: {}".format(n_gpu))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device: {}".format(device))

    num_embeddings = 256
    # Select a batch size for training
    batch_size = 16
    """
    mode: train
      or  predict
    """
    mode = "train"

    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")

    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

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

    # train valid split
    train, valid = train_test_split(train, test_size=0.2, random_state=42)

    X_train = train["features"].values.tolist()
    X_valid = valid["features"].values.tolist()

    Y_train = y_split(train, label_cols)
    Y_valid = y_split(valid, label_cols)

    train_masks = train["masks"].values.tolist()
    valid_masks = valid["masks"].values.tolist()

    # Convert all of our input ids and attention masks into
    # torch tensors, the required datatype
    X_train = torch.tensor(X_train)
    X_valid = torch.tensor(X_valid)

    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    Y_valid = torch.tensor(Y_valid, dtype=torch.float32)

    train_masks = torch.tensor(train_masks, dtype=torch.long)
    valid_masks = torch.tensor(valid_masks, dtype=torch.long)

    # Create an iterator of our data with torch DataLoader. This helps save on
    # memory during training because, unlike a for loop,
    # with an iterator the entire dataset does not need to be loaded into memory

    train_data = TensorDataset(X_train, train_masks, Y_train)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=batch_size)

    validation_data = TensorDataset(X_valid, valid_masks, Y_valid)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data,
                                       sampler=validation_sampler,
                                       batch_size=batch_size)

    model = XLNetForMultiLabelSequenceClassification(num_labels=len(Y_train[0]), device=device)

    # Freeze pretrained xlnet parameters
    model.unfreeze_xlnet_decoder()

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01, correct_bias=False)

    num_epochs = 3
    model_save_path = "xlnet_weights.bin"


    if mode is "train":
        model, train_loss_set, valid_loss_set = train_model(model, num_epochs=num_epochs, optimizer=optimizer,
                                                        train_dataloader=train_dataloader,
                                                        valid_dataloader=validation_dataloader,
                                                        model_save_path=model_save_path,
                                                        device=device
                                                        )
    else:
        # load model
        model, epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist = load_model(model_save_path)

    num_labels = len(label_cols)

    sample = pd.read_csv("../data/sample_submission.csv")
    raw_data, pred_probs = generate_predictions(model, test, num_labels, device=device, batch_size=batch_size)
    predicts = (pred_probs>0.5) *1 
    df = pd.DataFrame(raw_data)
    df.to_csv("raw_pred_probs_predict_xlnet.csv", index=False)
    predicts = predicts.reshape(predicts.shape[1], predicts.shape[0])
    for index, label in enumerate(label_cols):
        sample[label] = predicts[index]
    # save output
    if not os.path.exists("./submission"):
        os.mkdir("./submission")
    now = datetime.datetime.now()
    sample.to_csv("submission_XLNET_{}_{}ep.csv".format(now.timestamp(), num_epochs), index=False)


def y_split(data, label_cols):
    y = []
    for label in label_cols:
        y.append(data[label])
    y = np.array(y)
    y = y.reshape(y.shape[1], y.shape[0])
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


class XLNetForMultiLabelSequenceClassification(torch.nn.Module):

    def __init__(self, num_labels=2, device="cpu"):
        super(XLNetForMultiLabelSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.device = device
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.classifier = torch.nn.Linear(768, num_labels)

        torch.nn.init.xavier_normal_(self.classifier.weight)


    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, labels=None):
        # last hidden layer
        last_hidden_state = self.xlnet(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)
        # pool the outputs into a mean vector
        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        logits = self.classifier(mean_last_hidden_state)

        # [ADD SIGMOID FUNC]
        # logits = logits.sigmoid()

        if labels is not None:
            pos_weight = torch.ones([self.num_labels]) * 10
            pos_weight = pos_weight.to(self.device)
            loss_fct = BCEWithLogitsLoss(pos_weight)
            # loss_fct = BCELoss()
            # loss_fct = MultiLabelSoftMarginLoss()

            # loss = loss_fct(logits.view(-1, self.num_labels),
            #                 labels.view(-1, self.num_labels))

            loss = loss_fct(logits, labels)

            return loss
        else:
            return logits

    def freeze_xlnet_decoder(self):
        """
        Freeze XLNet weight parameters. They will not be updated during training.
        """
        for param in self.xlnet.parameters():
            param.requires_grad = False

    def unfreeze_xlnet_decoder(self):
        """
        Unfreeze XLNet weight parameters. They will be updated during training.
        """
        for param in self.xlnet.parameters():
            param.requires_grad = True

    def pool_hidden_state(self, last_hidden_state):
        """
        Pool the output vectors into a single mean vector
        """
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state

def roc_auc_score_FIXED(y_true, y_pred):
    try:
        score = roc_auc_score(y_true, y_pred)
    except ValueError:
        score = accuracy_score(y_true, np.rint(y_pred))
    return score

def train_model(model, num_epochs,
                optimizer,
                train_dataloader, valid_dataloader,
                model_save_path,
                train_loss_set=[], valid_loss_set=[],
                lowest_eval_loss=None, start_epoch=0,
                device="cpu"
                ):
    """
    Train the model and save the model with the lowest validation loss
    """

    model.to(device)

    # model = torch.nn.DataParallel(model)  # make parallel
    # cudnn.benchmark = True

    # trange is a tqdm wrapper around the normal python range
    for i in trange(num_epochs, desc="Epoch"):
        # if continue training from saved model
        actual_epoch = start_epoch + i

        # Training

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0
        num_train_samples = 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            # store train loss
            tr_loss += loss.item()
            num_train_samples += b_labels.size(0)
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # scheduler.step()

        # Update tracking variables
        epoch_train_loss = tr_loss / num_train_samples
        train_loss_set.append(epoch_train_loss)


        print("Train loss: {}".format(epoch_train_loss))

        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables
        eval_loss = 0
        num_eval_samples = 0

        # Evaluate data for one epoch
        for batch in valid_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate validation loss
                loss = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                # store valid loss
                eval_loss += loss.item()
                num_eval_samples += b_labels.size(0)

        epoch_eval_loss = eval_loss / num_eval_samples
        valid_loss_set.append(epoch_eval_loss)

        print("Valid loss: {}".format(epoch_eval_loss))

        if lowest_eval_loss == None:
            lowest_eval_loss = epoch_eval_loss
            # save model
            save_model(model, model_save_path, actual_epoch,
                       lowest_eval_loss, train_loss_set, valid_loss_set)
        else:
            if epoch_eval_loss < lowest_eval_loss:
                lowest_eval_loss = epoch_eval_loss
                # save model
                save_model(model, model_save_path, actual_epoch,
                           lowest_eval_loss, train_loss_set, valid_loss_set)
        print("\n")

    return model, train_loss_set, valid_loss_set


def save_model(model, save_path, epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist):
    """
    Save the model to the path directory provided
    """
    model_to_save = model.module if hasattr(model, 'module') else model
    checkpoint = {'epochs': epochs,
                  'lowest_eval_loss': lowest_eval_loss,
                  'state_dict': model_to_save.state_dict(),
                  'train_loss_hist': train_loss_hist,
                  'valid_loss_hist': valid_loss_hist
                  }
    torch.save(checkpoint, save_path)
    print("Saving model at epoch {} with validation loss of {}".format(epochs,
                                                                       lowest_eval_loss))
    return


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

    pred_probs = np.array([]).reshape(0, num_labels)
    raw_data = np.array([]).reshape(0, num_labels)

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
            raw_data = np.vstack([raw_data, logits.cpu().numpy()])
            logits = logits.sigmoid()
            pred_probs = np.vstack([pred_probs, logits.cpu().numpy()])

    return raw_data, pred_probs


if __name__ == "__main__":
    main()
