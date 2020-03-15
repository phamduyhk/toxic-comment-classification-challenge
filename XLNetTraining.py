# coding: utf-8
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from pytorch_transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification
from pytorch_transformers import AdamW

from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("device: {}".format(device))

df = pd.read_csv("./data/train.csv", encoding='utf-8')
sentences = df['comment_text'].values

test_df = pd.read_csv("./data/test.csv", encoding='utf-8')
test_sentences = test_df['comment_text']

test_tokenized_texts = [tokenizer.tokenize(sent) for sent in test_sentences]
print("test sentence sample: {}".format(test_tokenized_texts[0]))

# define output dataframe
output = pd.read_csv("./data/sample_submission.csv")

for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    labels = df[label].values

    tokenizer = XLNetTokenizer.from_pretrained(
        'xlnet-base-cased', do_lower_case=True)

    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    print("Tokenize the first sentence:")
    print(tokenized_texts[0])

    # Set the maximum sequence length. The longest sequence in our training set is 47, but we'll leave room on the end anyway.
    MAX_LEN = 128

    # Use the XLNet tokenizer to convert the tokens to their index numbers in the XLNet vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    test_input_ids = [tokenizer.convert_tokens_to_ids(
        x) for x in test_tokenized_texts]

    # Pad our input tokens
    input_ids = pad_sequences(
        input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    test_input_ids = pad_sequences(
        test_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Create attention masks
    attention_masks = []
    test_attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # Create a mask of 1s for test token
    for seq in test_input_ids:
        seq_mask = [float(i > 0) for i in seq]
        test_attention_masks.append(seq_mask)

    # Use train_test_split to split our data into train and validation sets for training

    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                                        random_state=2018, test_size=0.2)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                           random_state=2018, test_size=0.2)

    # Convert all of our data into torch tensors, the required datatype for our model

    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    test_inputs = torch.tensor(test_input_ids)
    test_masks = torch.tensor(test_attention_masks)

    # Select a batch size for training. For fine-tuning with XLNet, the authors recommend a batch size of 32, 48, or 128. We will use 32 here to avoid memory issues.
    batch_size = 512

    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
    # with an iterator the entire dataset does not need to be loaded into memory

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(
        validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(
        validation_data, sampler=validation_sampler, batch_size=batch_size)

    test_data = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(
        test_data, sampler=validation_sampler, batch_size=batch_size)

    # Load XLNEtForSequenceClassification, the pretrained XLNet model with a single linear classification layer on top.

    model = XLNetForSequenceClassification.from_pretrained(
        "xlnet-base-cased", num_labels=2)
    model.cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    # This variable contains all of the hyperparemeter information our training loop needs
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=2e-5)

    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    # Store our loss and accuracy for plotting
    train_loss_set = []

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 2

    # trange is a tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):

        # Training

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            logits = outputs[1]
            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                output = model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask)
                logits = output[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

    # prediction
    predicts = []
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            output = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
            logits = output[0]

        # Move logits and labels to CPU
        pred = logits.detach().cpu().numpy().tolist()
        predicts += pred

    output[label] = predicts


# save output
now = datetime.datetime.now()
output.to_csv("submission_XLNET_{}.csv".format(now.timestamp()), index=False)
