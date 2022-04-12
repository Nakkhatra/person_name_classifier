from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertModel
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import f1_score, precision_score

import re
import torch
from transformers import BertTokenizer

from collections import defaultdict
import copy
import random
import os
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd

cudnn.benchmark = True

# Set the parameters
params = {
    "device": "cuda",
    "lr": 5e-5,
    "batch_size": 64,
    "num_workers": 4,
    "epochs": 30,
    "num_classes": 1
}


def text_preprocessing(s):
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Remove trailing whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# text_preprocessing("Md. Shahrin Nakkhatra")

# Create a function to tokenize a set of texts


def preprocessing_for_bert(data, tokenizer):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=32,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,    # Return attention mask
            truncation=True
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


def splitData(data, sample_size):
    """
    Generate data splits
    """

    """## Fixing class imbalance"""

    data = data.dropna()

    data_name = data[data['label'] == 1]

    data_no_name = data[data['label'] == 0]

    data_no_name_downsampled = data_no_name.sample(data_name.shape[0])

    data_balanced = pd.concat([data_name, data_no_name_downsampled])

    """### Taking a portion of the entire dataset (300000 samples in total) as I do not have the resources to train the huge entire dataset"""

    data_balanced_small = data_balanced.sample(sample_size)

    X_train, X_test, y_train, y_test = train_test_split(
        data_balanced_small['name'], data_balanced_small['label'], stratify=data_balanced_small['label'])

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, stratify=y_train)

    return X_train, X_val, X_test, y_train, y_val, y_test


def createDataloader(X_train, X_val, X_test, y_train, y_val, y_test, tokenizer):
    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(np.array(y_train))
    val_labels = torch.tensor(np.array(y_val))
    test_labels = torch.tensor(np.array(y_test))

    train_inputs, train_masks = preprocessing_for_bert(
        list(np.array(X_train)), tokenizer)
    val_inputs, val_masks = preprocessing_for_bert(
        list(np.array(X_val)), tokenizer)
    test_inputs, test_masks = preprocessing_for_bert(
        list(np.array(X_test)), tokenizer)

    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=params["batch_size"])

    # Create the DataLoader for our validation set
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(
        val_data, sampler=val_sampler, batch_size=params["batch_size"])

    # Create the DataLoader for our test set
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(
        test_data, sampler=test_sampler, batch_size=params["batch_size"])

    return train_dataloader, test_dataloader, val_dataloader

# Create the BertClassfier class


class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """

    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, params['num_classes']

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits


def count_parameters(model):
    print("===========================================================")
    return f"All parameters: {sum(p.numel() for p in model.parameters())} | Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"


def initialize_model(params, train_dataloader, freeze_bert):

    model = BertClassifier(freeze_bert=freeze_bert)

    # Tell PyTorch to run the model on GPU
    model.to(params["device"])

    criterion = nn.BCEWithLogitsLoss().to(params["device"])

    # Create the optimizer
    optimizer = AdamW(model.parameters(),
                      lr=params['lr'],    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

    total_steps = len(train_dataloader) * params['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)

    print(count_parameters(model))

    return model, optimizer, criterion, scheduler


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def train_model(train_loader, model, criterion, optimizer, scheduler, epoch, params):
    metric_monitor = MetricMonitor()
    model.train()

    stream = tqdm(train_loader)

    for i, (batch) in enumerate(stream, start=1):
        # Clear any previously calculated gradient
        model.zero_grad()

        b_input_ids, b_attn_mask, b_labels = tuple(
            t.to(params['device']) for t in batch)
        b_labels = b_labels.float().view(-1, 1)

        # target = target.to(params["device"], non_blocking=True).float().view(-1, 1)

        output = model(b_input_ids, b_attn_mask)
        loss = criterion(output, b_labels)

        accuracy = calculate_accuracy(output, b_labels)

        metric_monitor.update("Loss", loss.item())

        metric_monitor.update("Accuracy", accuracy)

        # brack prop
        loss.backward()
        optimizer.step()
        scheduler.step()

        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(
                epoch=epoch, metric_monitor=metric_monitor)
        )

    return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Accuracy']['avg']


def validate(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()
    model.eval()

    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, batch in enumerate(stream, start=1):
            b_input_ids, b_attn_mask, b_labels = tuple(
                t.to(params['device']) for t in batch)
            b_labels = b_labels.float().view(-1, 1)
            # target = target.to(params["device"], non_blocking=True).float().view(-1, 1)

            output = model(b_input_ids, b_attn_mask)
            loss = criterion(output, b_labels)

            accuracy = calculate_accuracy(output, b_labels)

            metric_monitor.update("Loss", loss.item())

            metric_monitor.update("Accuracy", accuracy)

            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(
                    epoch=epoch, metric_monitor=metric_monitor)
            )
    # print(metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Accuracy']['avg'])
    return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Accuracy']['avg']


def calculate_accuracy(output, target):
    output = torch.sigmoid(output) >= 0.5
    target = target == 1.0
    return torch.true_divide((target == output).sum(dim=0), output.size(0)).item()


def saveModel(model, out_folder: str, epoch, optimizer):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, out_folder)


def training_loop(model, train_dataloader, val_dataloader, optimizer, criterion, scheduler, params, out_folder):
    train_losses = []
    train_acc = []

    val_losses = []
    val_acc = []

    prev_accuracy = 0
    prev_loss = 2**31-1
    best_epoch = 0
    prev_best_epoch = 0

    for epoch in range(1, params["epochs"] + 1):
        train_loss, train_accuracy = train_model(
            train_dataloader, model, criterion, optimizer, scheduler, epoch, params)
        val_loss, val_accuracy = validate(
            val_dataloader, model, criterion, epoch, params)

        # Saving the weights only if validation accuracy increases by 8% or validation loss decreases by 5% or when val accuracy increases and loss decreases at the same time
        if (val_accuracy >= (prev_accuracy * 1.08)) or (val_loss <= (prev_loss * 1.05)) or (val_accuracy >= prev_accuracy and val_loss <= prev_loss):
            best_epoch = epoch
            saveModel(model=model, out_folder=out_folder,
                      epoch=epoch, optimizer=optimizer)

        prev_best_epoch = epoch
        prev_accuracy = val_accuracy
        prev_loss = val_loss

        train_losses.append(train_loss)

        train_acc.append(train_accuracy)
        val_losses.append(val_loss)
        val_acc.append(val_accuracy)

    return train_losses, train_acc, val_losses, val_acc, best_epoch


"""### Plotting loss and accuracy curves for train and validation for choosing best weights"""


def plot_curve(property_val, property_train, curve_type: str, sample_size: int):
    """
    curve_type: It is a string to be added on the filename to represent whether it's a loss curve or accuracy curve
    """
    plt.figure(figsize=(10, 5))
    plt.title(f"Training and Validation Loss with {sample_size} samples")
    plt.plot(property_val, label="val")
    plt.plot(property_train, label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'images/{curve_type}_curves_{sample_size//1000}k.jpg')
    plt.show()


"""### Loading saved weights for testing and inference"""


def calculate_metrics(output, target):
    output = torch.sigmoid(output) >= 0.5
    target = target == 1.0
    target = target.cpu()
    output = output.cpu()

    return torch.true_divide((target == output).sum(dim=0), output.size(0)).item(), f1_score(target, output), precision_score(target, output)


def load_model(model, optimizer, weightsdir: str):
    model_path = weightsdir
    checkpoint = torch.load(model_path, map_location=params['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch


def test_pipeline(model, test_loader, epoch, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(test_loader)

    with torch.no_grad():
        for i, batch in enumerate(stream, start=1):
            b_input_ids, b_attn_mask, b_labels = tuple(
                t.to(params['device']) for t in batch)
            b_labels = b_labels.float().view(-1, 1)
            # target = target.to(params["device"], non_blocking=True).float().view(-1, 1)

            output = model(b_input_ids, b_attn_mask)

            if i == 1:
                all_outputs = output
            else:
                all_outputs = torch.cat((all_outputs, output), 0)

            accuracy, _, _ = calculate_metrics(
                output, b_labels)

            metric_monitor.update("Accuracy", accuracy)

            stream.set_description(
                "Epoch: {epoch}. Test. {metric_monitor}".format(
                    epoch=epoch, metric_monitor=metric_monitor)
            )

    return metric_monitor.metrics['Accuracy']['avg'], all_outputs
