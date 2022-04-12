from posixpath import split
from data_preparation import load_and_prepare
import fire

from numpy import NaN
import pandas as pd
import re
import os
from tqdm import tqdm
from pipeline import *
import warnings
from data_preparation import load_and_prepare


def load_data(in_folder: str):
    """
    in_folder: name of the folder where the labelled dataset will be saved as .csv file
    """
    return load_and_prepare(in_folder)


def split_data(data, sample_size):
    """
    Generate data splits
    """
    return splitData(data, sample_size)


def save_model(model, out_folder: str, epoch, optimizer):
    """
    Serialise the model to an output folder 
    """
    return saveModel(model, out_folder, epoch, optimizer)


def train(in_folder: str, out_folder: str, sample_size: int, params=params, plot=True) -> None:
    """
    Consume the data from the input folder to generate the model
    and serialise it to the out_folder.

    in_folder:

    Added: While training, best weights will be saved to the out_folder
    """
    # NOTE: Calling the functions and training . . .

    print('Loading the dataset from drive. . .')
    # Load the dataset
    data = pd.read_csv(in_folder)

    print('Splittiing the dataset into train, val and test sets. . .')
    # Split the dataset into train, validation and test split
    X_train, X_val, X_test, y_train, y_val, y_test = splitData(
        data, sample_size)

    # print(f'Data split: train: {len(y_train)}, validation: {len(y_val)}, test: {len(y_test)}')

    # Load the BERT tokenizer
    print('Loading BERT tokenizer. . .')
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-multilingual-cased', do_lower_case=True)

    print('Creating train, val and test dataloaders. . .')
    # Create train, val and test dataloaders
    train_dataloader, test_dataloader, val_dataloader = createDataloader(
        X_train, X_val, X_test, y_train, y_val, y_test, tokenizer=tokenizer)

    print('Initializing the model. . .')
    # Initialize the model
    bert_classifier, optimizer, criterion, scheduler = initialize_model(
        params, train_dataloader, freeze_bert=True)

    print('Attention! Training is strating!. . .')
    # Train the model and save the weights
    train_losses, train_acc, val_losses, val_acc, best_epoch = training_loop(
        bert_classifier, train_dataloader, val_dataloader, optimizer, criterion, scheduler, params, out_folder)

    print(f"Training complete!!! Best weights found for epoch: {best_epoch}")

    print("Plotting learning curves. . .")
    # Plot the learning curves
    if plot:
        plot_curve(val_losses, train_losses,
                   curve_type='loss', sample_size=sample_size)
        plot_curve(val_acc, train_acc, curve_type='accuracy',
                   sample_size=sample_size)


def evaluate_model(in_folder: str, weightsdir: str, sample_size: int, params=params, print_reports=True):
    """
    in_folder: directory of the entire dataset, this function will generate test data from that
    weightsdir: directory of the trained weights of the model
    Evaluate your model against the test data.
    """
    print("===========================================================")
    print('Loading the dataset from drive. . .')
    # Load the dataset
    data = pd.read_csv(in_folder)

    print("===========================================================")
    print('Splittiing the dataset into train, val and test sets. . .')
    # Split the dataset into train, validation and test split
    X_train, X_val, X_test, y_train, y_val, y_test = splitData(
        data, sample_size)

    # print(f'Data split: train: {len(y_train)}, validation: {len(y_val)}, test: {len(y_test)}')

    # Load the BERT tokenizer
    print("===========================================================")
    print('Loading BERT tokenizer. . .')
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-multilingual-cased', do_lower_case=True)

    print("===========================================================")
    print('Creating train, val and test dataloaders. . .')
    # Create train, val and test dataloaders
    train_dataloader, test_dataloader, _ = createDataloader(
        X_train, X_val, X_test, y_train, y_val, y_test, tokenizer=tokenizer)

    print("===========================================================")
    print('Initializing the model. . .')
    # Initialize the model
    bert_classifier, optimizer, criterion, scheduler = initialize_model(
        params, train_dataloader, freeze_bert=True)

    print("===========================================================")
    print('Loading the model. . .')
    # Load the saved model
    loaded_epoch = load_model(model=bert_classifier, optimizer=optimizer,
                              weightsdir=weightsdir)

    print("===========================================================")
    print('Evaluating the model against test dataset. . .')
    # Evaluate the model
    test_accuracy, all_outputs = test_pipeline(model=bert_classifier,
                                               test_loader=test_dataloader, epoch=loaded_epoch, params=params)

    all_outputs = torch.sigmoid(all_outputs) >= 0.5
    all_outputs = all_outputs.tolist()
    y_test = torch.Tensor(y_test.values).float().tolist()

    f1, precision = f1_score(
        y_test, all_outputs), precision_score(y_test, all_outputs)

    print("===========================================================")

    if print_reports:
        print("Plotting the confusion matrix . . .")

        from sklearn.metrics import confusion_matrix, classification_report
        from matplotlib import pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_test, all_outputs)

        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.savefig(f'images/confusion_matrix_{sample_size//1000}k.jpg')
        plt.show()

        print(classification_report(y_test, all_outputs))

    return f"Test F1 Score: {round(f1,4)} | Test precision: {round(precision, 4)}"


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    fire.Fire()
