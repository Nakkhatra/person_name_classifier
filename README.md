## People Name Classifier

#### The model was trained by fine tuning multilingual BERT. It was trained in 2 different ways: 1. Freezing the encoder 2. Without freezing and also training the encoder. 450,000 data was taken from the whole dataset and split into train, validation and test sets due to time and resource limitations.

#### Whole project link in google drive: https://drive.google.com/drive/folders/1HUN3ySvjDFXcgNJnHCOtqyS6tu5TFAnb?usp=sharing
#### **Set the root directory to 'person_name_classifier/name-classifer/'** (All function and utils are called with respect to this root directory)

#### Download and extract the name.ttl and person.ttl into the 'data/' directory: Download link: https://episerver99-my.sharepoint.com/:u:/g/personal/zsolt_pocsaji_episerver_com/EeHdaCtLJP1HlurrLtBOKxcBsq4JhkiLJqw7f1sAmwa_ZQ?e=uNbuqJ

#### Download the weights and put into the 'weights/' directory (Also can be found in the above link in the 'name_classifier/weights/' directory:
- 300k-32BS-multilingual-encoder-unfroze.pth : https://drive.google.com/file/d/16ppv9ioC1oX54HmWbznyGea4q_IDIq01/view?usp=sharing
- 450k-64BS-multilingual-encoder-froze-30epochs.pth : https://drive.google.com/file/d/1qPY79qiCLTjsEXvqIyp9mZiDvuj-dBLg/view?usp=sharing

#### The test set accuracy of the 2nd model is about **92%** and for the 1st model is about **87.6%**. But the 1st model **'weights/300k-32BS-multilingual-encoder-unfroze.pth'** is really heavy and if the GPU is not powerful enough, it will throw CUDA out of memory error while loading the model. However, the 2nd model **'weights/450k-64BS-multilingual-encoder-froze-30epochs.pth'** is much lighter since the number of trainable parameters are much lower than the first one. The training of the 1st model is shown in the google colab notebook in the 'colab_notebooks/' directory. It can be reproduced by training the model by setting freeze_bert to False.

#### The learning curves after training and the confusion matrix after evaluating the model are in the 'images/' folder.

### **Pipeline:**

#### Setup the environment using the requirements.txt file

### creating the labelled dataset ('data/labelled_dataset.csv') from the name.ttl and person.ttl files:
```
python name_classifier.py load_data --in-folder 'data'
```

#### This will create the labelled_dataset.csv in the data directory.

### Preparing the dataset, splitting the dataset into train, validation and test sets and training the model:
```
python name_classifier.py --train --in_folder 'data/labelled_dataset.csv' --out_folder 'weights/450k-64BS-multilingual-encoder-froze-30epochs.pth'   --sample_size 450000 --plot False
```

Example: 
```
python name_classifier.py --train --in_folder --out_folder  --sample_size --plot False
```

	* --in_folder: corresponds to the data containing the csv files
	* --out_folder: corresponds to a folder where the trained model will be serialised to
	* --sample_size: How much of the data to be taken to split into train, validation and test (~60%, ~20%, ~20% split). The model was trained with sample_size 450,000
	* --plot: (bool) True by default. If set to false, the model will not plot the learning curves after finishing training.

### This will evaluate the model on the test dataset and print the scores along with confusion matrix if print_reports is set to True.
```
python name_classifier.py evaluate_model --in_folder --weightsdir --sample_size --print_reports
```
Example:
```
python name_classifier.py evaluate_model --in_folder 'data/labelled_dataset.csv' --weightsdir 'weights/450k-64BS-multilingual-encoder-froze-25epochs.pth' --sample_size 450000 --print_reports False
```

	* --in_folder: Directory of the labelled dataset .csv file
	* --weightsdir: Directory of the trained saved weights
	* --sample_size: How much of the data to be taken to split into train, validation and test (~60%, ~20%, ~20% split). The model was trained with sample_size 450,000
	* --print_reports: (bool) By default it is True, if False, it will not print the confusion matrix.




