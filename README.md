# Introduction 
 Anomaly detection in time series is a difficult task for Deep learning models, especially when the anomalies are inconsistent
 in nature and very rare. Here, a LSTM (Long Short-term memory) based Autoencoder model is used, trained on normal operating 
 data from the source, and then asked to reconstruct other data. Based on validation/test data, a threshold is selected to
 determine the best limit for classifying data into anomalous and normal.
 
 Anomaly generation in time series is a <strong>very</strong> difficult task for Deep learning models, especially when the anomalies are inconsistent
 in nature and very rare. Here, a LSGAN (Least Squares Generative Adversarial Network) is used, trained on normal operating 
 data from the source, being able to learn the distribution of individual tags.
 
 ### Notes for the assessor
 This repository is a final commit of the original repository which was hosted on Azure Devops, with access restricted to select Shell
 Research UK personell only. Whilst all the code used in this project is available, to run the code yourself you will require the dataset and list of LPC tags
 which cannot be uploaded here. Additionally, the commit history of the original version control performed for this project is not available here.
 If you need access to either of these, please contact me and I can arrange a demonstration to be assessed. 

# Summary
The code here covers the full processes of anomaly detection and generation, starting from importing the data and pre-processing, right up to making inferences
with an untrained model for detection and right up to learning the distribution of a tags and generating a time series from the data for generation.
Here is a quick summary of what happens:

### Data pre-processing
The data coming in is usually quite messy. The main issue is with some features haveing a lot of missing values in the time
the data as been recorded for. Additionally, there is some data in there coressponding to purposefully abnormal behaviour, 
such as when the machines were offline for non-anomaly related issues. To tackle this, two functions are run over the dataset:
- ```remove_offline_data```: Removes all data that coressponds to offline or purposefully abnormal behaviour in the assets, with 
criteria defined by specialist engineers
- ```remove_invalid_features```: Removes all features from the dataset that meet certain criteria, such as having too many missing
values or being too difficult to interpolate etc. 

Additionally, we observe some scale changes in many features where the units for measurement or recording from the sensors
changed and were/were not changed back at irregular intervals, hence working with these raw values would be a near impossible
task. Instead, the deltas were taken of the data, where the simple difference from one observation to the previous was 
calculated to avoid this issue entirely and also remove any long term linear trends from the data.

Another inclusion here for the generative models specifically was another criterion to remove featuers which have too many consecutive
repeated values, causing spikes in the distributions of the data. This not only allowed us to remove features that would pose an issue when training, but
also reduced the number of featuers that we had to work with, allowing the size of our GAN models to be manageable.

Once these have been applied, a clean and standardised (normalised) version of the dataset will be saved as clean_dataset.csv.

### Data Loading and support functions
To create the data sets and loaders required for feeding data into the model and assessment functions easily, a number of 
functions and processes were developed to set everything up. Essentially, the idea is to separate data into strided, possibly
overlapping sequences that are split into non-overlapping training, validation and test sets (and supersets are made for
normal and anomalous data too) and then placed into data loaders. This makes many tasks down the line much easier. 

Anomaly detection support functions include those for handling anomaly indexes and assessing the trained models in various
and robust ways, and a full exploration of the models latent space embeddings with TSNE as well as its reconstruction errors using 
certain metrics are explored, before giving a final report and visualisation of its classification capabilities

The support functions for anomaly generation revolve around assessing the generated distributions against the real, and
with a small tweak you can see the generators output capabilities live, during training if needed. Additionally, a few other functions for
creating new subsets of the data are included, such as being able to clip the extreme values off the data for better training and so on.

### Model initialisation and training 
Given its own module entirely, the model initialisation, definition and training functions make use of all the other infrastruture
set up so far to make it very simple to train a model, customisable to some extent, for a suitable dataset. The provided
hyperparameters are carefully chosen over extensive testing, but can easily be changed for a slightly different problem. 

# How to run
Once you've acquired a dataset, its quite easy to get started. Move the dataset into the same directory as where this repository
has been cloned, and simply run the ```detect_anomalies.py``` or ```generate_anomalies.py``` scripts. You can do this from an IDE or from a terminal with
```python detect_anomalies.py```/```python generate_anomalies.py``` and it handles itself from there. 

# Requirements
Aside from an appropriate dataset, youll also need as a minimum:
- Python 3.6

And the following (not included with a pip/anaconda python installation):
- Pytorch 1.1
- Cuda 10.0
- pandas_ml 0.61
- sklearn 0.21
- numba 0.45


