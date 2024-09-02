# EEGnew
An analysis and classification tasks correspondingly identify risk and danger in pre-event and on-event.

## Prerequisites:

```plaintext
torch==2.0.1
numpy==1.23.5
scikit-learn==1.2.2
matplotlib==3.7.1
h5py==3.7.0
```
Required Packages are listed in [`requirements.txt`](requirements.txt)

To install the required dependencies, please use the following commands:

For Pip installation

```bash
pip install -r requirements.txt
```

For Conda installation

```bash
conda install -r requirements.txt
```

The EEG device used is Neuroscan. If another device is used, the channel list should be adjusted accordingly.

## all topoplot

**P300**

P300 component topography is set to 50ms/step, starting at -200ms and ends in 999ms.

**dataset**

Segmented data of a subject, Subject 1, consisting 16 separate EEG data `.set` files. Each file contains 25 randomly selected events from 14 distint traffic event scenario settings.

**Topography_plot.m**

Code to plot topographic in time range of -0.2s to 1s. Path and steps should be adjusted before reading another data. The code require EEGLAB to run.

Topographic plot is set with default steps of 100ms/step, starting from -2000ms to 999ms.

Run **topoplot_main.m** to plot topograph. Check path and dataset format before running the code. The code requires EEGLAB to run.

## Data
Datasets of all subjects for Classification tasks. Classification data consists data with high risk events such as event 1,2,7,8,13, and 14 combined together labelled as `all_events`, `X1` is data with high risk target ( with P300 component ) while `X2` is non-target and `X3` is data with low risk target ( no P300 component ) . `sub1_1_data.mat` is a train set while the rest are test sets.

## Classifier
Please refer to [`Classifer`](Classifier)

**CSP+XGBDIM**

[`XGBDIM.py`](Classifier\CSP+XGBDIM\XGBDIM.py) train and test the P300 component datasets, time domain -0.2s to 1s.

[`CSP.py`](Classifier\CSP+XGBDIM\CSP.py) train and test all data across -2s to -1s.

To run the model with both classfiers, run [`main.py`](Classifier\CSP+XGBDIM\main.py) 

**EEGModels**

[`EEGModels.py`](Classifier\EEGModels\EEGModels.py) train and test the P300 component datasets, time domain -0.2s to 1s.
Include EEGNet, ShallowConvNet and DeepConvNet SOTA models.

[`CSP.py`](Classifier\EEGModels\CSP.py) train and test all data across -2s to -1s.

To run the model with both classfiers, run [`CSP_EEGModels_LR.py`](Classifier\EEGModels\CSP_EEGModels_LR.py) 

**CRNN**

[`CRNN.py`](Classifier\CRNN\CRNN.py) holds the structure of CRNN. Train and test all data across -2s to -1s and the P300 component datasets, time domain -0.2s to 1s

[`Data.py`](Classifier\CRNN\Data.py) and [`utils.py`](Classifier\CRNN\utils.py) process the input data (including labelling) and save results.

To run the model, run [`main.py`](Classifier\CRNN\main.py)