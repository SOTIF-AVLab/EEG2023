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

To install the required dependencies, use the following commands:

For Pip installation

```bash
pip install -r requirements.txt
```

For Conda installation

```bash
conda install -r requirements.txt
```

The EEG device used is Neuroscan. If another device is used, the channel list should be adjusted accordingly.

## P300
P300 component topography is set to 50ms/step, starting at -200ms and ends in 999ms.

**Example Data**

Segmented data of a subject, Subject 1, consisting 16 separate EEG data `.set` files. Each file contains 25 randomly selected events from 14 distint traffic event scenario settings.

**Topography_plot.m**

Code to plot topographic in time range of -0.2s to 1s. Path should be adjusted before reading another data. The code require EEGLAB to run.

## all topoplot
Topographic plot with 100ms/step, starting from -2000ms to 999ms.

Run **topoplot_main.m** to plot topograph. Check path and dataset format before running the code. The code requires EEGLAB to run.

## data
An example datasets (Subject1) for Classification tasks. Classification data consists data with high risk events such as event 1,2,7,8,13, and 14. `sub1_1_data.mat` is a train set while the rest are test sets.

## Classifier
`XGB.py` train and test the P300 component datasets, time domain -0.2s to 1s.

`CSP.py` train and test all data across -2 seconds to -1 second.

To run the model with both classfiers, run `main.py` 
