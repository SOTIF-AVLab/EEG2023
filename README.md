# EEGnew
An analysis and classification tasks correspondingly identify risk and danger in pre-event and on-event.

Require:

`torch` | `numpy` | sklearn.svm | sklearn.metrics | matplotlib | h5py

The EEG device used is Neuroscan. If another device is used, the channel list should be adjusted accordingly.

# P300
P300 component topography is set to 50ms per step. The topography starts at -200ms and ends in 999ms.
# all topoplot
Topographic plot with 150ms per step. Starting from -2000ms to 999ms.
# data
An example datasets for Classification tasks.
# Classifier
XGB.py train and test the P300 component datasets, time domain -0.2s to 1s.
CSP.py train and test all data across -2 seconds to -1 second.
main.py runs both XGB.py and CSP.py
