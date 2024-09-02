import numpy as np
from sklearn.utils import shuffle
import os
import scipy.io as sio
import random

# 3000 datapoints downsampled to 750 for each trial
# 54 channels are used for each trial
# prepare the data for the model
# label 1: high-risk
# label 2: safe
# label 3: low-risk
class Data():
    def __init__(self, data_path, subject, sub_idx=1):
        self.subject = 'subject' + str(subject)
        self.data_path = data_path
        self.sub_idx = sub_idx
    def read_data(self, dataset):
        if dataset is not None:
            print('Importing data')
            N_trainsets = np.shape(dataset)[0]  # 1  # np.shape(trainset)
            All_X1 = []
            All_X2 = []
            All_X3 = []
            for idx_set in range(N_trainsets):
                data = sio.loadmat(
                    os.path.join(
                        self.data_path, 'sub' + str(self.sub_idx) + '_' + str(dataset[idx_set]) + '_data.mat'
                    )
                )
                X1 = np.float32(data['X1'])
                X2 = np.float32(data['X2'])
                X3 = np.float32(data['X3'])
               
                if len(All_X1) == 0:
                    All_X1 = X1[:, 500:1250, :].copy()
                    All_X2 = X2[:, 500:1250, :].copy()
                    All_X3 = X3[:, 500:1250, :].copy()
                else:
                    if len(X1.shape)==2:
                        X1 = X1[:,:,np.newaxis]
                    else:
                        All_X1 = np.concatenate((All_X1, X1[:, 500:1250, :]), axis=2)
                    if len(X2.shape)==2:
                        X2 = X2[:,:,np.newaxis]
                    else:
                        All_X2 = np.concatenate((All_X2, X2[:, 500:1250, :]), axis=2)
                    if len(X3.shape)==2:
                        X3 = X3[:,:,np.newaxis]
                    else:
                        All_X3 = np.concatenate((All_X3, X3[:, 500:1250, :]), axis=2)
                    
            
            X1 = All_X1.copy()
            del All_X1
            X2 = All_X2.copy()
            del All_X2
            X3 = All_X3.copy()
            del All_X3
            print('Importing data finished')
            return X1, X2, X3
        else:
            print('Dataset is None')
            exit(1)
            
    def label_data(self, X1, X2, X3):
        data_with_labels = []
        for i in range(np.size(X1, 2)):
            data_with_labels.append((X1[:, :, i], 1))
        
        for j in range(np.size(X2, 2)):
            data_with_labels.append((X2[:, :, j], 2))

        for k in range(np.size(X3, 2)):
            data_with_labels.append((X3[:, :, k], 3))
        
        return data_with_labels
    
    def prepare_training_testing_data(self):
        idx = np.array([1, 2, 3, 4])
        X1, X2, X3 = self.read_data(idx)
        shuffle(X1)
        shuffle(X2)
        shuffle(X3)
        print('X1 shape:', X1.shape)
        # 60% training, 40% testing
        training_set = self.label_data(X1[:, :, :int(0.6 * np.size(X1, 2))], X2[:, :, :int(0.6 * np.size(X2, 2))], X3[:, :, :int(0.6 * np.size(X3, 2))])
        testing_set = self.label_data(X1[:, :, int(0.6 * np.size(X1, 2)):], X2[:, :, int(0.6 * np.size(X2, 2)):], X3[:, :, int(0.6 * np.size(X3, 2)):])

        return training_set, testing_set
    
    def prepare_data(self):
        training_set, testing_set = self.prepare_training_testing_data()

        train_data = [item[0] for item in training_set]
        x_train = np.stack(train_data, axis=2)
        train_label = [item[1] for item in training_set]
        y_train = np.stack(train_label)

        test_data = [item[0] for item in testing_set]
        x_test = np.stack(test_data, axis=2)
        test_label = [item[1] for item in testing_set]
        y_test = np.stack(test_label)

        # permute the data
        return x_train, y_train, x_test, y_test