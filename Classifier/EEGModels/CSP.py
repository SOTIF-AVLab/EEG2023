# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:58:41 2023

@author: Kevi023
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, accuracy_score
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py
import random

class CSPClassifier:
    def __init__(self, data_path, subject, sub_idx):
        self.training_set = None
        self.testing_set = None
        self.X = None
        self.Y = None
        self.T = None
        
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
                data = h5py.File(
                    os.path.join(
                        self.data_path, 'sub' + str(self.sub_idx) + '_' + str(dataset[idx_set]) + '_data.mat'
                    )
                )
                X1 = np.float32(np.transpose(data['X1']))
                X2 = np.float32(np.transpose(data['X2']))
                X3 = np.float32(np.transpose(data['X3']))
               
                if All_X1 == []:
                    All_X1 = X1[:, :375, :].copy()
                    All_X2 = X2[:, :375, :].copy()
                    All_X3 = X3[:, :375, :].copy()
                else:
                    if len(X1.shape)==2:
                        X1 = X1[:,:,np.newaxis]
                    else:
                        All_X1 = np.concatenate((All_X1, X1[:, :375, :]), axis=2)
                    if len(X2.shape)==2:
                        X2 = X2[:,:,np.newaxis]
                    else:
                        All_X2 = np.concatenate((All_X2, X2[:, :375, :]), axis=2)
                    if len(X3.shape)==2:
                        X3 = X3[:,:,np.newaxis]
                    else:
                        All_X3 = np.concatenate((All_X3, X3[:, :375, :]), axis=2)
                    
                    # concatenate X2 and X3
                    
            All_X1 = np.concatenate((All_X1, All_X3), axis=2)
            
            X1 = All_X1.copy()
            del All_X1
            X2 = All_X2.copy()
            del All_X2
            print('Importing data finished')
            return X1, X2
        else:
            print('Dataset is None')
            exit(1)
            
    def label_data(self, X1, X2):
        data_with_labels = []
        for i in range(np.size(X1, 2)):
            data_with_labels.append((X1[:, :, i], 1))
        
        for j in range(np.size(X2, 2)):
            data_with_labels.append((X2[:, :, j], 0))
        
        return data_with_labels
    
    def prepare_training_data(self):
        idx = np.array([1])
        X1, X2 = self.read_data(idx)
        self.training_set = self.label_data(X1, X2)
        random.shuffle(self.training_set)
        
    def prepare_testing_data(self):
        idx = np.array([2, 3, 4])
        X1, X2 = self.read_data(idx)
        self.testing_set = self.label_data(X1, X2)
    
    def get_data(self, dataset):

        
        X1, X2 = self.read_data(dataset)

    
    def extract_CSP(self, EEGSignals, CSPMatrix, nFilterPairs):
        """
        Extract CSP features from EEG signals.
        
        Args:
            EEGSignals (dict): Dictionary containing EEG data in the 'x' field, with shape [Nc, Ns, Nt].
            CSPMatrix (ndarray): Previously learned CSP matrix.
            nFilterPairs (int): Number of pairs of CSP filters to be used.
                                The number of features extracted will be twice this value.
        
        Returns:
            ndarray: Extracted features as a [Nt, (nFilterPairs * 2 + 1)] matrix,
                    with the class labels as the last column.
        """
        nTrials = EEGSignals['x'].shape[2]
        features = np.zeros((nTrials, 2 * nFilterPairs + 1))
        Filter = CSPMatrix[np.arange(0, CSPMatrix.shape[0], CSPMatrix.shape[0] - nFilterPairs), :]

        # Extracting the CSP features from each trial
        for t in range(nTrials):
            # Projecting the data onto the CSP filters
            projectedTrial = np.dot(Filter, EEGSignals['x'][:, :, t])

            # Generating the features as the log variance of the projected signals
            variances = np.var(projectedTrial, axis=1)
            for f in range(len(variances)):
                features[t, f] = np.log(1 + variances[f])

        return features# ... (your existing extract_CSP code here)

    def learn_CSP(self, EEGSignals, classLabels):
        """
        Learn the Common Spatial Patterns (CSP) matrix from EEG signals.

        Args:
            EEGSignals (dict): Dictionary containing EEG data in the 'x' field, with shape (nchannel, nsample, ntrials).
            classLabels (ndarray): Labels indicating the class of each trial.

        Returns:
            ndarray: The CSPMatrix.
        """
        nChannels = EEGSignals['x'].shape[0]
        nTrials = EEGSignals['x'].shape[2]
        nClasses = classLabels.shape[0]

        if nClasses != 2:
            print('Error! CSP can only be used for two classes')
            return

        # Create the covariance matrix for each class
        covMatrices = [None] * nClasses

        # Computing the normalized covariance matrices for each trial
        trialCov = np.zeros((nChannels, nChannels, nTrials))
        for t in range(nTrials):
            E = EEGSignals['x'][:, :, t]
            EE = np.dot(E, E.T)
            trialCov[:, :, t] = EE / np.trace(EE)

        # Computing the covariance matrix for each class
        for c in range(nClasses):
            indices = EEGSignals['y'] == classLabels[c]
            covMatrices[c] = np.mean(trialCov[:, :, indices], axis=2)

        # The total covariance matrix
        covTotal = covMatrices[0] + covMatrices[1]

        # Whitening the transform of the total covariance matrix
        eigenvalues, Ut = np.linalg.eigh(covTotal)
        eigenvalues = eigenvalues[::-1]  # Reverse the order
        Ut = Ut[:, ::-1]  # Reverse the order
        P = np.diag(np.sqrt(1.0 / eigenvalues)) @ Ut.T  # Matrix multiplication

        # Transforming covariance matrix of the first class using P
        transformedCov1 = P @ covMatrices[0] @ P.T

        # EVD of the transformed covariance matrix
        eigenvalues, U1 = np.linalg.eigh(transformedCov1)
        eigenvalues = eigenvalues[::-1]  # Reverse the order
        U1 = U1[:, ::-1]  # Reverse the order
        CSPMatrix = U1.T @ P

        return CSPMatrix

    def CSP_Feature(self):
        EEGSignals = {}
        data = [item[0] for item in self.training_set]
        EEGSignals['x'] = np.stack(data, axis=2)
        label = [item[1] for item in self.training_set]
        EEGSignals['y'] = np.stack(label)
        self.Y =  EEGSignals['y']

        classLabels = np.unique(EEGSignals['y'])
        CSPMatrix = self.learn_CSP(EEGSignals, classLabels)

        nFilterPairs = 1

        # Extract the feature of the training data
        self.X = self.extract_CSP(EEGSignals, CSPMatrix, nFilterPairs)

        # Extract the features of the testing data
        EEGSignals = {}
        data = [item[0] for item in self.testing_set]
        EEGSignals['x'] = np.stack(data, axis=2)
        label = [item[1] for item in self.testing_set]
        EEGSignals['y'] = np.stack(label)
        self.T = self.extract_CSP(EEGSignals, CSPMatrix, nFilterPairs)
    
    # def plot_CSP(self):
    #
    #     color_S = np.array([0, 102, 255]) / 255
    #     color_D = np.array([255, 0, 102]) / 255
    #
    #     # Plot the training set
    #     plt.figure()
    #     pos = np.where(self.Y == 1)[0]
    #     plt.plot(self.X[pos, 0], self.X[pos, 1], 'x', color=color_D, linewidth=2)
    #
    #     pos = np.where(self.Y == 0)[0]
    #     plt.plot(self.X[pos, 0],self. X[pos, 1], 'x', color=color_S, linewidth=2)
    #     plt.xlabel("Principal Feature 1")
    #     plt.ylabel("Principal Feature 2")
    #     plt.legend(['Target', 'Non-Target'])
    #     plt.title("Training Set")
    #
    #     # Plot the testing set
    #     plt.figure()
    #     label = [item[1] for item in self.testing_set]
    #     YT = np.stack(label)
    #     pos = np.where(YT == 1)[0]
    #     plt.plot(self.T[pos, 0], self.T[pos, 1], 'x', color=color_D, linewidth=2)
    #
    #     pos = np.where(YT == 0)[0]
    #     plt.plot(self.T[pos, 0], self.T[pos, 1], 'x', color=color_S, linewidth=2)
    #     plt.xlabel("Principal Feature 1")
    #     plt.ylabel("Principal Feature 2")
    #     plt.legend(['Target', 'Non-Target'])
    #     plt.title("Testing Set")
    #
    #     save_X_path = os.path.join(self.data_path, self.subject + '_X.mat')
    #     save_Y_path = os.path.join(self.data_path, self.subject + '_Y.mat')
    #     save_T_path = os.path.join(self.data_path, self.subject + '_T.mat')
    #     sio.savemat(save_X_path, {"X": self.X})
    #     sio.savemat(save_Y_path, {"Y": self.Y})
    #     sio.savemat(save_T_path, {"T": self.T})


    def classification(self):
        self.prepare_testing_data()
        self.prepare_training_data()
        self.CSP_Feature()
        # self.plot_CSP()
        
        print('######## Training the SVM classifier ########')
        
        model = SVC(probability=True)  # SVM classifier with probability estimates
        model.fit(self.X, self.Y)
        
        predictedProb = model.predict_proba(self.T)[:, 1]  # Predicted probability of class 1
        
        label = [item[1] for item in self.testing_set]
        actualLabel = np.stack(label)

        predictedLabel = model.predict(self.T)
        
        TP = np.sum(np.logical_and(actualLabel == 1, predictedLabel == 1))
        TN = np.sum(np.logical_and(actualLabel == 0, predictedLabel == 0))
        FP = np.sum(np.logical_and(actualLabel == 0, predictedLabel == 1))
        FN = np.sum(np.logical_and(actualLabel == 1, predictedLabel == 0))
        
        TPR = TP / (TP + FN)
        FPR = FP / (TN + FP)
        TNR = TN / (TN + FP)
        
        BA = (TPR + TNR) / 2
        ACC = accuracy_score(actualLabel, predictedLabel)
        
        print(f'True Positive Rate (TPR): {TPR:.4f}')
        print(f'False Positive Rate (FPR): {FPR:.4f}')
        print(f'Balance Accuracy: {BA:.4f}')
        print(f'Accuracy: {ACC:.4f}')
        
        # Compute and plot ROC curve
        fpr, tpr, _ = roc_curve(actualLabel, predictedProb)
        roc_auc = auc(fpr, tpr)
        
        # plt.figure()
        # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title(f'ROC Curve (AUC = {roc_auc:.2f})')
        # plt.legend(loc='lower right')
        # plt.show()
        
        return predictedLabel#, BA, ACC, TPR, FPR
    
if __name__ == '__main__':
    eventNumber = [1,2,7,8,13,14]
    # subject_number = [1,2,3,4,5,6,7,8,10,11,12]
    subject_number = [1,2,3,4,5,6,7,8,10,11,12]
    current_dir = os.getcwd()

    for subjectNumber in subject_number:

        
        subject_path = f'{current_dir}\..\Data\subject{subjectNumber}'
        for event_Number in eventNumber:
            event_path = f'{subject_path}\event{event_Number}'
            csp = CSPClassifier(event_path, subjectNumber, 1)
            csplabel = csp.classification()#,ba,acc,tpr,fpr
        
# =============================================================================
#             result_file_name_2 = f'{subject_path}\Subject{subjectNumber}_event{event_Number}_results_V4（csp）.txt'
#         
#             with open(result_file_name_2, 'w') as result_file:
#                 result_file.write(f'BA: {ba}\n')
#                 result_file.write(f'ACC: {acc}\n')
#                 result_file.write(f'TPR: {tpr}\n')
#                 result_file.write(f'FPR: {fpr}\n')
# =============================================================================