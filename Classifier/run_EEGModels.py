from EEGModels import EEGNet, ShallowConvNet, DeepConvNet
import numpy as np
import os
import h5py
import random
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

# 3000 datapoints downsampled to 750 for each trial
# 63 channels are used for each trial
# prepare the data for the model
class Data():
    def __init__(self, data_path, subject, sub_idx):
        self.training_set = None
        self.testing_set = None
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
                    ), 'r'
                )
                X1 = np.float32(np.transpose(data['X1']))
                X2 = np.float32(np.transpose(data['X2']))
                X3 = np.float32(np.transpose(data['X3']))
               
                if len(All_X1) == 0:
                    All_X1 = X1[:, :, :].copy()
                    All_X2 = X2[:, :, :].copy()
                    All_X3 = X3[:, :, :].copy()
                else:
                    if len(X1.shape)==2:
                        X1 = X1[:,:,np.newaxis]
                    else:
                        All_X1 = np.concatenate((All_X1, X1[:, :, :]), axis=2)
                    if len(X2.shape)==2:
                        X2 = X2[:,:,np.newaxis]
                    else:
                        All_X2 = np.concatenate((All_X2, X2[:, :, :]), axis=2)
                    if len(X3.shape)==2:
                        X3 = X3[:,:,np.newaxis]
                    else:
                        All_X3 = np.concatenate((All_X3, X3[:, :, :]), axis=2)
                    
                    # concatenate X2 and X3
                    
            All_X2 = np.concatenate((All_X3, All_X2), axis=2)
            
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
        return self.training_set
        
    def prepare_testing_data(self):
        idx = np.array([2, 3, 4])
        X1, X2 = self.read_data(idx)
        self.testing_set = self.label_data(X1, X2)
        return self.testing_set
    
    def get_data(self, dataset):        
        X1, X2 = self.read_data(dataset)

eventNumber = [1,2,7,8,13,14]
# subject_number = [1,2,3,4,5,6,7,8,10,11,12]
subject_number = [1] # [1,2,3,4,5,6,7,8,10,11,12]
current_dir = os.getcwd()

# define model1 as EEGNet, SharllowConvNet and DeepConvNet
model1 = EEGNet(nb_classes = 2, Chans = 60, Samples = 750)
model2 = ShallowConvNet(nb_classes = 2, Chans = 60, Samples = 750)
model3 = DeepConvNet(nb_classes = 2, Chans = 60, Samples = 750)


for subject in subject_number:
    subject_path = f'{current_dir}\..\Data\subject{subject}'
    for event_Number in eventNumber:
        event_path = f'{subject_path}\event{event_Number}'
        data = Data(event_path, subject, 1)

        training_set = data.prepare_training_data()
        testing_set = data.prepare_testing_data()

        train_data = [item[0] for item in training_set]
        x_train = np.stack(train_data, axis=2)
        train_label = [item[1] for item in training_set]
        y_train = np.stack(train_label)

        test_data = [item[0] for item in testing_set]
        x_test = np.stack(test_data, axis=2)
        test_label = [item[1] for item in testing_set]
        y_test = np.stack(test_label)

        # permute the data
        x_train = np.transpose(x_train, (2, 0, 1))
        x_test = np.transpose(x_test, (2, 0, 1))
        y_train = to_categorical(y_train)

        # change model to model1, model2, and model3 to try on different sota models 
        model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        fittedModel = model2.fit(x_train, y_train, epochs=100, validation_split=0.2)
        predicted = model2.predict(x_test)
        predicted_labels = np.argmax(predicted, axis=1)
        # Generate a classification report
        report = classification_report(y_test, predicted_labels)

        print(report)