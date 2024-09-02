from EEGModels import EEGNet, ShallowConvNet, DeepConvNet
import os
from CSP import CSP_Stacking
from sklearn.linear_model import LogisticRegression
from Data import Data
import numpy as np
from sklearn.metrics import balanced_accuracy_score,roc_curve,accuracy_score
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt



def custom_loss(class_weights):
    class_weights = K.variable(class_weights)
    
    def loss(y_true, y_pred):
        # Ensure predictions sum to 1 along the last axis (class probabilities)
        y_pred = K.softmax(y_pred, axis=-1)
        
        # Weighted cross-entropy calculation
        weighted_loss = - K.sum(class_weights * y_true * K.log(y_pred + K.epsilon()), axis=-1)
        
        return weighted_loss
    
    return loss

def evaluate_and_save(predicted_labels, y_test, subject_path, subject, name, continous = False):
    if continous:
        predicted_labels = np.argmax(predicted_labels, axis=1)
    ba = balanced_accuracy_score(y_test, predicted_labels)
    fpr,tpr,threshold = roc_curve(y_test, predicted_labels)
    acc = accuracy_score(y_test, predicted_labels)

    # plot the roc curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % ba)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    fig_path = os.path.join(f'{subject_path}/../../Output/Fig', f'all_events')
    os.makedirs(fig_path, exist_ok=True)
    roc_file_name = os.path.join(fig_path, f'Subject{subject}_roc_curve_{name}.png')
    plt.savefig(roc_file_name)

    # result_file_name = f'{subject_path}/../../Output/Result/Subject{subject}_event{event_Number}_results_{name}.txt'
    # with open(result_file_name, 'w') as result_file:
    #     result_file.write(f'BA: {ba}\n')
    #     result_file.write(f'TPR: {tpr}\n')
    #     result_file.write(f'FPR: {fpr}\n')
    return ba, tpr[1], fpr[1], acc

if __name__ == '__main__':

    # Set memory growth to True to release GPU memory allocated by TensorFlow
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # eventNumber = [8]
    subject_number = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]
    # subject_number = [1]
    current_dir = os.getcwd()

    # define model1 model2 model3 as EEGNet, ShallowConvNet and DeepConvNet
    model1 = EEGNet(nb_classes = 2, kernLength = 125, Chans = 60, Samples = 300)
    model2 = ShallowConvNet(nb_classes = 2, Chans = 60, Samples = 300)
    model3 = DeepConvNet(nb_classes = 2, Chans = 60, Samples = 300)
    csp_model = CSP_Stacking(nFilterPairs=1)

        
    dims = (6, 10, 4)
    acc = []
    
    for subject in subject_number:
        subject_path = f'{current_dir}/../Data1/subject{subject}'
        
        event_path = f'{subject_path}/events'
        data = Data(event_path, subject, 1)

        performance = np.empty(dims, dtype=float)
        

        for i in range(10): 
            x_train, y_train, x_test, y_test = data.prepare_data()

            """
            # part2 data for EEGNet, ShallowConvNet and DeepConvNet are [:, 450:, :]
            # part2 x data needs to be transpose (2, 0, 1)
            
            # part2 data treat label 1 as 1, label [2, 3] as 0
            # part2 y_train needs to be one-hot encoded
            # part1 data for CSP are [0, :250, 0]
            # part1 data treat label [1, 3] as 1, label 2 as 0
            """
            total_samples = np.concatenate([x_train,x_test],axis=2)
            x_train_part2 = x_train[:, 450:, :]

            x_train_part2 = np.transpose(x_train_part2, (2, 0, 1))
            x_test_part2 = x_test[:, 450:, :]
            x_test_part2 = np.transpose(x_test_part2, (2, 0, 1))

            y_train_part2 = (y_train == 1).astype(int)

            y_train_part2 = to_categorical(y_train_part2)

            y_test_part2 = (y_test == 1).astype(int)    
                
            x_train_part1 = x_train[:, :250, :]
            x_test_part1 = x_test[:, :250, :]
            
            y_train_part1 = (y_train != 2).astype(int)   
            y_test_part1 = (y_test != 2).astype(int)
            
            # Define the cost of misclassification

            # stacking model1 model2 model3
            #v2 parameters: 2 2.5 2.5
            #v3 parameters: 1.5 2 2
            #v4 parameters: 3 3 2.5
            #v5 parameters: 3 3 4
            #v6 parameters: 3 5 4
            model1.compile(loss=custom_loss(3), optimizer='adam', metrics=['accuracy'])
            model2.compile(loss=custom_loss(5), optimizer='adam', metrics=['accuracy'])
            model3.compile(loss=custom_loss(4), optimizer='adam', metrics=['accuracy'])
            # stacking model1 model2 model3

            # Train individual models
            model1.fit(x_train_part2, y_train_part2, epochs=50, validation_split=0.2)
            model2.fit(x_train_part2, y_train_part2, epochs=50, validation_split=0.2)
            model3.fit(x_train_part2, y_train_part2, epochs=50, validation_split=0.2)

            csp_model.fit(x_train_part1, y_train_part1)

            # get the training data from 
            meta_train1 = model1.predict(x_train_part2)
            meta_train2 = model2.predict(x_train_part2)
            meta_train3 = model3.predict(x_train_part2)

            [meta_train_csp_prob, meta_train_csp_label] = csp_model.predict(x_train_part1)
            # concatenate the predictions on the training data
            """TO DO: change scale of the input data to ensure all the classifier have the same scale"""
            meta_train_data_csp = np.vstack((meta_train1[:, 1], meta_train2[:, 1], meta_train3[:, 1], meta_train_csp_prob[:, 0]))
            meta_train_data_csp = np.transpose(meta_train_data_csp)
            meta_train_data = np.vstack((meta_train1[:, 1], meta_train2[:, 1], meta_train3[:, 1]))
            meta_train_data = np.transpose(meta_train_data)
            
            meta_learner = LogisticRegression()
            meta_learner_csp = LogisticRegression()

            y_train_part2 = np.argmax(y_train_part2, axis=1)
            meta_learner.fit(meta_train_data, y_train_part2)
            meta_learner_csp.fit(meta_train_data_csp, y_train_part2)

            coefficients = meta_learner.coef_
            print(coefficients)

            # get the test data from model1, model2, model3
            meta_test1 = model1.predict(x_test_part2)
            meta_test2 = model2.predict(x_test_part2)
            meta_test3 = model3.predict(x_test_part2)

            [meta_test_csp_prob, meta_test_csp_label] = csp_model.predict(x_test_part1)
            # concatenate the predictions on the test data
            meta_test_data_csp = np.vstack((meta_test1[:, 1], meta_test2[:, 1], meta_test3[:, 1], meta_test_csp_prob[:, 0]))
            meta_test_data_csp = np.transpose(meta_test_data_csp)
            meta_test_data = np.vstack((meta_test1[:, 1], meta_test2[:, 1], meta_test3[:, 1]))
            meta_test_data = np.transpose(meta_test_data)

            meta_predicted_labels = meta_learner.predict(meta_test_data)
            meta_predicted_labels_csp = meta_learner_csp.predict(meta_test_data_csp)

            # evaluate and save the result
            performance[0, i] = evaluate_and_save(meta_predicted_labels_csp, y_test_part2, subject_path, subject, 'Stacking(CSP)', False)
            performance[1, i] = evaluate_and_save(meta_predicted_labels, y_test_part2, subject_path, subject, 'Stacking', False)
            performance[2, i] = evaluate_and_save(meta_test1, y_test_part2, subject_path, subject, 'EEGNet', True)
            performance[3, i] = evaluate_and_save(meta_test2, y_test_part2, subject_path, subject, 'ShallowConvNet', True)
            performance[4, i] = evaluate_and_save(meta_test3, y_test_part2, subject_path, subject, 'DeepConvNet', True)
            performance[5, i] = evaluate_and_save(meta_test_csp_label, y_test_part1, subject_path, subject, 'CSP', False)
            
        acc.append(performance[:,:,0])       
            # meta_data_file_name = f'{subject_path}/Subject{subject}_event{event_Number}_meta_data_(Stacking).txt'
            # compare = np.vstack((y_test, meta_test_data[:, 0], meta_test_data[:, 1], meta_test_data[:, 2], meta_predicted_labels))
            # compare = np.transpose(compare)
            # np.savetxt(f'{subject_path}/Subject{subject}_event{event_Number}_meta_data_V4(Stacking).txt', compare, delimiter=',')
            
        # print("the average balanced accuracy is: " + str(np.mean(performance, axis=1)))
        
        # save the performance as csv
        performance_file_path = os.path.join(f'{subject_path}/../../Output/Result', f'Events')
        os.makedirs(performance_file_path, exist_ok=True)
        for idx in range(6): 
            performance_file_name = os.path.join(performance_file_path, f'Subject{subject}_model{str(idx)}_performance_v6.csv')
            np.savetxt(performance_file_name, performance[idx, :], delimiter=',')
        
        #acc1 = acc1.reshape((len(subject_number),10))
    # After the loop where you save performance results in CSV files
    # Create a figure and axes
    num_elements = len(acc)
    num_plots_per_element = acc[0].shape[0]
    num_data_per_plot = acc[0].shape[1]
    acc_trans = [np.transpose(array) for array in acc]
    fig, axes = plt.subplots(num_elements,1,figsize=(20,40))
    
    for i in range(num_elements):
        axes[i].boxplot(acc_trans[i][:])
        axes[i].set_title(f'Subject {i+1} all models BA')
        axes[i].set_ylabel('BA')

    
    fig_path = os.path.join(f'{subject_path}/../../Output/Fig', f'Events')
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(os.path.join(fig_path, "Accuracy_Boxplot_v6.png"))
    plt.show()