import os
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.utils import resample
from sklearn.metrics import classification_report, balanced_accuracy_score, roc_curve

def weighted_categorical_crossentropy(weights):
    """
    Weighted categorical crossentropy loss function.
    """
    # Calculate the sum of the weights
    total = sum(weights)
    # Normalize the weights
    class_weights = [weight / total for weight in weights]
    
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # Scale predictions so that the class probabilities of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # Clip predictions to avoid log(0) or log(1) which would lead to NaNs
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # Calculate cross-entropy loss
        loss = K.cast(y_true, 'float32') * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss

# # Define class weights (replace with your actual class weights)
# class_weights = [1, 10]  # Example weights for two classes

# # Compile the model with the weighted categorical crossentropy loss
# model1.compile(loss=weighted_categorical_crossentropy(class_weights), optimizer='adam', metrics=['accuracy'])

def evaluate_and_save(predicted_labels, y_true, dir_path, model_name, subject):
    # evaluate the models
    y_pred = np.argmax(predicted_labels, axis=1)
    
    cr = classification_report(y_true, y_pred)

    # get balance accuracy
    ba = balanced_accuracy_score(y_true, y_pred)

    # get TRP, FPR
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    # save cr
    file_path = os.path.join(dir_path, f'subject{subject}_{model_name}_classification_report.txt')
    with open(file_path, 'w') as f:
        f.write(cr)
        f.write(f'\nBalance Accuracy: {ba}')
        f.write(f'\nTrue Positive Rate: {tpr[1]}')
        f.write(f'\nFalse Positive Rate: {fpr[1]}')
    f.close()
    file_path = os.path.join(dir_path, f'subject{subject}_{model_name}_plot_metrics.txt')
    with open(file_path, 'w') as f:
        f.write(f'{ba}, {tpr[1]}, {fpr[1]}')
    f.close()
    return cr, ba, tpr[1], fpr[1]

def resample_data(x_train, y_train):
    train_minority = x_train[y_train == 1]
    train_majority = x_train[y_train != 1]
    y_train_minority = y_train[y_train == 1]
    y_train_majority = y_train[y_train != 1]

    # upsample the minority class
    train_minority_upsampled = resample(train_minority, replace=True, n_samples=len(train_majority), random_state=123)
    y_train_minority_upsampled = resample(y_train_minority, replace=True, n_samples=len(train_majority), random_state=123)

    x_train = np.concatenate((train_majority, train_minority_upsampled))
    y_train = np.concatenate((y_train_majority, y_train_minority_upsampled))
    return x_train, y_train

def relabel_data(y_train, y_test):
    # Part 1: Risk Prediction              
    # Both high-risk and low-risk scenarios are considered as existence of potential risks (positive)
    y_train_part1 = (y_train != 2).astype(int)   
    y_test_part1 = (y_test != 2).astype(int)

    y_train_part1 = to_categorical(y_train_part1)

    # Whole: Danger Indentification
    y_train = (y_train == 1).astype(int)
    y_test = (y_test == 1).astype(int)

    y_train = to_categorical(y_train) 

    return y_train_part1, y_test_part1, y_train, y_test