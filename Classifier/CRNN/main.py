# import necessary python packages
import os
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle

# import our own packages
from CRNN import EEG_CRNN_Merge, log, square
from Data import Data
from utils import weighted_categorical_crossentropy, evaluate_and_save, resample_data, relabel_data

parser = argparse.ArgumentParser(description='Train EEG-CRNN Model')
parser.add_argument('--subject', type=int, default=1, help='subject number')
parser.add_argument('--event', type=str, default='all_event', help='event type')

parser.add_argument('--default_data_dir', action='store_true', help='use default data directory')
parser.add_argument('--no_default_data_dir', action='store_false', dest='default_data_dir', help='do not use default data directory')
parser.add_argument('--data_dir', type=str, default=None, help='data directory')

parser.add_argument('--default_output_dir', action='store_true', help='use default output directory')
parser.add_argument('--no_default_output_dir', action='store_false', dest='default_output_dir', help='do not use default output directory')
parser.add_argument('--output_dir', type=str, default=None, help='output directory')

parser.add_argument('--default_model_dir', action='store_true', help='use default model directory')
parser.add_argument('--no_default_model_dir', action='store_false', dest='default_model_dir', help='do not use default model directory')
parser.add_argument('--model_dir', type=str, default=None, help='model directory')

parser.add_argument('--resample', action='store_true', help='resample data, choose True when training on imbalanced data')
parser.add_argument('--no_resample', action='store_false', dest='resample', help='do not resample data')
parser.add_argument('--shuffle', action='store_true', help='shuffle data')
parser.add_argument('--no_shuffle', action='store_false', dest='shuffle', help='do not shuffle data')

parser.add_argument('--train_part2_final', action='store_true', help='train part 2 with final model')
parser.add_argument('--no_train_part2_final', action='store_false', dest='train_part2_final', help='do not train part 2 with final model')
parser.add_argument('--evaluate_part1', action='store_true', help='evaluate part 1')
parser.add_argument('--no_evaluate_part1', action='store_false', dest='evaluate_part1', help='do not evaluate part 1')
parser.add_argument('--evaluate_part2', action='store_true', help='evaluate part 2')
parser.add_argument('--no_evaluate_part2', action='store_false', dest='evaluate_part2', help='do not evaluate part 2')

parser.add_argument('--part1_loss_weights', type=float, default=[1, 1], help='loss weights for part 1')
parser.add_argument('--part2_loss_weights', type=float, default=[1, 10], help='loss weights for part 2')
parser.add_argument('--loss_weights', type=float, default=[1, 3], help='loss weight for the whole model')

args = parser.parse_args()

# get data path
if args.default_data_dir:
    data_path = f'../Data/subject{args.subject}/{args.event}'
else: 
    data_path = args.data_dir

# get output path
if args.default_output_dir:
    output_path = f'../Output/EEG_CRNN/subject{args.subject}/{args.event}'
else:
    output_path = args.output_dir

# get model path
if args.default_model_dir:
    model_path = f'../Model/EEG_CRNN/subject{args.subject}/{args.event}/CRNN.h5'
else:
    model_path = args.model_dir

if __name__ == "__main__":

    # create output directory
    os.makedirs(output_path, exist_ok=True)

    # load data
    data = Data(data_path, args.subject)
    x_train, y_train, x_test, y_test = data.prepare_data()

    x_train = np.transpose(x_train, (2, 0, 1))
    x_test = np.transpose(x_test, (2, 0, 1))

    # resample and shuffle data
    if args.resample:
        x_train, y_train = resample_data(x_train, y_train)
    if args.shuffle:
        x_train, y_train = shuffle(x_train, y_train) 

    # relabel data
    y_train_part1, y_test_part1, y_train, y_test_part2 = relabel_data(y_train, y_test)

    # load model
    CRNN_part1, CRNN_part2, CRNN = EEG_CRNN_Merge(nb_classes_part1 = 2, nb_classes_part2 = 2, Chans = 54, Samples = 750) 

    # define model checkpoint
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # -----------------------------------------Train Part 1-----------------------------------------
    
    class_weights = args.part1_loss_weights

    # Freeze all layers except CRNN_Part1
    for layer in CRNN.layers:
        layer.trainable = False
    for layer in CRNN_part1.layers:
        layer.trainable = True
    
    # train CRNN_Part1 here
    metrics = ['accuracy']
    CRNN_part1.compile(loss = weighted_categorical_crossentropy(class_weights), optimizer = 'adam', metrics = metrics)
    CRNN_part1.fit(x_train, y_train_part1, epochs = 30, validation_split = 0.3)

    # -----------------------------------------Train Part 2-----------------------------------------
   
    class_weights = args.part2_loss_weights
   
    # Freeze all layers except CRNN_Part2
    for layer in CRNN.layers:
        layer.trainable = False
    for layer in CRNN_part2.layers:
        layer.trainable = True

    # train CRNN_Part2 here
    metrics = ['accuracy']
    CRNN_part2.compile(loss = weighted_categorical_crossentropy(class_weights), optimizer = 'adam', metrics = metrics)
    CRNN_part2.fit(x_train, y_train, epochs = 50, validation_split = 0.3)

    # -----------------------------------------Train Whole-----------------------------------------
    
    class_weights = args.loss_weights

    # unfreeze RNN layers (Part 2 Layers)
    for layer in CRNN.layers:
        layer.trainable = True
    for layer in CRNN_part1.layers:
        layer.trainable = False
    for layer in CRNN_part2.layers:
        layer.trainable = args.train_part2_final
    # train the whole CRNN
    metrics = ['accuracy']
    CRNN.compile(loss = weighted_categorical_crossentropy(class_weights), optimizer = 'adam', metrics = metrics)
    CRNN.fit(x_train, y_train, epochs = 30, validation_split = 0.3, callbacks=[checkpoint])

    best_CRNN = load_model(model_path, custom_objects={'square': square, 'log': log, 'loss': weighted_categorical_crossentropy([1, 3])})
    
    # test the models
    # test 1: Risk Prediction
    if args.evaluate_part1:
        print("-----------------------EEG_CRNN_part1-----------------------")
        test_part1 = CRNN_part1.predict(x_test)       
        cr, ba, tpr, fpr = evaluate_and_save(test_part1, y_test_part1, output_path, "EEG_CRNN_part1", args.subject)
        print(cr)
    
    # test 2: Danger identification
    if args.evaluate_part2:
        print("-----------------------EEG_CRNN_part2-----------------------") 
        test_part2 = CRNN_part2.predict(x_test)      
        cr, ba, tpr, fpr = evaluate_and_save(test_part2, y_test, output_path, "EEG_CRNN_part2", args.subject)
        print(cr)

    test_crnn = best_CRNN.predict(x_test)
    print("-----------------------EEG_CRNN-----------------------")
    cr, ba, tpr, fpr = evaluate_and_save(test_crnn, y_test, output_path, "EEG_CRNN_Best", args.subject)
    print(cr)