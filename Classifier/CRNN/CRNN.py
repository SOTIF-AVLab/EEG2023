from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Permute
from tensorflow.keras.layers import Reshape, Concatenate
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Conv3D, AveragePooling3D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

# define customised activation functions, inspired by FBCSP
def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))

# define the slicing function, used to slice the input data based on the task and time window, e.g. Danger Identification, Risk Prediction. 
def slice(x, start_idx, end_idx):
    return x[:, :, start_idx:end_idx, :]

# define the CNN part of the CRNN model, will be used twice in the final model to extract features from two different time windows
def CNN_part(input_main, start_idx, end_idx, feature, nb_classes, Chans, Samples, dropoutRate, 
             conv1_kernel_3d, conv1_stride_3d, conv1_kernel_num,
             conv2_kernel_3d, conv2_stride_3d, conv2_kernel_num,
             conv3_kernel_3d, conv3_stride_3d, conv3_kernel_num,
             pool_kernel_3d, pool_stride_3d):
    
    # input_main = Input((Chans, Samples, 1))
    input = Lambda(slice, arguments={'start_idx' : start_idx, 'end_idx' : end_idx})(input_main)

    # reshape each input into 3D tensor, based on the location of the electrodes
    reshape1 = Reshape((6, 9, Samples, 1))(input)

    # 3D convolution to extract local spatial features and local temporal features
    conv3d1 = Conv3D(conv1_kernel_num, conv1_kernel_3d, strides = conv1_stride_3d, padding = 'valid', kernel_constraint = max_norm(2., axis=(0,1,2,3)))(reshape1)
    conv3d2 = Conv3D(conv2_kernel_num, conv2_kernel_3d, strides = conv2_stride_3d, padding = 'valid', kernel_constraint = max_norm(2., axis=(0,1,2,3)))(conv3d1)
    bn1 = BatchNormalization()(conv3d2)
    act1 = Activation(square)(bn1)
    pool3d = AveragePooling3D(pool_kernel_3d, strides = pool_stride_3d)(act1)
    act2 = Activation(log)(pool3d)

    # 3D convolution to extract global features
    conv3d3 = Conv3D(conv3_kernel_num, conv3_kernel_3d, strides = conv3_stride_3d, padding = 'valid', kernel_constraint = max_norm(2., axis=(0,1,2,3)))(act2)
    act3 = Activation('elu')(conv3d3)
    act3 = Permute((3, 1, 2, 4))(act3)

    # flatten the data, reshape encoded data into 1D tensor to fit RNN part
    flatten1 = Flatten()(act3)
    reshape2 = Reshape((-1, feature))(flatten1)

    drop1 = Dropout(dropoutRate)(flatten1)
    dense = Dense(nb_classes, kernel_constraint = max_norm(0.5))(drop1)
    output = Activation('softmax')(dense)

    return reshape2, Model(inputs = input_main, outputs = output)

def EEG_CRNN_Merge(nb_classes_part1 = 2, nb_classes_part2 = 2, Chans = 54, Samples = 750, dropoutRate = 0.5, 
                   global_feature = 32, GRU_num = 16):
    
    input_main = Input((Chans, Samples, 1))

    # Part1: Risk Prediction
    reshape2_part1, model_part1 = CNN_part(input_main, start_idx = 0, end_idx = 375, dropoutRate = dropoutRate, feature = global_feature, 
                                           nb_classes = nb_classes_part1, Chans = Chans, Samples = 375,
                                            conv1_kernel_3d = (3, 3, 1), conv1_stride_3d = (3, 3, 1), conv1_kernel_num = 32,
                                            conv2_kernel_3d = (1, 1, 25), conv2_stride_3d = (1, 1, 1), conv2_kernel_num = 32,
                                            conv3_kernel_3d = (2, 3, 1), conv3_stride_3d = (1, 1, 1), conv3_kernel_num = global_feature,
                                            pool_kernel_3d = (1, 1, 75), pool_stride_3d = (1, 1, 15))
    
    # Part2: Danger Identification
    reshape2_part2, model_part2 = CNN_part(input_main, start_idx = 450, end_idx = 750, dropoutRate = dropoutRate, feature = global_feature, 
                                           nb_classes = nb_classes_part2, Chans = Chans, Samples = 300,
                                            conv1_kernel_3d = (3, 3, 1), conv1_stride_3d = (1, 1, 1), conv1_kernel_num = 32,
                                            conv2_kernel_3d = (1, 1, 13), conv2_stride_3d = (1, 1, 1), conv2_kernel_num = 32,
                                            conv3_kernel_3d = (4, 7, 1), conv3_stride_3d = (1, 1, 1), conv3_kernel_num = global_feature,
                                            pool_kernel_3d = (1, 1, 50), pool_stride_3d = (1, 1, 50))

    #merge data from part1 and part2 
    merged = Concatenate(axis=1)([reshape2_part1, reshape2_part2])
    

    # RNN part, using GRU
    gru = GRU(GRU_num, return_sequences = True)(merged)

    # Dropout
    drop1 = Dropout(dropoutRate)(gru)

    # Flatten the data
    flatten2 = Flatten()(drop1)

    # Fully connected layer
    dense = Dense(nb_classes_part2, kernel_constraint = max_norm(0.5))(flatten2)

    # Softmax activation function
    softmax = Activation('softmax')(dense)

    return model_part1, model_part2, Model(inputs=input_main, outputs=softmax)


    
# legacy version, not used in the final model
# def EEG_CRNN_legacy(nb_classes = 2, Chans = 54, Samples = 250, dropoutRate = 0.5, 
#              conv1_kernel_3d = (3, 3, 1), conv1_stride_3d = (3, 3, 1), conv1_kernel_num = 32,
#              conv2_kernel_3d = (1, 1, 25), conv2_stride_3d = (1, 1, 1), conv2_kernel_num = 32,
#              conv3_kernel_3d = (2, 3, 1), conv3_stride_3d = (1, 1, 1), conv3_kernel_num = 32,
#              pool_kernel_3d = (1, 1, 75), pool_stride_3d = (1, 1, 15)):

#     feature = 1 * 1 * conv1_kernel_num
    
#     input_main = Input((Chans, Samples, 1))

#     # reshape each input into 3D tensor, based on the location of the electrodes
#     # expected output : 6 * 9 * Samples
#     reshape1 = Reshape((6, 9, Samples, 1))(input_main)

#     # 3D convolution to extract local spatial features
#     conv3d1 = Conv3D(conv1_kernel_num, conv1_kernel_3d, strides = conv1_stride_3d, padding = 'valid', kernel_constraint = max_norm(2., axis=(0,1,2,3)))(reshape1)

#     # 3D convolution to extract temporal features
#     conv3d2 = Conv3D(conv2_kernel_num, conv2_kernel_3d, strides = conv2_stride_3d, padding = 'valid', kernel_constraint = max_norm(2., axis=(0,1,2,3)))(conv3d1)

#     # Batch normalization
#     bn1 = BatchNormalization()(conv3d2)

#     # Activation function
#     act1 = Activation(square)(bn1)

#     # 3D average pooling
#     pool3d = AveragePooling3D(pool_kernel_3d, strides = pool_stride_3d)(act1)

#     # Activation function
#     act2 = Activation(log)(pool3d)

#     # 3D convolution to extract global spatial features
#     conv3d3 = Conv3D(conv3_kernel_num, conv3_kernel_3d, strides = conv3_stride_3d, padding = 'valid', kernel_constraint = max_norm(2., axis=(0,1,2,3)))(act2)

#     # Activation function
#     act3 = Activation('elu')(conv3d3)

#     # Permute the data
#     act3 = Permute((3, 1, 2, 4))(act3)

#     # Flatten the data
#     flatten1 = Flatten()(act3)

#     # Reshape the data to fit RNN part, reshape based on the time dimension
#     reshape2 = Reshape((-1, feature))(flatten1)

#     # RNN part, using GRU
#     gru = GRU(32, return_sequences = True)(reshape2)

#     # Dropout
#     drop1 = Dropout(dropoutRate)(gru)

#     # Flatten the data
#     flatten2 = Flatten()(drop1)

#     # Fully connected layer
#     dense = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten2)

#     # Softmax activation function
#     softmax = Activation('softmax')(dense)

#     return Model(inputs=input_main, outputs=softmax)

# CNN part only, for test purpose only, not used in the final model
# def EEG_CRNN_CNNonly(nb_classes = 2, Chans = 54, Samples = 250, dropoutRate = 0.5, 
#              conv1_kernel_3d = (3, 3, 1), conv1_stride_3d = (3, 3, 1), conv1_kernel_num = 32,
#              conv2_kernel_3d = (1, 1, 50), conv2_stride_3d = (1, 1, 1), conv2_kernel_num = 32,
#              conv3_kernel_3d = (2, 3, 1), conv3_stride_3d = (1, 1, 1), conv3_kernel_num = 32,
#              pool_kernel_3d = (1, 1, 75), pool_stride_3d = (1, 1, 30)):

#     feature = 1 * 1 * conv1_kernel_num
    
#     input_main = Input((Chans, Samples, 1))

#     # reshape each input into 3D tensor, based on the location of the electrodes
#     # expected output : 6 * 9 * Samples
#     reshape1 = Reshape((6, 9, Samples, 1))(input_main)

#     # 3D convolution to extract local spatial features
#     conv3d1 = Conv3D(conv1_kernel_num, conv1_kernel_3d, strides = conv1_stride_3d, padding = 'valid', kernel_constraint = max_norm(2., axis=(0,1,2,3)))(reshape1)

#     # 3D convolution to extract temporal features
#     conv3d2 = Conv3D(conv2_kernel_num, conv2_kernel_3d, strides = conv2_stride_3d, padding = 'valid', kernel_constraint = max_norm(2., axis=(0,1,2,3)))(conv3d1)

#     # Batch normalization
#     bn1 = BatchNormalization()(conv3d2)

#     # Activation function
#     act1 = Activation(square)(bn1)

#     # 3D average pooling
#     pool3d = AveragePooling3D(pool_kernel_3d, strides = pool_stride_3d)(act1)

#     # Activation function
#     act2 = Activation(log)(pool3d)

#     # 3D convolution to extract global spatial features
#     conv3d3 = Conv3D(conv3_kernel_num, conv3_kernel_3d, strides = conv3_stride_3d, padding = 'valid', kernel_constraint = max_norm(2., axis=(0,1,2,3)))(act2)

#     # Activation function
#     act3 = Activation('elu')(conv3d3)

#     # Permute the data
#     act3 = Permute((3, 1, 2, 4))(act3)

#     # Dropout
#     drop1 = Dropout(dropoutRate)(act3)

#     # Flatten the data
#     flatten2 = Flatten()(drop1)

#     # Fully connected layer
#     dense = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten2)

#     # Softmax activation function
#     softmax = Activation('softmax')(dense)

#     return Model(inputs=input_main, outputs=softmax)