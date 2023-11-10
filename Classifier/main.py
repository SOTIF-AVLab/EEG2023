import os

from XGBDIM import XGBDIM
from CSP import CSPClassifier
import numpy as np
import pickle

'''
(data_path, sub_idx, trainset, validationset,
model_path,
n_cutpoint, win_len, chan_xlen, chan_ylen, step_x, step_y,
eta_global, eta_local, alpha_global, alpha_local, Nb, N_epoch, C1, C0, max_N_sub_model, gstf_weight, 
validation_flag, validation_step, crossentropy_flag, random_downsampling)
'''
'''
Note:   the validation set in this study is used as the test set 
        for the purpose of comparing the performance of different number of sub-models.
        
        It is suggested that 
        the etas and alphas of global and local models are set in the range of [0.1, 0.5] and [0.01, 0.05] respectively.
        The choice will rarely affect the performance.
        
        The batch size is set to 100 in this study.
        The number of epochs is suggested to be set around 20. Too many epochs can cause the gradient to deplete prematurely.
        
        The number of groups (N_multiple) is set to 20 in this study. You can try more groups to see if it improves the performance, if your
        GPU is strong.
        N_multiple = 1 means XGB-DIM, not multi-XGB-DIM.
'''
# 改这个路径，建新的文件
eventNumber = [1,2,7,8,13,14]
# subject_number = [1,2,3,4,5,6,7,8,10,11,12]
subject_number = [13,14]#[1,2,3,4,5,6,7,8,10,11,12]
current_dir = os.getcwd()

for subjectNumber in subject_number:

    
    subject_path = f'{current_dir}\..\Data\subject{subjectNumber}'
    for event_Number in eventNumber:
        event_path = f'{subject_path}\event{event_Number}'
        
        csp = CSPClassifier(event_path, subjectNumber, 1)
        csp_label = csp.classification()
        
        model_path = f'{current_dir}\Subject{subjectNumber}_event{event_Number}'
        os.makedirs(model_path, exist_ok=True)
        
        xgb = XGBDIM(event_path, 1, np.array([1]), np.array([1]),
                 model_path,
                 50, 6, 3, 3, 3, 3,
                 0.5, 0.1, 0.01, 0.05, 100, 20, 1, 1, 59, 0.3, True, 30, True, True, N_multiple = 50)

        xgb.train_model()

# p1 = pickle.dumps(xgb)
# with open('xgb.pkl', 'wb') as f:
#     pickle.dump(xgb, f)
#如果 test set不止一个, 修改这个np.array([2,3,4])对应sub_1_2,3,4 (如果没记错 咳咳
        ba, acc, tpr, fpr, auc, ba1, acc1, tpr1, fpr1, auc1 = xgb.test(np.array([2,3,4]), 1, False, csp_label)
        # Create a filename for the text file
        result_file_name = f'{subject_path}\Subject{subjectNumber}_event{event_Number}_results_V4（csp+xgb）.txt'
        result_file_name_2 = f'{subject_path}\Subject{subjectNumber}_event{event_Number}_results_V4（xgb）.txt'

        # Write the results to the text file
        with open(result_file_name_2, 'w') as result_file:
            result_file.write(f'BA: {ba1}\n')
            result_file.write(f'ACC: {acc1}\n')
            result_file.write(f'TPR: {tpr1}\n')
            result_file.write(f'FPR: {fpr1}\n')
            result_file.write(f'AUC: {auc1}\n')
        # Write the results to the text file
        with open(result_file_name, 'w') as result_file:
            result_file.write(f'BA: {ba}\n')
            result_file.write(f'ACC: {acc}\n')
            result_file.write(f'TPR: {tpr}\n')
            result_file.write(f'FPR: {fpr}\n')
            result_file.write(f'AUC: {auc}\n')
            
            
        print('-----XGB------BA %f ACC %f TPR %f FPR %f AUC %f' % (ba1, acc1, tpr1, fpr1, auc1))
        print('--------------BA %f ACC %f TPR %f FPR %f AUC %f' % (ba, acc, tpr, fpr, auc))
        
