'''
XGBDIM.py
Author: Bowen Li, Tsinghua University
'''
import numpy as np
import os
import h5py
import torch as tc
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as metrics_auc
import matplotlib.pyplot as plt
# tc.autograd.set_detect_anomaly(True)
'''
data_path: the path of the data
sub_idx: the index of the subject
trainset: the index of the training set, example: np.array([1,2,3])
validationset: the index of the validation set, example: np.array([4,5,6])
model_path: the path of the model
n_cutpoint: the number of cutpoints, the removed points are ones before img onset, example: 50 (200 ms for 250 Hz)
win_len: the length of the window
chan_xlen: the number of the channels in x direction
chan_ylen: the number of the channels in y direction
step_x: the step of the channels in x direction
step_y: the step of the channels in y direction
eta: the regulation coefficient of the L2 norm
alpha: learning rate
Nb: the number of P/N samples in one batch
N_epoch: the number of epochs
C1: the weight of the positive samples
C0: the weight of the negative samples
max_N_model: the maximum number of sub models
gstf_weight: the weight of the GSTF, example: 0.3
validation_flag: whether to use the validation set
validation_step: the step of the validation, example: 100, means that the validation is performed every 100 sub models
crossentropy_flag: whether to calculate the cross entropy loss
random_downsampling_flag: whether to use the random downsampling for the negative samples
'''
class model_GSTF(nn.Module):
    def __init__(self, N_chan, N_time):
        super(model_GSTF, self).__init__()
        self.W_global = nn.Parameter(0.01 * tc.randn(N_chan, 1))
        self.Q_global = nn.Parameter(0.01 * tc.randn(N_time, 1))
        self.b_global = nn.Parameter(tc.zeros(1).float())
        self.Gamma_global = nn.Parameter(tc.ones(1).float())
        self.Beta_global = nn.Parameter(tc.zeros(1).float())
        # self.h = 0

    def forward(self, x):
        # input x: (60, 250, batch_size) -> (batch_size, 250, 60)
        x = self.Gamma_global * x + self.Beta_global
        h = tc.bmm(x, self.W_global.repeat(x.shape[0], 1, 1))
        h = tc.mm(h.squeeze(-1), self.Q_global) + self.b_global
        s = tc.sigmoid(h)
        return s, h

class model_local(nn.Module):
    def __init__(self, T_local):
        super(model_local, self).__init__()
        self.w_local = nn.Parameter(0.01 * tc.randn(T_local, 1))
        self.b_local = nn.Parameter(tc.zeros(1).float())
        self.gamma_local = nn.Parameter(tc.ones(1).float())
        self.beta_local = nn.Parameter(tc.zeros(1).float())
        # self.fk = 0

    def forward(self, x):
        # input x: (batch_size, T_local)
        x = self.gamma_local * x + self.beta_local
        f = tc.mm(x, self.w_local) + self.b_local
        # self.fk = f
        return f

class loss_model_global(nn.Module):
    def __init__(self):
        super(loss_model_global, self).__init__()

    def forward(self, x, y):
        # return crossentropy loss
        # input x: (batch_size, 1)
        # input y: (batch_size, 1)
        return tc.mean(-y * tc.log(x + 1e-6) - (1 - y) * tc.log(1 - x + 1e-6))

class loss_model_local(nn.Module):
    def __init__(self, G_k, H_k):
        super(loss_model_local, self).__init__()
        self.G_k = G_k
        self.H_k = H_k

    def forward(self, x, index):
        # input x: (batch_size, 1)
        # G_k: (batch_size, 1)
        # H_k: (batch_size, 1)
        return tc.mean(self.G_k[index] * x + 0.5 * self.H_k[index] * (x ** 2))

class Dataset_global(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.indices = list(range(x.shape[0]))

    def __getitem__(self, index):
        # Get a single sample from the dataset
        x = self.x[index, :, :]
        label = self.y[index]
        indices = self.indices[index]
        # Convert to tensors and return
        # x = tc.FloatTensor(x)
        # label = tc.LongTensor(label)
        return x, label, indices

    def __len__(self):
        # Return the size of the dataset
        return len(self.x)

class Dataset_local(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.indices = list(range(x.shape[0]))

    def __getitem__(self, index):
        # Get a single sample from the dataset
        x = self.x[index]
        label = self.y[index]
        indices = self.indices[index]
        # Convert to tensors and return
        # x = tc.FloatTensor(x)
        # label = tc.LongTensor(label)
        return x, label, indices

    def __len__(self):
        # Return the size of the dataset
        return len(self.x)
'''
dataset = MyDataset(X, y)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
'''
class XGBDIM():

    def __init__(self, data_path, sub_idx, trainset, validationset,
                 model_path,
                 n_cutpoint, win_len, chan_xlen, chan_ylen, step_x, step_y,
                 eta_global, eta_local, alpha_global, alpha_local, Nb, N_epoch, C1, C0, max_N_model, gstf_weight,
                 validation_flag, validation_step, crossentropy_flag, random_downsampling_flag, N_multiple):

        self.data_path = data_path
        self.model_path = model_path

        self.trainset = trainset
        self.validationset = validationset
        self.sub_idx = sub_idx

        self.win_len = win_len
        self.chan_xlen = chan_xlen
        self.chan_ylen = chan_ylen
        self.chan_len = chan_xlen * chan_ylen
        self.step_x = step_x
        self.step_y = step_y
        self.T_local = win_len * self.chan_len
        self.N_multiple = N_multiple
        self.N_multiple_save = N_multiple

        self.eta_global = eta_global
        self.eta_local = eta_local
        self.alpha_global = alpha_global
        self.alpha_local = alpha_local
        self.Nb = Nb
        self.N_epoch = N_epoch
        self.C1 = C1
        self.C0 = C0
        self.max_N_model = max_N_model
        self.gstf_weight = gstf_weight
        self.validation_flag = validation_flag
        self.validation_step = validation_step
        self.crossentropy_flag = crossentropy_flag
        self.random_downsampling_flag = random_downsampling_flag
        temp = np.array(range(1, 55))
        self.channel_loc = np.array([temp[0:9], temp[9:18], temp[18:27], temp[27:36], temp[36:45], temp[45:54]])

        self.n_cutpoint = n_cutpoint # number of points before stimulus
        self.beta1 = 0.9
        self.beta2 = 0.999

    def get_3Dconv(self):
        Nx_channel = np.shape(self.channel_loc)[1]
        channel_xst = np.arange(0, Nx_channel - self.chan_xlen + 1, self.step_x)

        Ny_channel = np.shape(self.channel_loc)[0]
        channel_yst = np.arange(0, Ny_channel - self.chan_ylen + 1, self.step_y)

        self.chan_len = self.chan_xlen * self.chan_ylen

        idx_conv = 0
        channel_conv = []
        for idx_y in channel_yst:
            for idx_x in channel_xst:
                idx_conv = idx_conv + 1
                cup = self.channel_loc[idx_y: idx_y + self.chan_ylen, idx_x: idx_x + self.chan_xlen]
                channel_conv.append(cup.T.reshape(-1))
        self.channel_conv = np.array(channel_conv)

        self.channel = np.array(list(range(6, 54)) + [58, 54, 60] + list(range(55, 58)))

        step_win = int(self.win_len/2)# round(win_len/2)
        self.window_st = np.array(range(1, 250 - self.win_len + 1, step_win))
        self.window_ov = self.window_st + self.win_len - 1

        self.N_model = np.shape(self.window_st)[0] * np.shape(self.channel_conv)[0]
        self.N_win = np.shape(self.window_st)[0]
        self.N_chanwin = np.shape(self.channel_conv)[0]

        self.N_conv = self.N_win * self.N_chanwin
        self.N_model = min([self.N_conv, self.max_N_model])

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
                    All_X1 = X1[:, 450:, :].copy()
                    All_X2 = X2[:, 450:, :].copy()
                    All_X3 = X3[:, 450:, :].copy()
                else:
                    if len(X1.shape)==2:
                        X1 = X1[:,:,np.newaxis]
                    else:
                        All_X1 = np.concatenate((All_X1, X1[:, 450:, :]), axis=2)
                    if len(X2.shape)==2:
                        X2 = X2[:,:,np.newaxis]
                    else:
                        All_X2 = np.concatenate((All_X2, X2[:, 450:, :]), axis=2)
                    if len(X3.shape)==2:
                        X3 = X3[:,:,np.newaxis]
                    else:
                        All_X3 = np.concatenate((All_X3, X3[:, 450:, :]), axis=2)
                    
                    # concatenate X2 and X3
                    
            All_X2 = np.concatenate((All_X3, All_X2), axis=2)
            
            X1 = All_X1.copy()
            del All_X1
            X2 = All_X2.copy()
            del All_X2
            K1 = np.shape(X1)[2]
            K2 = np.shape(X2)[2]
            print('Importing data finished')
            return X1, X2, K1, K2
        else:
            print('Dataset is None')
            exit(1)

    def get_3D_cuboids(self, X1, X2, K1, K2): # torch.masked_select(input,mask) is FASTER?
        Tset = np.zeros((K1, self.T_local, self.N_chanwin * self.N_win))
        NTset = np.zeros((K2, self.T_local, self.N_chanwin * self.N_win))
        idx_conv = -1
        for idx_chan in range(self.N_chanwin):
            for idx_win in range(self.N_win):
                idx_conv += 1
                for k in range(K1):
                    cup = np.squeeze(
                        X1[self.channel[self.channel_conv[idx_chan, :] - 1] - 1,
                        self.window_st[idx_win] - 1:self.window_ov[idx_win],
                        k])
                    Tset[k, :, idx_conv] = cup.T.flatten()
                for k in range(K2):
                    cup = np.squeeze(
                        X2[self.channel[self.channel_conv[idx_chan, :] - 1] - 1,
                        self.window_st[idx_win] - 1:self.window_ov[idx_win],
                        k])
                    NTset[k, :, idx_conv] = cup.T.flatten()
        return Tset, NTset

    def preprocess(self, X1, X2, K1, K2):
        X1 = np.delete(X1, list(range(self.n_cutpoint)), axis=1)
        X2 = np.delete(X2, list(range(self.n_cutpoint)), axis=1)
        for k in range(K1):
            X1[:, :, k] = X1[:, :, k] - np.mean(X1[:, :, k], axis=1)[:, np.newaxis]
        for k in range(K2):
            X2[:, :, k] = X2[:, :, k] - np.mean(X2[:, :, k], axis=1)[:, np.newaxis]
        print('EEG processed !')
        return X1, X2

    def get_data(self, dataset):
        start_time = time.time()
        X1, X2, K1, K2 = self.read_data(dataset)
        # X1 = tc.tensor(X1)
        # X2 = tc.tensor(X2)
        X1, X2 = self.preprocess(X1, X2, K1, K2)
        print('Time cost: ', time.time() - start_time)
        start_time = time.time()
        Tset, NTset = self.get_3D_cuboids(X1, X2, K1, K2)

        Tset_global = X1.copy()
        NTset_global = X2.copy()
        print('EEG reshaped !')
        print('Time cost: ', time.time() - start_time)
        return Tset, NTset, Tset_global, NTset_global, K1, K2

    def get_order_step(self):
        print('Learning the order and step of sub models')
        if self.N_model > 1:
            M1 = np.mean(self.Tset_train, 0).T
            M2 = np.mean(self.NTset_train, 0).T
            distance_0_1 = np.zeros(self.N_conv)
            for idx_conv in range(self.N_conv):
                R1 = 0
                for k in range(self.K1):
                    R1 = R1 + np.matrix(self.Tset_train[k, :, idx_conv] - M1[idx_conv, :]).T * np.matrix(
                        self.Tset_train[k, :, idx_conv] - M1[idx_conv, :])
                R1 = R1 / self.K1
                R2 = 0
                for k in range(self.K2):
                    R2 = R2 + np.matrix(self.NTset_train[k, :, idx_conv] - M2[idx_conv, :]).T * np.matrix(
                        self.NTset_train[k, :, idx_conv] - M2[idx_conv, :])
                R2 = R2 / self.K2
                distance_0_1[idx_conv] = np.trace(
                    np.linalg.pinv(R1 + R2) @ np.matrix(M2[idx_conv, :] - M1[idx_conv, :]).T @ np.matrix(
                        M2[idx_conv, :] - M1[idx_conv, :]))
            dist = np.sort(distance_0_1)[::-1]
            self.I_sort = np.argsort(distance_0_1)[::-1]

            # print(I_sort)
            dist = dist / max(dist)
            weight_conv = -1 / (self.N_conv - 1) / (self.N_conv - 1) * np.multiply(np.arange(self.N_conv),
                                                                                   np.arange(self.N_conv)) + 1
            self.lr_model = 0.5 * np.multiply(weight_conv, dist)
            print('Learning the order and step of sub models finished!')
            # return I_sort, lr_model
        else:
            print('Please run the global model! This is not a ensemble model!')
            exit(1)

    def batchnormalize(self, X_local, m, sigma):
        Ns = X_local.shape[0]
        X_local_BN = (X_local - m.repeat(Ns, 1)) / tc.sqrt(sigma.repeat(Ns, 1))
        return X_local_BN

    def batchnormalize_global(self, X_global, M_global, Sigma_global):
        # [Ch, Te, Ns] = X_global.shape
        if X_global.dim == 2:
            X_global_BN = (X_global - M_global) / tc.sqrt(Sigma_global)
        else:
            X_global_BN = (X_global - M_global.unsqueeze(2)) / tc.sqrt(Sigma_global.unsqueeze(2))

        return X_global_BN

    def slidemean(self, x_old, x_now, n_old, n_now):
        x_update = (x_old * n_old + x_now * n_now) / (n_old + n_now)
        return x_update


    def decision_value(self, X_minibatch_BN, X_minibatch_BN_global, model_onegroup, Ns):
        # h = tc.zeros((Ns, 1)).cuda(0)
        N_model = len(model_onegroup)
        gstf = model_onegroup[0]
        s, h = gstf(X_minibatch_BN_global)
        s = s.detach()
        h = h.detach()
        if N_model > 1:
            h = self.gstf_weight * h

            for idx_model in range(1, N_model):
                local_model = model_onegroup[idx_model]
                f = local_model(X_minibatch_BN[:, :, idx_model - 1]).detach()
                h = h + self.lr_model[idx_model - 1, 0] * f

            s = tc.sigmoid(h)
            # print('decision value s:', s)
        return s, h

    def validation(self, model_all):
        N_model = len(model_all[0])
        Ns = self.X_validation_global.shape[2]
        s_mean = 0
        for idx_group in range(self.N_multiple):
            model_onegroup = model_all[idx_group]
            X_global_BN = self.batchnormalize_global(self.X_validation_global,
                                                     self.M_global[:, :, idx_group],
                                                     self.Sigma_global[:, :, idx_group])
            X_minibatch_BN = tc.zeros((Ns, self.T_local, N_model - 1)).float().cuda(0)
            for idx_conv in range(N_model - 1):
                X_minibatch_BN[:, :, idx_conv] = self.batchnormalize(self.X_validation[:, :, idx_conv],
                                                                     self.M_local[idx_conv, :, idx_group],
                                                                     self.Sigma[idx_conv, :, idx_group])
            s, h = self.decision_value(X_minibatch_BN, X_global_BN.permute(2, 1, 0), model_onegroup, Ns)
            s_mean = s_mean + s
        s_mean = s_mean / self.N_multiple
        idx_1 = tc.where(tc.eq(self.label_validation, 1))[0]
        idx_2 = tc.where(tc.eq(self.label_validation, 0))[0]

        y_predicted_final = s_mean.clone()
        y_predicted_final[s_mean >= 0.5] = 1
        y_predicted_final[s_mean < 0.5] = 0

        acc = (y_predicted_final == self.label_validation).sum() / y_predicted_final.shape[0]

        n_positive = self.label_validation.eq(1).sum()
        n_negative = self.label_validation.eq(0).sum()
        n_tp = (y_predicted_final.T * self.label_validation.T).sum()
        n_fp = ((y_predicted_final.T == 1) * (self.label_validation.T == 0)).sum()

        tpr = n_tp / n_positive
        fpr = n_fp / n_negative
        fpr_1, tpr_1, thresholds = roc_curve(self.label_validation.cpu().numpy(), s_mean.detach().cpu().numpy())
        auc = metrics_auc(fpr_1, tpr_1)

        return acc, tpr, fpr, auc

    def get_GH(self, X_global, X_local, label, model_group, idx_group):
        N_model = len(model_group)
        Ch, Te, Ns = X_global.shape
        # X_global_BN_1 = self.batchnormalize_global(X_global, self.M_global, self.Sigma_global)
        # X_minibatch_BN = np.zeros((Ns, self.T_local, N_model - 1))
        if N_model == 1:
            s, self.h_GH_temp[idx_group] = self.decision_value(0, X_global.permute(2, 1, 0), model_group, Ns)
            # s = 1 / (1 + tc.exp(-self.h_GH_temp))
            self.h_GH_temp[idx_group] = self.h_GH_temp[idx_group] * self.gstf_weight
        else:
            # X_minibatch_BN = self.batchnormalize(X_local[:, :, N_model-2], self.M_local[N_model-2, :], self.Sigma[N_model-2, :])
            local_model = model_group[-1]
            f_k = local_model(X_local)
            # f_k = local_model.fk
            # plt.figure()
            # plt.title(N_model)
            # plt.plot(self.h_GH_temp.cpu().detach().numpy())
            self.h_GH_temp[idx_group] = self.h_GH_temp[idx_group] + self.lr_model[N_model-2, 0] * f_k
            # plt.plot(self.h_GH_temp.cpu().detach().numpy())
            # plt.plot(f_k.cpu().detach().numpy())
            # plt.show()

            s = tc.sigmoid(self.h_GH_temp[idx_group])
        G_k = (self.C1 - self.C0) * s * label - self.C1 * label + self.C0 * s
        H_k = ((self.C1 - self.C0) * label + self.C0) * s * (1 - s)

        # self.h_T_temp = self.h_GH_temp[label == 1].view(tc.sum(label == 1), 1)
        # self.h_N_temp = self.h_GH_temp[label == 0].view(tc.sum(label == 0), 1)
        # G_k_T = G_k[label == 1].view(tc.sum(label == 1), 1)
        # H_k_T = H_k[label == 1].view(tc.sum(label == 1), 1)
        # G_k_N = G_k[label == 0].view(tc.sum(label == 0), 1)
        # H_k_N = H_k[label == 0].view(tc.sum(label == 0), 1)
        return G_k, H_k

    def update_BN_global(self, X_train_global_BN, idx_batch, idx_group):

        m_old = self.M_global[:, :, idx_group].clone()
        sigma_old = self.Sigma_global[:, :, idx_group].clone()

        m_now = tc.mean(X_train_global_BN, dim=2)  # batch mean  #check
        idx_BN = idx_batch
        self.M_global[:, :, idx_group] = self.slidemean(m_old, m_now, idx_BN * self.N_batch, self.N_batch)
        sigma_now = tc.mean((X_train_global_BN - self.M_global[:, :, idx_group].unsqueeze(2)) ** 2, dim=2)
        self.Sigma_global[:, :, idx_group] = self.slidemean(sigma_old, sigma_now, idx_BN * self.N_batch, self.N_batch)  # update sigma

    def update_BN_local(self, X_train_local_BN, idx_batch, N_model, idx_group):
        idx_BN = idx_batch
        m_old = self.M_local[N_model - 2, :, idx_group].clone()
        sigma_old = self.Sigma[N_model - 2, :, idx_group].clone()
        m_now = tc.mean(X_train_local_BN, dim = 0)  # batch mean
        self.M_local[N_model - 2, :, idx_group] = self.slidemean(m_old, m_now, idx_BN * self.N_batch, self.N_batch)  # update mean
        # m = self.M_local[N_model - 2, :, idx_group].clone().detach()

        sigma_now = tc.mean((X_train_local_BN - self.M_local[N_model - 2, :, idx_group]) ** 2, axis=0)
         # update sigma
        sigma = self.slidemean(sigma_old, sigma_now, idx_BN * self.N_batch,
                                                     self.N_batch)
        sigma[sigma == 0] = tc.mean(sigma)
        self.Sigma[N_model - 2, :, idx_group] = sigma.clone()


    def load_model(self):
        filename = 'Model_' + str(self.sub_idx) + '_MG_' + str(self.N_multiple_save) + '.pt'
        Model_load = tc.load(os.path.join(self.model_path, filename))
        model_all = Model_load['Model']
        Sigma_global = Model_load['Sigma_global']
        M_global = Model_load['M_global']
        Sigma_local = Model_load['Sigma_local']
        M_local = Model_load['M_local']
        # 'Model_order'
        # 'local_weight'
        # 'GSTF_weight'
        Model_order = Model_load['Model_order']
        self.lr_model = Model_load['local_weight']
        self.gstf_weight = Model_load['GSTF_weight']
        return model_all, Sigma_global, M_global, Sigma_local, M_local, Model_order

    def test(self, testset, N_group, continue_test, csp_label):
        try:
            self.Tset_test
        except:
            Tset_test_exists = False
            self.model_all, self.Sigma_global, self.M_global, self.Sigma_local, self.M_local, self.Model_order = self.load_model()
            self.Sigma_global, self.M_global, self.Sigma_local, self.M_local, self.Model_order = \
                self.Sigma_global.cuda(0), self.M_global.cuda(0), self.Sigma_local.cuda(0), self.M_local.cuda(0), self.Model_order.cuda(0)
            for i in range(len(self.model_all)):
                for j in range(len(self.model_all[0])):
                    self.model_all[i][j] = self.model_all[i][j].cuda(0)
            # print('Model of Subject %d is loaded' % self.sub_idx)
            self.get_3Dconv()
            self.Tset_test, self.NTset_test, self.Tset_test_global, self.NTset_test_global, self.K1t, self.K2t = \
                self.get_data(testset)

            self.Tset_test = tc.from_numpy(self.Tset_test).float().cuda(0)
            self.NTset_test = tc.from_numpy(self.NTset_test).float().cuda(0)
            self.Tset_test_global = tc.from_numpy(self.Tset_test_global).float().cuda(0)
            self.NTset_test_global = tc.from_numpy(self.NTset_test_global).float().cuda(0)

            self.N_local_model = len(self.model_all[0]) - 1
            self.X_test = tc.cat((self.Tset_test, self.NTset_test), dim=0)
            self.X_test_global = tc.cat((self.Tset_test_global, self.NTset_test_global), dim=2)
            self.label_test = tc.cat((tc.ones((self.K1t, 1)), tc.zeros((self.K2t, 1))), dim=0).cuda(0)
            self.X_test = self.X_test[:, :, self.Model_order[:self.N_local_model]]
            print('Samples obtained!')
        else:
            Tset_test_exists = True
            print('Continue test!')



        N_group = np.min([N_group, len(self.model_all)])

        Ns = self.X_test_global.shape[2]
      
        offset = -0.45 * abs(csp_label - 1)
        s_mean = 0
        for idx_group in range(N_group):
            model_onegroup = self.model_all[idx_group]
            X_global_BN = self.batchnormalize_global(self.X_test_global,
                                                     self.M_global[:, :, idx_group],
                                                     self.Sigma_global[:, :, idx_group])
            X_minibatch_BN = tc.zeros((Ns, self.T_local, self.N_local_model)).float().cuda(0)
            for idx_local in range(self.N_local_model):
                X_minibatch_BN[:, :, idx_local] = self.batchnormalize(self.X_test[:, :, idx_local],
                                                                     self.M_local[idx_local, :, idx_group],
                                                                     self.Sigma_local[idx_local, :, idx_group])
            s, h = self.decision_value(X_minibatch_BN, X_global_BN.permute(2, 1, 0), model_onegroup, Ns)
            s_mean = s_mean + s
        s_mean = s_mean / N_group
    
        idx_1 = tc.where(tc.eq(self.label_test, 1))[0]
        idx_2 = tc.where(tc.eq(self.label_test, 0))[0]

        y_predicted_final = s_mean.clone()
        y_predicted_final[s_mean >= 0.5] = 1
        y_predicted_final[s_mean < 0.5] = 0

        acc1 = (y_predicted_final == self.label_test).sum() / y_predicted_final.shape[0]

        n_positive = self.label_test.eq(1).sum()
        n_negative = self.label_test.eq(0).sum()
        n_tp = (y_predicted_final.T * self.label_test.T).sum()
        n_fp = ((y_predicted_final.T == 1) * (self.label_test.T == 0)).sum()

        tpr1 = n_tp / n_positive
        fpr1 = n_fp / n_negative
        fpr_1, tpr_1, thresholds = roc_curve(self.label_test.cpu().numpy(), s_mean.detach().cpu().numpy())
        auc1 = metrics_auc(fpr_1, tpr_1)
        ba1 = (tpr1 + (1 - fpr1)) / 2
        
        print("########CSP+XGB########")
        
        offset = tc.from_numpy(offset)
        offset_gpu = offset.to('cuda:0')
        offset_gpu = tc.unsqueeze(offset_gpu,1)
        s_mean = s_mean + offset_gpu
        idx_1 = tc.where(tc.eq(self.label_test, 1))[0]
        idx_2 = tc.where(tc.eq(self.label_test, 0))[0]

        y_predicted_final = s_mean.clone()
        y_predicted_final[s_mean >= 0.5] = 1
        y_predicted_final[s_mean < 0.5] = 0

        acc = (y_predicted_final == self.label_test).sum() / y_predicted_final.shape[0]

        n_positive = self.label_test.eq(1).sum()
        n_negative = self.label_test.eq(0).sum()
        n_tp = (y_predicted_final.T * self.label_test.T).sum()
        n_fp = ((y_predicted_final.T == 1) * (self.label_test.T == 0)).sum()

        tpr = n_tp / n_positive
        fpr = n_fp / n_negative
        fpr_1, tpr_1, thresholds = roc_curve(self.label_test.cpu().numpy(), s_mean.detach().cpu().numpy())
        auc = metrics_auc(fpr_1, tpr_1)
        ba = (tpr + (1 - fpr)) / 2

        # for i in range(len(model_all)):
        #     for j in range(len(model_all[0])):
        #         model_all[i][j] = model_all[i][j].detach().cpu()
        if continue_test:
            pass
        else:
            self.Tset_test = self.Tset_test.clone().detach().cpu()
            self.NTset_test = self.NTset_test.clone().detach().cpu()
            self.Tset_test_global = self.Tset_test_global.clone().detach().cpu()
            self.NTset_test_global = self.NTset_test_global.clone().detach().cpu()
            self.Sigma_global = self.Sigma_global.clone().detach().cpu()
            self.M_global = self.M_global.clone().detach().cpu()
            self.Sigma_local = self.Sigma_local.clone().detach().cpu()
            self.M_local = self.M_local.clone().detach().cpu()
            self.Model_order = self.Model_order.clone().detach().cpu()
            self.X_test = self.X_test.clone().detach().cpu()
            self.X_test_global = self.X_test_global.clone().detach().cpu()
            X_minibatch_BN = X_minibatch_BN.clone().detach().cpu()
            X_global_BN = X_global_BN.clone().detach().cpu()
            self.label_test = self.label_test.clone().detach().cpu()
            y_predicted_final = y_predicted_final.clone().detach().cpu()
            s = s.clone().detach().cpu()
            s_mean = s_mean.clone().detach().cpu()
        ba = ba.detach().cpu().numpy()
        acc = acc.detach().cpu().numpy()
        tpr = tpr.detach().cpu().numpy()
        fpr = fpr.detach().cpu().numpy()
        ba1 = ba1.detach().cpu().numpy()
        acc1 = acc1.detach().cpu().numpy()
        tpr1= tpr1.detach().cpu().numpy()
        fpr1 = fpr1.detach().cpu().numpy()
        tc.cuda.empty_cache()
        return ba, acc, tpr, fpr, auc,ba1,acc1,tpr1,fpr1,auc1

    def train_model(self):
        self.get_3Dconv()
        self.Tset_train, self.NTset_train, self.Tset_train_global, self.NTset_train_global, self.K1, self.K2 = \
            self.get_data(self.trainset)

        self.get_order_step()
        self.NTset_train = self.NTset_train[:, :, self.I_sort[:self.N_model]]
        self.Tset_train = self.Tset_train[:, :, self.I_sort[:self.N_model]]

        self.Tset_train = tc.from_numpy(self.Tset_train).float().cuda(0)
        self.NTset_train = tc.from_numpy(self.NTset_train).float().cuda(0)
        self.Tset_train_global = tc.from_numpy(self.Tset_train_global).float().cuda(0)
        self.NTset_train_global = tc.from_numpy(self.NTset_train_global).float().cuda(0)
        self.lr_model = tc.from_numpy(self.lr_model).view(-1,1).float().cuda(0)

        'Downsampling negative samples'
        idx_nontarget_train = list(range(self.K2))
        N_group = self.K2 // self.K1
        if self.random_downsampling_flag:
            indices = random.sample(idx_nontarget_train, N_group * self.K1)
            idx_selected_2 = [indices[(i*self.K1):(i*self.K1+self.K1)] for i in range(N_group)]
        else:
            idx_selected_2 =[idx_nontarget_train[i:self.K1*N_group:N_group] for i in range(N_group)]

        self.N_multiple = np.min([self.N_multiple, N_group])
        X_train_local_all = []
        X_train_global_all = []
        for idx_group in range(self.N_multiple):
            X_train_local_all.append(
                tc.cat((self.Tset_train, self.NTset_train[idx_selected_2[idx_group], :, :]), dim=0)
            )
            X_train_global_all.append(
                tc.cat((self.Tset_train_global, self.NTset_train_global[:, :, idx_selected_2[idx_group]]), dim=2)
            )


        if np.array_equal(self.trainset, self.validationset):
            Tset_validation = self.Tset_train.clone().detach()
            Tset_validation_global = self.Tset_train_global.clone().detach()
            NTset_validation = self.NTset_train.clone().detach()
            NTset_validation_global = self.NTset_train_global.clone().detach()
            self.K1v = self.K1
            self.K2v = self.K2
            self.I_sort = tc.from_numpy(self.I_sort.copy()).cuda(0)
            self.X_validation = tc.cat([Tset_validation, NTset_validation], dim=0)
        else:
            Tset_validation, NTset_validation, Tset_validation_global, NTset_validation_global, self.K1v, self.K2v = \
                self.get_data(self.validationset)

            Tset_validation = tc.from_numpy(Tset_validation).float().cuda(0)
            NTset_validation = tc.from_numpy(NTset_validation).float().cuda(0)
            Tset_validation_global = tc.from_numpy(Tset_validation_global).float().cuda(0)
            NTset_validation_global = tc.from_numpy(NTset_validation_global).float().cuda(0)
            self.I_sort = tc.from_numpy(self.I_sort.copy()).cuda(0)
            self.X_validation = tc.cat([Tset_validation, NTset_validation], dim=0)
            self.X_validation = self.X_validation[:, :, self.I_sort[:self.N_model]]

        self.X_validation_global = tc.cat([Tset_validation_global, NTset_validation_global], dim=2)

        self.Nb = np.min([self.Nb, self.K1])
        self.label_validation = tc.cat((tc.ones((self.K1v, 1)), tc.zeros((self.K2v, 1))), dim=0).float().cuda(0)

        self.K2 = self.K1
        self.N_batch = self.Nb * 2
        self.label_train = tc.cat([tc.ones((self.Nb, 1)), tc.zeros((self.Nb, 1))], dim=0).float().cuda(0)
        self.label_all = tc.cat((tc.ones((self.K1, 1)), tc.zeros((self.K2, 1))), dim=0).float().cuda(0)

        # X_train_global_all = tc.cat((self.Tset_train_global, self.NTset_train_global), dim=2)
        # X_train_local_all = tc.cat((self.Tset_train, self.NTset_train),dim=0)
        '''INITIALIZATION'''
        N_sp = 1
        N_te = 1

        self.Sigma_global = tc.zeros((self.Tset_train_global.shape[0], self.Tset_train_global.shape[1],
                                      self.N_multiple)).float().cuda(0)
        self.M_global = tc.zeros((self.Tset_train_global.shape[0], self.Tset_train_global.shape[1],
                                  self.N_multiple)).float().cuda(0)

        'local model'
        self.M_local = tc.zeros((self.N_model, self.T_local, self.N_multiple)).float().cuda(0)
        self.Sigma = tc.zeros((self.N_model, self.T_local, self.N_multiple)).float().cuda(0)

        '''
        XGB-DIM
        '''
        model_all = [[] for i in range(self.N_multiple)]
        idx_model = 0
        X_train_local_BN_last_model = [0] * self.N_multiple
        X_train_global_BN = [0] * self.N_multiple
        self.h_GH_temp = [0] * self.N_multiple

        crossentropy = loss_model_global() #nn.CrossEntropyLoss()
        momentum = list(np.linspace(0.5, 0.99, num=self.N_epoch, endpoint=True))
        for idx_conv in range(self.N_model + 1):
            if idx_conv == 0:
                start_time = time.time()
                idx_model += 1  #
                for idx_group in range(self.N_multiple):
                    X_train_global_group = X_train_global_all[idx_group]
                    self.update_BN_global(X_train_global_group, 0, idx_group)
                    X_train_global_BN[idx_group] = self.batchnormalize_global(X_train_global_group,
                                                                   self.M_global[:, :, idx_group],
                                                                   self.Sigma_global[:, :, idx_group])
                    dataset_global = Dataset_global(X_train_global_BN[idx_group].permute(2, 1, 0), self.label_all)
                    dataloader_global = DataLoader(dataset_global, batch_size=int(self.N_batch), shuffle=True)
                    model= model_GSTF(self.Tset_train_global.shape[0], self.Tset_train_global.shape[1]).cuda(0)
                    criterion = loss_model_global()
                    optimizer_global = optim.Adam(model.parameters(), lr=self.alpha_global)

                    for idx_epoch in range(self.N_epoch):
                        # optimizer_global.momentum = momentum[idx_epoch]
                        for idx_iteration, (data, target, indices) in enumerate(dataloader_global):
                            optimizer_global.zero_grad()
                            outputs, _ = model(data) #X_train_global_BN, idx_batch
                            loss = criterion(outputs.view(-1, 1), target.view(-1, 1)) \
                                   + self.eta_global * model.W_global.norm(2)**2 + self.eta_global * model.Q_global.norm(2)**2
                            # with tc.autograd.detect_anomaly():
                            loss.backward()
                            optimizer_global.step()
                        # print('Group', idx_group, ' -- Epoch: ', idx_epoch+1, '-- GSTF loss: ', loss.item())  #, end='\t'
                    with tc.no_grad():
                        model_all[idx_group].append(model)
                        # print('GSTF time cost: ', time.time() - start_time)
                        # print('Subject %d Model %d/300 done !' % (self.sub_idx, idx_model))
                # validation
                if self.validation_flag:
                    Accvalidation, tpr, fpr, auc = self.validation(model_all)
                    print('--------------ACC %f TPR %f FPR %f AUC %f' % (Accvalidation, tpr, fpr, auc))
                    tc.cuda.empty_cache()
            else:  #
                start_time = time.time()
                idx_model = idx_model + 1
                for idx_group in range(self.N_multiple):
                    X_train_local_group = X_train_local_all[idx_group]
                    self.update_BN_local(X_train_local_group[:, :, idx_model-2], 0, idx_model, idx_group)
                    X_train_local_BN = self.batchnormalize(X_train_local_group[:, :, idx_model-2],
                                                           self.M_local[idx_model-2, :, idx_group],
                                                           self.Sigma[idx_model-2, :, idx_group])

                    dataset_local = Dataset_local(X_train_local_BN, self.label_all)

                    G_k, H_k = self.get_GH(X_train_global_BN[idx_group], X_train_local_BN_last_model[idx_group],
                                           self.label_all, model_all[idx_group], idx_group)
                    X_train_local_BN_last_model[idx_group] = X_train_local_BN.clone().detach()

                    if self.crossentropy_flag:
                        s_temp = tc.sigmoid(self.h_GH_temp[idx_group])
                        cross_entropy_temp = crossentropy(s_temp.view(1, -1), self.label_all.view(1, -1))
                        # print('Cross_entropy: ', cross_entropy_temp.item())
                    dataloader_local = DataLoader(dataset_local, batch_size=int(self.N_batch), shuffle=True)

                    model = model_local(self.T_local).cuda(0)
                    optimizer_local = optim.Adam(model.parameters(), lr=self.alpha_local)
                    criterion = loss_model_local(G_k.clone().detach(), H_k.clone().detach())
                    for idx_epoch in range(self.N_epoch):
                        # optimizer_local.momentum = momentum[idx_epoch]
                        for idx_iteration, (data, target, indices) in enumerate(dataloader_local):
                            optimizer_local.zero_grad()
                            outputs_local = model(data)
                            loss_local = criterion(outputs_local, indices) + self.eta_local * model.w_local.norm(2)**2
                            # with tc.autograd.detect_anomaly():
                            loss_local.backward()
                            optimizer_local.step()
                        # print('Epoch: ', idx_epoch+1, '-- Local loss: ', loss_local.item(), end='\t')
                    # print('Subject %d Model %d/300 in Group %d done ! Time cost %f' % (self.sub_idx, idx_model, idx_group, time.time() - start_time))
                    with tc.no_grad():
                        model_all[idx_group].append(model)
            if self.validation_flag and idx_model % self.validation_step == 0 and idx_model != 0:
                Accvalidation, tpr, fpr, auc = self.validation(model_all)
                tc.cuda.empty_cache()
                print('--------------ACC %f TPR %f FPR %f AUC %f' % (Accvalidation, tpr, fpr, auc))


        # for i in range(len(model_all)):
        #     for j in range(len(model_all[0])):
        #         model_all[i][j] = model_all[i][j].detach().cpu()
        self.Tset_train = self.Tset_train.clone().detach().cpu()
        self.NTset_train = self.NTset_train.clone().detach().cpu()
        self.Tset_train_global = self.Tset_train_global.clone().detach().cpu()
        self.NTset_train_global = self.NTset_train_global.clone().detach().cpu()
        self.lr_model = self.lr_model.clone().detach().cpu()

        Tset_validation = Tset_validation.clone().detach().cpu()
        NTset_validation = NTset_validation.clone().detach().cpu()
        Tset_validation_global = Tset_validation_global.clone().detach().cpu()
        NTset_validation_global = NTset_validation_global.clone().detach().cpu()

        self.I_sort = self.I_sort.clone().detach().cpu()
        self.X_validation = self.X_validation.clone().detach().cpu()
        self.X_validation_global = self.X_validation_global.clone().detach().cpu()
        self.label_validation = self.label_validation.clone().detach().cpu()
        self.label_train = self.label_train.clone().detach().cpu()
        self.label_all = self.label_all.clone().detach().cpu()

        self.Sigma_global = self.Sigma_global.clone().detach().cpu()
        self.M_global = self.M_global.clone().detach().cpu()
        self.M_local = self.M_local.clone().detach().cpu()
        self.Sigma = self.Sigma.clone().detach().cpu()
        # dataset_local = dataset_local.clone().detach().cpu()
        # dataset_global = dataset_global.clone().detach().cpu()

        for idx_group in range(self.N_multiple):
            self.h_GH_temp[idx_group] = self.h_GH_temp[idx_group].clone().detach().cpu()
            X_train_local_all[idx_group] = X_train_local_all[idx_group].clone().detach().cpu()
            X_train_global_all[idx_group] = X_train_global_all[idx_group].clone().detach().cpu()
            X_train_local_BN_last_model[idx_group] = X_train_local_BN_last_model[idx_group].clone().detach().cpu()
            X_train_global_BN[idx_group] = X_train_global_BN[idx_group].clone().detach().cpu()

        tc.cuda.empty_cache()
        filename = 'Model_' + str(self.sub_idx) + '_MG_' + str(self.N_multiple_save) + '.pt'
        tc.save({'Model': model_all,
                 'M_global': self.M_global,
                 'Sigma_global': self.Sigma_global,
                 'M_local': self.M_local,
                 'Sigma_local': self.Sigma,
                 'Model_order': self.I_sort,
                 'local_weight': self.lr_model,
                 'GSTF_weight': self.gstf_weight}, os.path.join(self.model_path, filename))
        # x = tc.load(os.path.join(self.model_path, filename))
        print('Model Training Finished !')
        # np.savez(os.path.join(self.model_path, filename), W_global=self.W_global, Q_global=self.Q_global, b_global=self.b_global,
        #          Gamma_global=self.Gamma_global, Beta_global=self.Beta_global, Sigma_global=self.Sigma_global, M_global=self.M_global,
        #          W_local=self.W_local, Gamma=self.Gamma, Beta=self.Beta, Sigma=self.Sigma, M_local=self.M_local, lr_model=self.lr_model,
        #          conv_sort=self.I_sort, Accvalidation_all=Accvalidation_all, tpr_all=tpr_all, fpr_all=fpr_all,
        #          auc_all=auc_all)



