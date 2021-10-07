import numpy as np
import scipy.sparse as sp
import torch
import csv

from torch.utils.data import Dataset
# from neural_fp import *
import math
import os
import scipy
from sklearn.utils import shuffle, resample
import pickle
#import openbabel
#import pybel
import operator
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.manifold import TSNE, Isomap, MDS, locally_linear_embedding, SpectralEmbedding




use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor

def data_filter(x_all, y_all, target, sizes, tasks, smile_list, size_cutoff=1000):
    idx_row = []
    for i in range(0, len(sizes)):
        if sizes[i] <= size_cutoff:
            idx_row.append(i)
    x_select = [x_all[i] for i in idx_row]
    y_select = [y_all[i] for i in idx_row]
    smile_select = [smile_list[i] for i in idx_row]

    idx_col = []
    for task in tasks:
        for i in range(0, len(target)):
            if task == target[i]:
                idx_col.append(i)
    y_task = [[each_list[i] for i in idx_col] for each_list in y_select]

    return(x_select, y_task, smile_select)

def normalize(mx):
    """Row-normalize sparse matrix"""
    mx_abs = np.absolute(mx)
    #rowsum = np.array(mx.sum(1))
    rowsum = np.array(mx_abs.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def feature_normalize(x_all):
    """Min Max Feature Scalling for Atom Feature Matrix"""
    feature_num = x_all[0][0].shape[1]
    feature_min_dic = {}
    feature_max_dic = {}
    for i in range(len(x_all)):
        afm = x_all[i][0]
        afm_min = afm.min(0)
        afm_max = afm.max(0)
        for j in range(feature_num):
            if j not in feature_max_dic.keys():
                feature_max_dic[j] = afm_max[j]
                feature_min_dic[j] = afm_min[j]
            else:
                if feature_max_dic[j] < afm_max[j]:
                    feature_max_dic[j] = afm_max[j]
                if feature_min_dic[j] > afm_min[j]:
                    feature_min_dic[j] = afm_min[j]

    for i in range(len(x_all)):
        afm = x_all[i][0]
        feature_diff_dic = {}
        for j in range(feature_num):
            feature_diff_dic[j] = feature_max_dic[j]-feature_min_dic[j]
            if feature_diff_dic[j] ==0:
                feature_diff_dic[j] = 1
            afm[:,j] = (afm[:,j] - feature_min_dic[j])/(feature_diff_dic[j])
        x_all[i][0] = afm
    return x_all

class MolDatum():
    """
        Class that represents a train/validation/test datum
        - self.label: 0 neg, 1 pos -1 missing for different target.
    """
    def __init__(self, x, label, target, smile, index):
        self.adj = x[1]
        self.afm = x[0]
        self.bft = x[2]
        self.orderAtt = x[3]
        self.aromAtt = x[4]
        self.conjAtt = x[5]
        self.ringAtt = x[6]
        self.subtype = x[7]
        self.label = label
        self.target = target
        self.index = x[8]
        self.smile = smile

def construct_dataset(x_all, y_all, target, smile_all):
    output = []
    for i in range(len(x_all)):
        output.append(MolDatum(x_all[i], y_all[i], target, smile_all[i], i))
    return(output)

class MolDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list):
        """
        @param data_list: list of MolDatum
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        adj, afm, TypeAtt, orderAtt, aromAtt, conjAtt, ringAtt = self.data_list[key].adj, self.data_list[key].afm, self.data_list[key].bft, \
                                                              self.data_list[key].orderAtt, self.data_list[key].aromAtt, \
                                                                      self.data_list[key].conjAtt, self.data_list[key].ringAtt
        subtype = self.data_list[key].subtype
        label = self.data_list[key].label
        smile = self.data_list[key].smile
        index = self.data_list[key].index
        return (adj, afm, TypeAtt, orderAtt, aromAtt, conjAtt, ringAtt, label, smile, subtype, index)

def mol_collate_func_reg(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    adj_list = []
    afm_list =[]
    label_list = []
    size_list = []
    typeAtt_list = []
    smile_list = []
    subtype_list = []
    index_list = []
    orderAtt_list, aromAtt_list, conjAtt_list, ringAtt_list = [], [], [], []
    for datum in batch:
        label_list.append(datum[7])
        size_list.append(datum[0].shape[0])
        smile_list.append(datum[8])
        index_list.append(datum[10])

    max_size = np.max(size_list) # max of batch. 55 for solu, 115 for lipo, 24 for freesolv
    #max_size = max_molsize #max_molsize 132
    btf_len = datum[2].shape[0]
    # padding
    for datum in batch:
        filled_adj = np.zeros((max_size, max_size), dtype=np.float32)
        filled_adj[0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[0]
        filled_afm = np.zeros((max_size, 24), dtype=np.float32)
        filled_afm[0:datum[0].shape[0], :] = datum[1]
        filled_TypeAtt = np.zeros((btf_len, max_size, max_size), dtype=np.float32)
        filled_TypeAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[2]

        filled_orderAtt = np.zeros((4, max_size, max_size), dtype=np.float32)
        filled_orderAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[3]

        filled_aromAtt = np.zeros((2, max_size, max_size), dtype=np.float32)
        filled_aromAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[4]

        filled_conjAtt = np.zeros((2, max_size, max_size), dtype=np.float32)
        filled_conjAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[5]

        filled_ringAtt = np.zeros((2, max_size, max_size), dtype=np.float32)
        filled_ringAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[6]

        filled_subtype = np.zeros((max_size, 1), dtype=np.float32)
        filled_subtype[0:datum[0].shape[0], :] = datum[9]

        adj_list.append(filled_adj)
        afm_list.append(filled_afm)
        typeAtt_list.append(filled_TypeAtt)
        orderAtt_list.append(filled_orderAtt)
        aromAtt_list.append(filled_aromAtt)
        conjAtt_list.append(filled_conjAtt)
        ringAtt_list.append(filled_ringAtt)
        subtype_list.append(filled_subtype)

    if use_cuda:
        return ([torch.from_numpy(np.array(adj_list)).cuda(), torch.from_numpy(np.array(afm_list)).cuda(),
                 torch.from_numpy(np.array(typeAtt_list)).cuda(), torch.from_numpy(np.array(orderAtt_list)).cuda(),
                 torch.from_numpy(np.array(aromAtt_list)).cuda(), torch.from_numpy(np.array(conjAtt_list)).cuda(),
                 torch.from_numpy(np.array(ringAtt_list)).cuda(),
                 torch.from_numpy(np.array(label_list)).cuda(), torch.from_numpy(np.array(subtype_list)).cuda(),
                 torch.from_numpy(np.array(size_list)).cuda(), torch.from_numpy(np.array(index_list)).cuda()])
    else:
        return ([torch.from_numpy(np.array(adj_list)), torch.from_numpy(np.array(afm_list)),
                 torch.from_numpy(np.array(typeAtt_list)), torch.from_numpy(np.array(orderAtt_list)),
                 torch.from_numpy(np.array(aromAtt_list)), torch.from_numpy(np.array(conjAtt_list)),
                 torch.from_numpy(np.array(ringAtt_list)),
                 torch.from_numpy(np.array(label_list)), torch.from_numpy(np.array(subtype_list)),
                 torch.from_numpy(np.array(size_list)), torch.from_numpy(np.array(index_list))])

def mol_collate_func_class(batch):

    adj_list = []
    afm_list =[]
    label_list = []
    size_list = []
    typeAtt_list = []
    orderAtt_list, aromAtt_list, conjAtt_list, ringAtt_list = [], [], [], []
    subtype_list = []
    index_list = []

    for datum in batch:
        label_list.append(datum[7])
        size_list.append(datum[0].shape[0])
        index_list.append(datum[10])
    max_size = np.max(size_list) # max of batch    222 for hiv, 132 for tox21,
    btf_len = datum[2].shape[0]
    #max_size = max_molsize #max_molsize 132
    # padding
    for datum in batch:
        filled_adj = np.zeros((max_size, max_size), dtype=np.float32)
        filled_adj[0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[0]
        filled_afm = np.zeros((max_size, 24), dtype=np.float32)
        filled_afm[0:datum[0].shape[0], :] = datum[1]

        filled_typeAtt = np.zeros((btf_len, max_size, max_size), dtype=np.float32)
        filled_typeAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[2]

        filled_orderAtt = np.zeros((4, max_size, max_size), dtype=np.float32)
        filled_orderAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[3]

        filled_aromAtt = np.zeros((2, max_size, max_size), dtype=np.float32)
        filled_aromAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[4]

        filled_conjAtt = np.zeros((2, max_size, max_size), dtype=np.float32)
        filled_conjAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[5]

        filled_ringAtt = np.zeros((2, max_size, max_size), dtype=np.float32)
        filled_ringAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[6]

        filled_subtype = np.zeros((max_size, 1), dtype=np.float32)
        filled_subtype[0:datum[0].shape[0], :] = datum[9]

        adj_list.append(filled_adj)
        afm_list.append(filled_afm)
        typeAtt_list.append(filled_typeAtt)
        orderAtt_list.append(filled_orderAtt)
        aromAtt_list.append(filled_aromAtt)
        conjAtt_list.append(filled_conjAtt)
        ringAtt_list.append(filled_ringAtt)
        subtype_list.append(filled_subtype)

    if use_cuda:
        return ([torch.from_numpy(np.array(adj_list)).cuda(), torch.from_numpy(np.array(afm_list)).cuda(),
                 torch.from_numpy(np.array(typeAtt_list)).cuda(), torch.from_numpy(np.array(orderAtt_list)).cuda(),
                 torch.from_numpy(np.array(aromAtt_list)).cuda(), torch.from_numpy(np.array(conjAtt_list)).cuda(),
                 torch.from_numpy(np.array(ringAtt_list)).cuda(),
                 FloatTensor(label_list), torch.from_numpy(np.array(subtype_list)).cuda(),
                 torch.from_numpy(np.array(size_list)).cuda(), torch.from_numpy(np.array(index_list)).cuda()])
    else:
        return ([torch.from_numpy(np.array(adj_list)), torch.from_numpy(np.array(afm_list)),
             torch.from_numpy(np.array(typeAtt_list)),torch.from_numpy(np.array(orderAtt_list)),
                 torch.from_numpy(np.array(aromAtt_list)), torch.from_numpy(np.array(conjAtt_list)),
                 torch.from_numpy(np.array(ringAtt_list)),
                 FloatTensor(label_list), torch.from_numpy(np.array(subtype_list)),
                 torch.from_numpy(np.array(size_list)), torch.from_numpy(np.array(index_list))])

def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))

def weight_tensor(weights, labels):
    # when labels is variable
    weight_tensor = []
    a = IntTensor([1])
    b = IntTensor([0])
    if use_cuda:
        nan_mark = torch.from_numpy(np.float32(['nan'])).cuda()
    else:
        nan_mark = torch.from_numpy(np.float32(['nan']))

    for i in range(0, labels.data.shape[0]):
        for j in range(0, labels.data.shape[1]):
            try:
                if torch.equal(IntTensor([int(labels.data[i][j])]), a):
                    weight_tensor.append(weights[j][0])
                elif torch.equal(IntTensor([int(labels.data[i][j])]), b):
                    weight_tensor.append(weights[j][1])
                else:
                    weight_tensor.append(0)
            except ValueError:
                weight_tensor.append(0)
            else:
                pass
    if use_cuda:
        return (torch.from_numpy(np.array(weight_tensor, dtype=np.float32)).cuda())
    else:
        return(torch.from_numpy(np.array(weight_tensor, dtype=np.float32)))

def set_weight(y_all):
    weight_dic = {}
    pos_dic ={}
    neg_dic = {}
    for i in range(len(y_all)):
        for j in range(len(y_all[0])):
            if y_all[i][j] == 1:
                if pos_dic.get(j) is None:
                    pos_dic[j] = 1
                else:
                    pos_dic[j] += 1
            elif y_all[i][j] == 0:
                if neg_dic.get(j) is None:
                    neg_dic[j] = 1
                else:
                    neg_dic[j] += 1

    for key in pos_dic.keys():
        weight_dic[key] = [5000/pos_dic[key], 5000/neg_dic[key]]
    return(weight_dic)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('GraphConv_base') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    """
    if classname.find('Conv2d') != -1:
        m.weight.data.fill_(1.0)
    """

def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2


def data_padding(x, max_size):
    btf_len = x[0][2].shape[0]
    x_padded = []
    for data in x:  # afm, adj, bft
        filled_adj = np.zeros((max_size, max_size), dtype=np.float32)
        filled_adj[0:data[1].shape[0], 0:data[1].shape[0]] = data[1]
        filled_afm = np.zeros((max_size, 25), dtype=np.float32)
        filled_afm[0:data[0].shape[0], :] = data[0]
        filled_bft = np.zeros((btf_len, max_size, max_size), dtype=np.float32)
        filled_bft[:, 0:data[0].shape[0], 0:data[0].shape[0]] = data[2]

        x_padded.append([filled_adj, filled_afm, filled_bft])
    return(x_padded)

def construct_loader(x, y, target, smile, batch_size, shuffle=True):
    data_set = construct_dataset(x, y, target, smile)    #变成类
    data_set = MolDataset(data_set)
    loader = torch.utils.data.DataLoader(dataset=data_set,
                                               batch_size=batch_size,
                                               collate_fn=mol_collate_func_class,
                                               shuffle=shuffle)
    return loader

def construct_loader_reg(x, y, target, smile, batch_size, shuffle=True):
    data_set = construct_dataset(x, y, target, smile)
    data_set = MolDataset(data_set)
    loader = torch.utils.data.DataLoader(dataset=data_set,
                                               batch_size=batch_size,
                                               collate_fn=mol_collate_func_reg,
                                               shuffle=shuffle)
    return loader

def earily_stop(val_acc_history, tasks, early_stop_step_single,
                early_stop_step_multi, required_progress):
    """
    Stop the training if there is no non-trivial progress in k steps
    @param val_acc_history: a list contains all the historical validation acc
    @param required_progress: the next acc should be higher than the previous by
        at least required_progress amount to be non-trivial
    @param t: number of training steps
    @return: a boolean indicates if the model should earily stop
    """
    # TODO: add your code here
    if len(tasks) == 1:
        t = early_stop_step_single
    else:
        t = early_stop_step_multi

    if len(val_acc_history)>t:
        if val_acc_history[-1] - val_acc_history[-1-t] < required_progress:
            return True
    return False

from matplotlib import pyplot as plt
def tsne_plot(all, dataset, epoch, number = 10,  n_components=2, random_state=2, early_exaggeration=30.0, perplexity=30, n_iter=500):
    X = all[0]
    num = []
    num.append(X.shape[0])
    y = torch.zeros(num[0])
    for i in range(len(all)):
        if i != 0:
            ele = all[i].data.cpu()
            X = torch.cat((X, ele))
            num.append(ele.shape[0])
            y_i = torch.ones(ele.shape[0]) * i
            y = torch.cat((y, y_i))
    #X = torch.cat((X_1, X_2))
    #y_1 = torch.zeros([num_car])
    #y_2 = torch.ones([num_oxy])
    #y = torch.cat((y_1,y_2))
    tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity,
                early_exaggeration = early_exaggeration, n_iter=n_iter)
    X_2d = tsne.fit_transform(X)
    plt.figure(figsize=(6, 5))
    #colors = np.random.rand(len(num))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', "lightpink",  "cyan", "gold", (0.1, 0.2, 0.3),  (0.2, 0.3, 0.4),  (0.3, 0.4, 0.5), (0.1, 0.2, 0.8)]
    target_ids = range(2)
    #for i, c, label in zip(target_ids, colors, [0, 1]):
    pre_num = 0
    print("we have numbrer of atoms: ")
    print(*num)

    plot_list = range(10)
    if number == 10:
        plot_list = range(10)
    elif number == 9:
        plot_list = range(1, 10)
    elif number == 8:
        plot_list = range(1, 9)
    elif number == 7:
        plot_list = [1,2,3,4,6,7,8]
    elif number == 6:
        plot_list = [1,2,3,4,6,7]
    elif number == 5:
        plot_list = [1,2,3,4,7]
    elif number == 4:
        plot_list = [1,2,3,7]
    elif number == 3:
        plot_list = [1,2,7]
    elif number == 2:
        plot_list = [1,2]

    labels = ['B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']
    for i in range(len(num)):
        if i not in plot_list:
            continue
        current_num = pre_num + num[i]
        plt.scatter(X_2d[pre_num:current_num, 0], X_2d[pre_num:current_num, 1], alpha = 0.2, c=colors[i], label=labels[i])
        pre_num = current_num
    #plt.scatter(X_2d[:num_car, 0], X_2d[:num_car, 1], c=colors[i], label=0)
    #plt.scatter(X_2d[num_car:, 0], X_2d[num_car:, 1], c='b', label=1)

    plt.legend()
    #plt.show()
    plt.savefig('./tsne_plots/tsne_plot_{}_{}_{}.png'.format(dataset, epoch, number))
    plt.close()
    # 1. dump the representation to lacal disk. (both train and test)
    # 2. try different tsne parameter or other dimension reduction methods.
    # 3. show 10, 9, 8 , ... atoms plot
    # 4. label subtype of C, dont change the model, just show in tsne plot.

