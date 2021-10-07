from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import math
import torch
# import torch.nn as F
import torch.optim as optim
from torch.autograd import Variable
from utils import *
from models import *
from torch.utils.data import Dataset
from sklearn import metrics
from sklearn.utils import shuffle, resample
from sklearn.model_selection import train_test_split, KFold
import os
import matplotlib.pyplot as plt
from time import gmtime, strftime
import argparse
# from grap import *
from grap_radomfuyangben import *
# CUDA_VISIBLE_DEVICES=1

batch_size = 32
rs = 0
molfp = 'sum'
arch = 'Weighted_sum'
dr = 0.3
type = 'class'
print_freq = 50
dataset = 'kong'
eval_train_loader = False
write_file = True

n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5 = 30, 10, 10, 10, 10  # 30, 10, 10, 10, 10
n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5 = 60, 20, 20, 20, 20  # 60, 20, 20, 20, 20
weight_decay = 0.0001  # L-2 Norm
num_epochs = 6
learning_rate = 0.01
exp_data = kong()
n_den1, n_den2 = 64, 32



experiment_date = strftime("%y_%b_%d_%H:%M", gmtime()) +'New'
print(experiment_date)
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor

def test_model(loader, model, tasks):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    true_value = []
    all_out = []
    model.eval()
    out_value_dic = {}
    true_value_dic = {}
    for adj, afm, btf, orderAtt, aromAtt, conjAtt, ringAtt, labels,  size, index in loader:
        adj_batch, afm_batch, btf_batch, label_batch = Variable(adj), Variable(afm), Variable(btf), Variable(labels)
        orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch = Variable(orderAtt), Variable(aromAtt), Variable(
            conjAtt), Variable(ringAtt)
        size_batch = Variable(size)
        outputs, _, _ = model(adj_batch, afm_batch, btf_batch, orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch, size_batch)
        probs = torch.sigmoid(outputs)
        if use_cuda:
            out_list = probs.cpu().data.view(-1).numpy().tolist()
            all_out.extend(out_list)
            label_list = labels.cpu().numpy().tolist()
            true_value.extend([item for sublist in label_list for item in sublist])
            out_sep_list = probs.cpu().data.view(-1, len(tasks)).numpy().tolist()
        else:
            out_list = probs.data.view(-1).numpy().tolist()
            all_out.extend(out_list)
            label_list = labels.numpy().tolist()
            true_value.extend([item for item in label_list ])
            out_sep_list = probs.data.view(-1, len(tasks)).numpy().tolist()
    model.train()

    fpr, tpr, threshold = metrics.roc_curve(true_value, all_out, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return(auc)
def train(tasks, n_den1, n_den2,flodi,add2):

    train_loader,  test_loader = exp_data.get_train_val_test_loader(rs,batch_size,flodi,dataku,add1)
    model = EAGCN(n_bfeat=exp_data.n_bfeat, n_afeat=exp_data.n_afeat,
                  n_sgc1_1=n_sgc1_1, n_sgc1_2=n_sgc1_2, n_sgc1_3=n_sgc1_3, n_sgc1_4=n_sgc1_4, n_sgc1_5=n_sgc1_5,
                  n_sgc2_1=n_sgc2_1, n_sgc2_2=n_sgc2_2, n_sgc2_3=n_sgc2_3, n_sgc2_4=n_sgc2_4, n_sgc2_5=n_sgc2_5,
                  n_den1=n_den1, n_den2=n_den2, nclass=len(tasks),
                  dropout=dr, structure=arch, molfp_mode = molfp)
    if use_cuda:
        model.cuda()
    model.apply(weights_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    max_i = 0
    for epoch in range(num_epochs):
        for i, (adj, afm, btf, orderAtt, aromAtt, conjAtt, ringAtt, labels,size, index) in enumerate(train_loader):
            adj_batch, afm_batch, btf_batch, label_batch = Variable(adj), Variable(afm), Variable(btf), Variable(labels)
            orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch = Variable(orderAtt), Variable(
                aromAtt), Variable(
                conjAtt), Variable(ringAtt)
            size_batch = Variable(size)
            optimizer.zero_grad()
            outputs, _, _ = model(adj_batch, afm_batch, btf_batch, orderAtt_batch, aromAtt_batch, conjAtt_batch,
                            ringAtt_batch, size_batch)
            if exp_data.type == 'reg':
                criterion = torch.nn.MSELoss()
                if use_cuda:
                    criterion.cuda()
                loss = criterion(outputs.view(-1), label_batch.float().view(-1))
            else:
                non_nan_num = Variable(FloatTensor([(labels == 1).sum() + (labels == 0).sum()]))
                loss = F.binary_cross_entropy_with_logits(outputs.view(-1), \
                                                          label_batch.float().view(-1), \
                                                           size_average=False) / non_nan_num
            loss.backward()
            optimizer.step()
            if i > max_i:
                max_i = i
            # report performance
            if exp_data.type == 'class' and i%print_freq == 0:
                train_acc_tot = test_model(train_loader, model, tasks)
                test_acc_tot = test_model(test_loader, model, tasks)
                print(
                    'Epoch: [{}/{}], '
                    'Loss: {}, \n'
                    'Train AUC total: {}, \n'
                    'test AUC total: {}, \n'.format(
                        epoch + 1, num_epochs,  loss.data[0],train_acc_tot,
                         test_acc_tot))
                if write_file:
                    with open(dataku+add1+str(flodi)+'/'+add2+str(exp_data.sn)+'.txt', 'a') as fp:
                        fp.write(
                            '[{}/{}];{};{};{}\n'
                                .format(
                                epoch + 1, num_epochs,
                                loss.data[0], \
                                train_acc_tot,
                                  test_acc_tot))
            state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            torch.save(state, u'C:/备份/参考文献/lcn-疾病/EAGCN-master/eagcn_pytorch/'+dataku+add1+str(flodi)+'/'+str(epoch)+'modelpara.pth')
    return

tasks = exp_data.all_tasks    # or [task] if want to focus on one single task.

if use_cuda:
    position = 'server'
else:
    position = 'local'
dataku = 'TowAdd'
NFLOD= 10
# predata = prekong(dataku)
# predata.flod(NFLOD)
add1 = str(NFLOD)+'flodneg'
# for i in range(0,NFLOD):
#     train(tasks, n_den1, n_den2,i,'kongmengfan'+dataku)







from scipy import interp
model = EAGCN(n_bfeat=exp_data.n_bfeat, n_afeat=exp_data.n_afeat,
              n_sgc1_1=n_sgc1_1, n_sgc1_2=n_sgc1_2, n_sgc1_3=n_sgc1_3, n_sgc1_4=n_sgc1_4, n_sgc1_5=n_sgc1_5,
              n_sgc2_1=n_sgc2_1, n_sgc2_2=n_sgc2_2, n_sgc2_3=n_sgc2_3, n_sgc2_4=n_sgc2_4, n_sgc2_5=n_sgc2_5,
              n_den1=n_den1, n_den2=n_den2, nclass=1,
              dropout=dr, structure=arch, molfp_mode=molfp)


def test_model2(loader, model):
    true_value = []
    all_out = []
    model.eval()
    for adj, afm, btf, orderAtt, aromAtt, conjAtt, ringAtt, labels,  size, index in loader:
        adj_batch, afm_batch, btf_batch, label_batch = Variable(adj), Variable(afm), Variable(btf), Variable(labels)
        orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch = Variable(orderAtt), Variable(aromAtt), Variable(
            conjAtt), Variable(ringAtt)
        size_batch = Variable(size)
        outputs, _, _ = model(adj_batch, afm_batch, btf_batch, orderAtt_batch, aromAtt_batch, conjAtt_batch, ringAtt_batch, size_batch)
        probs = torch.sigmoid(outputs)
        if use_cuda:
            out_list = probs.cpu().data.view(-1).numpy().tolist()
            all_out.extend(out_list)
            label_list = labels.cpu().numpy().tolist()
            true_value.extend([item for sublist in label_list for item in sublist])
        else:
            out_list = probs.data.view(-1).numpy().tolist()
            all_out.extend(out_list)
            label_list = labels.numpy().tolist()
            true_value.extend([item for item in label_list ])
    fpr, tpr, threshold = metrics.roc_curve(true_value, all_out, pos_label=1)
    return fpr,tpr
def pl_test(modelname,flodi,molnum):
    checkpoint = torch.load(modelname+'/'+str(molnum)+'modelpara.pth')
    model.load_state_dict(checkpoint['net'])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])

    train_loader , test_loader = exp_data.get_train_val_test_loader(rs, batch_size, flodi, dataku, add1)
    fpr , tpr = test_model2(test_loader,model)
    auc = metrics.auc(fpr, tpr)
    return fpr,tpr,auc
fprs = []
tprs = []
aucs = []
mean_fpr=np.linspace(0,1,100)

for i in range(NFLOD):
    fpr,tpr,roc_auc = pl_test(dataku+add1+str(i),i,5)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    # 计算auc
    roc_auc = metrics.auc(fpr, tpr)
    aucs.append(roc_auc)
    # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d(area=%0.4f)' % (i, roc_auc))
    i += 1
mean_tpr=np.mean(tprs,axis=0)
mean_tpr[-1]=1.0
mean_auc=metrics.auc(mean_fpr,mean_tpr)#计算平均AUC值
std_auc=np.std(tprs,axis=0)
w= open(dataku+str(NFLOD)+'flodneg'+'mymodel_fpr.txt','w')
w.write(str(mean_fpr))
w.close()
w= open(dataku+str(NFLOD)+'flodneg'+'mymodel_tpr.txt','w')
w.write(str(mean_tpr))
w.close()
plt.plot(mean_fpr,mean_tpr,color='b',label=r'Mean ROC (area=%0.4f)'%mean_auc,lw=2,alpha=.8)
std_tpr=np.std(tprs,axis=0)
# tprs_upper=np.minimum(mean_tpr+std_tpr,1)
# tprs_lower=np.maximum(mean_tpr-std_tpr,0)
# plt.fill_between(mean_tpr,tprs_lower,tprs_upper,color='gray',alpha=.2)
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title( 'ROC')
plt.legend(loc='lower right')
plt.show()
