from __future__ import division
from __future__ import print_function
from models import *
from torch.utils.data import Dataset
from sklearn import metrics
from sklearn.utils import shuffle, resample
from sklearn.model_selection import train_test_split, KFold
import os
import matplotlib.pyplot as plt
from time import gmtime, strftime
import argparse
from grap import *
from scipy import interp

exp_data = kong()
rs = 0
batch_size = 32
n_sgc1_1, n_sgc1_2, n_sgc1_3, n_sgc1_4, n_sgc1_5 = 30, 10, 10, 10, 10  # 30, 10, 10, 10, 10
n_sgc2_1, n_sgc2_2, n_sgc2_3, n_sgc2_4, n_sgc2_5 = 60, 20, 20, 20, 20  # 60, 20, 20, 20, 20
n_den1, n_den2 = 64, 32
dr = 0.3
arch = 'Weighted_sum'
molfp = 'sum'
model = EAGCN(n_bfeat=exp_data.n_bfeat, n_afeat=exp_data.n_afeat,
              n_sgc1_1=n_sgc1_1, n_sgc1_2=n_sgc1_2, n_sgc1_3=n_sgc1_3, n_sgc1_4=n_sgc1_4, n_sgc1_5=n_sgc1_5,
              n_sgc2_1=n_sgc2_1, n_sgc2_2=n_sgc2_2, n_sgc2_3=n_sgc2_3, n_sgc2_4=n_sgc2_4, n_sgc2_5=n_sgc2_5,
              n_den1=n_den1, n_den2=n_den2, nclass=1,
              dropout=dr, structure=arch, molfp_mode=molfp)
def test_model(loader, model):
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
    train_loader , test_loader = exp_data.get_train_val_test_loader(rs, batch_size,flodi)
    fpr , tpr = test_model(test_loader,model)
    auc = metrics.auc(fpr, tpr)
    return fpr,tpr,auc
def pl_test_final(modelname,molnum,datasetd):
    checkpoint = torch.load(modelname+'/'+str(molnum)+'modelpara.pth')
    model.load_state_dict(checkpoint['net'])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    train_loader = exp_data.get_train_val_test_loader_final( batch_size,datasetd)
    fpr , tpr = test_model(train_loader,model)
    auc = metrics.auc(fpr, tpr)
    return fpr,tpr,auc
fpr,tpr,roc_auc = pl_test_final('Lnc2cancerfinalmodel',3,'lncRNAdisease')
plt.plot(fpr,tpr,color='b',label=r'lncRNAdisease ROC (AUC=%0.4f)'%roc_auc,lw=2,alpha=.8)
fpr,tpr,roc_auc = pl_test_final('lncRNAdiseasefinalmodel',2,'Lnc2cancer')
plt.plot(fpr,tpr,color='orange',label=r'Lnc2cancer ROC (AUC=%0.4f)'%roc_auc,lw=2,alpha=.8)
# 计算auc
# roc_auc = metrics.auc(fpr, tpr)

# 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来

# fprs = []
# tprs = []
# aucs = []
# mean_fpr=np.linspace(0,1,100)
#
# for i in range(0,10):
#     fpr,tpr,roc_auc = pl_test('finalmodel',i,5)
#     tprs.append(interp(mean_fpr, fpr, tpr))
#     tprs[-1][0] = 0.0
#     # 计算auc
#     roc_auc = metrics.auc(fpr, tpr)
#     aucs.append(roc_auc)
#     # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来
#     plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d(area=%0.4f)' % (i, roc_auc))
#     i += 1
# mean_tpr=np.mean(tprs,axis=0)
# mean_tpr[-1]=1.0
# mean_auc=metrics.auc(mean_fpr,mean_tpr)#计算平均AUC值
# std_auc=np.std(tprs,axis=0)
# w= open('mymodel_fpr.txt','w')
# w.write(str(mean_fpr))
# w.close()
# w= open('mymodel_tpr.txt','w')
# w.write(str(mean_tpr))
# w.close()
# plt.plot(mean_fpr,mean_tpr,color='b',label=r'Mean ROC (area=%0.4f)'%mean_auc,lw=2,alpha=.8)
# std_tpr=np.std(tprs,axis=0)

plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc='lower right')
plt.show()







# plt.plot(fpr[0].extend(fpr[1]), tpr[0].extend(tpr[1]), color='Red', label='lncRNAdisese\'s dataset AUC:(area = %0.4f)' % auc[1])
# plt.plot(fpr[2], tpr[2], color='RoyalBlue', label='Lnc2cancer\'s dataset AUC:(area = %0.4f)' % auc[2])
# plt.plot(fpr[3], tpr[3], color='coral', label='TowAdd\'s dataset AUC:(area = %0.4f)' % auc[3])
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False  Positive  Rate')
# plt.ylabel('True  Positive  Rate')
# plt.title('ROC')
# plt.legend(loc="lower right")
# plt.show()