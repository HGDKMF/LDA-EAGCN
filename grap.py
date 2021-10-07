# coding:utf-8
import pandas as pd
import csv
import ast
import codecs
import json
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os
import torch
import sklearn
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
class prekong():
    def __init__(self,dataset):
        self.sn = 7
        self.dataset = dataset
        self.n_bfeat = 14  # default= 14
        self.n_afeat = 20  # default= 20
    def sort_lcn(self,automname,sn,filename):
        pd1data = pd.read_csv('data/lcnlcn'+filename+'.txt', sep='\t', index_col=0)
        for i in pd1data.columns:
            if pd1data[i].count() == 0:
                pd1data.drop(labels=i, axis=1, inplace=True)
        data0 = pd1data.loc[automname]
        data0 = data0.sort_values(ascending=False)
        dlist =  data0[0:sn]
        return dlist
    def sort_dis(self,automname,sn,filename):
        pd1data = pd.read_pickle('data/disdis'+filename+'.pkl')

        data0 = pd1data.loc[automname]
        data0 = data0.sort_values(ascending=False)
        dlist =  data0[0:sn]
        return dlist
    def grapf_att(self):
        self.fn = 20
        guanxi = pd.read_excel(u'data/LcnName'+self.dataset+'.xlsx',sheet_name='Sheet1',index_col=None)
        x_1 = []
        for gx in guanxi.values:#一个图
            gx = list(gx)
            print(gx)
            afm =np.ones([ 2*self.sn, self.fn], dtype = np.float32)
            core_att = np.zeros([2, 2*self.sn, 2*self.sn], dtype = np.float32)
            nomal_att = np.zeros([2, 2*self.sn, 2*self.sn], dtype = np.float32)
            dd_att = np.zeros([2, 2*self.sn, 2*self.sn], dtype = np.float32)
            ll_att = np.zeros([2, 2*self.sn, 2*self.sn], dtype = np.float32)
            node_att = np.zeros([2, 2*self.sn, 2*self.sn], dtype = np.float32)
            lcn = gx[0]
            dis = gx[1]
            lcninfo =  self.sort_lcn(lcn,self.sn,self.dataset)
            disinfo = self.sort_dis(dis,self.sn,self.dataset)
            lcnsmilar = lcninfo.index.values.tolist()
            dissimlar = disinfo.index.values.tolist()
            adj_index = lcnsmilar+dissimlar
            graph_adj = pd.DataFrame(index=adj_index,columns=adj_index,data=0)
            for l in lcnsmilar:
                l_dis_dataframe =guanxi.loc[guanxi['lcn']==l]
                for d in dissimlar:
                    ld_d = l_dis_dataframe.loc[l_dis_dataframe['disease']==d]
                    if len(ld_d):
                        i, j = adj_index.index(l), adj_index.index(d)
                        if((str(l)==str(lcn))&(str(d)==str(dis))):
                            core_att[0:, i, j] =[1,1]
                        else:
                            nomal_att[0:, i, j] = [1,1]
                        graph_adj.loc[l,d] = 1
                        graph_adj.loc[d, l] = 1
            adj = graph_adj.values
            for i,lcnm in enumerate(lcninfo):
                if i !=0:
                    graph_adj.iloc[i, 0] = lcnm
                    graph_adj.iloc[0, i] = lcnm
                    ll_att[0:, i, 0] = [1, lcnm]
                    ll_att[0:, 0, i] = [1, lcnm]
            for i,dism in enumerate(disinfo):
                if i !=0:
                    graph_adj.iloc[i, 0] = dism
                    graph_adj.iloc[0, i] = dism
                    dd_att[0:, i, 0] = [1, dism]
                    dd_att[0:, 0, i] = [1, dism]
            node_co_n = []
            for i,dex in enumerate(adj_index):
                num = 0
                if(i<7):
                    num =len( guanxi.loc[guanxi['lcn'] == dex])
                else:
                    num = len(guanxi.loc[guanxi['disease'] == dex])
                node_co_n.append(num)
            for i,idex in enumerate(adj_index):
                for j, jdex in enumerate(adj_index):
                    if graph_adj.loc[idex,jdex]:
                        node_att[0:, i, j] = [1,node_co_n[i]+node_co_n[j]]


            x_1.append([afm.tolist(),adj.tolist(),core_att.tolist(),nomal_att.tolist(),node_att.tolist(),ll_att.tolist(),dd_att.tolist()])
        return x_1,len(x_1)
    def Negative_co(self,negn):
        x_2 = []
        file_all = pd.read_csv('data/result.txt',sep=';')
        for i in file_all.columns:
            if file_all[i].count() == 0:
                file_all.drop(labels=i, axis=1, inplace=True)
        file_all.sort_values(axis=0,ascending=False,by=['score'],inplace=True)
        dislist = file_all.reset_index(drop=True).iloc[0:int(negn/100),0].values
        pd1data = pd.read_csv('data/finalscore.txt',sep='\t',index_col=0)
        for i in pd1data.columns:
            if pd1data[i].count() == 0:
                pd1data.drop(labels=i, axis=1, inplace=True)
        disscore = pd1data.loc[dislist].iloc[:,292:]
        fn = 20
        guanxi = pd.read_excel(u'data/lcnNameTowAdd.xlsx', sheet_name='Sheet1', index_col=None)

        for disi in dislist:
            for disj in  disscore.loc[disi].sort_values(ascending=True).index[0:100]:
                afm = np.ones([2 * self.sn, fn], dtype=np.float32)
                core_att = np.zeros([2, 2 * self.sn, 2 * self.sn], dtype=np.float32)
                nomal_att = np.zeros([2, 2 * self.sn, 2 * self.sn], dtype=np.float32)
                dd_att = np.zeros([2, 2 *self.sn, 2 *self.sn], dtype=np.float32)
                ll_att = np.zeros([2, 2 *self.sn, 2 * self.sn], dtype=np.float32)
                node_att = np.zeros([2, 2 *self.sn, 2 *self.sn], dtype=np.float32)
                lcn = disj
                dis = disi
                lcninfo = self.sort_lcn(lcn, self.sn,'Towadd')
                disinfo = self.sort_dis(dis,self.sn,'Towadd')
                lcnsmilar = lcninfo.index.values.tolist()
                dissimlar = disinfo.index.values.tolist()
                adj_index = lcnsmilar + dissimlar
                graph_adj = pd.DataFrame(index=adj_index, columns=adj_index, data=0)
                for l in lcnsmilar:
                    l_dis_dataframe = guanxi.loc[guanxi['lcn'] == l]
                    for d in dissimlar:
                        ld_d = l_dis_dataframe.loc[l_dis_dataframe['disease'] == d]
                        if len(ld_d):
                            i, j = adj_index.index(l), adj_index.index(d)
                            if ((str(l) == str(lcn)) & (str(d) == str(dis))):
                                core_att[0:, i, j] = [1, 1]
                            else:
                                nomal_att[0:, i, j] = [1, 1]
                            graph_adj.loc[l, d] = 1
                            graph_adj.loc[d, l] = 1
                adj = graph_adj.values
                for i, lcnm in enumerate(lcninfo):
                    if i != 0:
                        graph_adj.iloc[i, 0] = lcnm
                        graph_adj.iloc[0, i] = lcnm
                        ll_att[0:, i, 0] = [1, lcnm]
                        ll_att[0:, 0, i] = [1, lcnm]
                for i, dism in enumerate(disinfo):
                    if i != 0:
                        graph_adj.iloc[i, 0] = dism
                        graph_adj.iloc[0, i] = dism
                        dd_att[0:, i, 0] = [1, dism]
                        dd_att[0:, 0, i] = [1, dism]
                node_co_n = []
                for i, dex in enumerate(adj_index):
                    num = 0
                    if (i < 7):
                        num = len(guanxi.loc[guanxi['lcn'] == dex])
                    else:
                        num = len(guanxi.loc[guanxi['disease'] == dex])
                    node_co_n.append(num)
                for i, idex in enumerate(adj_index):
                    for j, jdex in enumerate(adj_index):
                        if graph_adj.loc[idex, jdex]:
                            node_att[0:, i, j] = [1, node_co_n[i] + node_co_n[j]]
                x_2.append([afm.tolist(), adj.tolist(), core_att.tolist(), nomal_att.tolist(), node_att.tolist(), ll_att.tolist(), dd_att.tolist()])
        print(len(x_2))
        return x_2,len(x_2)
    def finaldata(self):
        x_select1, x1_n = self.grapf_att()
        x_select2, x2_n = self.Negative_co(x1_n)
        x_select = x_select1 + x_select2
        y_task = [[1, 0]] * len(x_select1) + [[0, 1]] * len(x_select2)
        x_select, y_task = sklearn.utils.shuffle(x_select, y_task)
        dataframe = pd.DataFrame({'x_select': x_select, 'y_task': y_task})
        dataframe.to_csv(self.dataset+"alldata.csv", sep=',')

    def disgraph_att(self,dis):
        self.fn = 20
        guanxi = pd.read_excel(u'data/LcnName' + self.dataset + '.xlsx', sheet_name='Sheet1', index_col=None)
        lncfile = pd.read_excel(u'data/LcnName' + self.dataset + '.xlsx', sheet_name='Sheet2', index_col=None)
        x_1 = []
        disinfo = self.sort_dis(dis, self.sn, self.dataset)
        dissimlar = disinfo.index.values.tolist()
        for lcn in lncfile.values:  # 一个图
            lcn = lcn[0]
            

            afm = np.ones([2 * self.sn, self.fn], dtype=np.float32)
            core_att = np.zeros([2, 2 * self.sn, 2 * self.sn], dtype=np.float32)
            nomal_att = np.zeros([2, 2 * self.sn, 2 * self.sn], dtype=np.float32)
            dd_att = np.zeros([2, 2 * self.sn, 2 * self.sn], dtype=np.float32)
            ll_att = np.zeros([2, 2 * self.sn, 2 * self.sn], dtype=np.float32)
            node_att = np.zeros([2, 2 * self.sn, 2 * self.sn], dtype=np.float32)


            lcninfo = self.sort_lcn(lcn, self.sn, self.dataset)

            lcnsmilar = lcninfo.index.values.tolist()

            adj_index = lcnsmilar + dissimlar
            graph_adj = pd.DataFrame(index=adj_index, columns=adj_index, data=0)
            for l in lcnsmilar:
                l_dis_dataframe = guanxi.loc[guanxi['lcn'] == l]
                for d in dissimlar:
                    ld_d = l_dis_dataframe.loc[l_dis_dataframe['disease'] == d]
                    if len(ld_d):
                        i, j = adj_index.index(l), adj_index.index(d)
                        if ((str(l) == str(lcn)) & (str(d) == str(dis))):
                            core_att[0:, i, j] = [1, 1]
                        else:
                            nomal_att[0:, i, j] = [1, 1]
                        graph_adj.loc[l, d] = 1
                        graph_adj.loc[d, l] = 1
            adj = graph_adj.values
            for i, lcnm in enumerate(lcninfo):
                if i != 0:
                    graph_adj.iloc[i, 0] = lcnm
                    graph_adj.iloc[0, i] = lcnm
                    ll_att[0:, i, 0] = [1, lcnm]
                    ll_att[0:, 0, i] = [1, lcnm]
            for i, dism in enumerate(disinfo):
                if i != 0:
                    graph_adj.iloc[i, 0] = dism
                    graph_adj.iloc[0, i] = dism
                    dd_att[0:, i, 0] = [1, dism]
                    dd_att[0:, 0, i] = [1, dism]
            node_co_n = []
            for i, dex in enumerate(adj_index):
                num = 0
                if (i < 7):
                    num = len(guanxi.loc[guanxi['lcn'] == dex])
                else:
                    num = len(guanxi.loc[guanxi['disease'] == dex])
                node_co_n.append(num)
            for i, idex in enumerate(adj_index):
                for j, jdex in enumerate(adj_index):
                    if graph_adj.loc[idex, jdex]:
                        node_att[0:, i, j] = [1, node_co_n[i] + node_co_n[j]]

      
            x_1.append(
                [afm.tolist(), adj.tolist(), core_att.tolist(), nomal_att.tolist(), node_att.tolist(), ll_att.tolist(),
                 dd_att.tolist()])
        return x_1, len(x_1)
    def disgraphdata(self,dis):
        x_select1, x1_n = self.disgraph_att(dis)
        y_task = [[1, 0]] * len(x_select1)
        dataframe = pd.DataFrame({'x_select': x_select1, 'y_task': y_task})
        dataframe.to_csv(r"data"+dis+".csv", sep=',')
    def flod(self,nflod):
        x_select1,x1_n = self.grapf_att()
        x_select2, x2_n = self.Negative_co(x1_n)
        x_select = x_select1 + x_select2
        print("x_select1、x_select2：",x_select1,x_select2)
        y_task = [[1,0]] * len(x_select1) + [[0,1]] *len( x_select2)
        x_select, y_task = sklearn.utils.shuffle(x_select, y_task)
        dataframe = pd.DataFrame({'x_select': x_select, 'y_task': y_task})
        dataframe.to_csv(r"alldata"+self.dataset, sep=',')
        f = open('alldata'+self.dataset, encoding='UTF-8')
        data = pd.read_csv(f)
        x_select = data['x_select'].values
        y_task = data['y_task'].values
        onenum = int(len(x_select)/10)
        for i in range(nflod):
            os.mkdir(self.dataset+str(nflod)+'flod'+str(i))
            x_selecttest = x_select[i*onenum:(i+1)*onenum]
            y_tasktest   = y_task[i*onenum:(i+1)*onenum]
            if i==0:
                y_tasktrain = y_task[(i + 1) * onenum:]
                x_selecttrain = x_select[(i+1)*onenum:]
                x_selectx = pd.DataFrame([str (x_selecttrain[i]) for i in range(len(x_selecttrain)) ], columns=['X_train'])
                y_taskx = pd.DataFrame([str (y_tasktrain[i]) for i in range(len(y_tasktrain)) ], columns=['y_train'])
                pd.concat([x_selectx, y_taskx], axis=1).to_csv(self.dataset+str(nflod)+'flod'+str(i)+'/train.csv', index=False, encoding='utf-8')
            elif i==nflod-1:
                y_tasktrain = y_task[0:i * onenum]
                x_selecttrain = x_select[0:i * onenum]
                x_selectx = pd.DataFrame([str (x_selecttrain[i]) for i in range(len(x_selecttrain)) ], columns=['X_train'])
                y_taskx = pd.DataFrame([str (y_tasktrain[i]) for i in range(len(y_tasktrain)) ], columns=['y_train'])
                pd.concat([x_selectx, y_taskx], axis=1).to_csv(self.dataset+str(nflod)+'flod'+str(i)+'/train.csv', index=False, encoding='utf-8')
            else:
                y_tasktrain1 = y_task[0:i * onenum]
                y_tasktrain2 = y_task[(i + 1) * onenum:]

                y_tasktrain = np.hstack((y_tasktrain1,y_tasktrain2))
                x_selecttrain1 = x_select[0:i*onenum]
                x_selecttrain2 = x_select[(i+1)*onenum:]
                x_selecttrain = np.hstack((x_selecttrain1,x_selecttrain2))
                x_selectx = pd.DataFrame([str (x_selecttrain[i]) for i in range(len(x_selecttrain)) ], columns=['X_train'])
                y_taskx = pd.DataFrame([str (y_tasktrain[i]) for i in range(len(y_tasktrain)) ], columns=['y_train'])
                pd.concat([x_selectx, y_taskx], axis=1).to_csv(self.dataset+str(nflod)+'flod'+str(i)+'/train.csv', index=False, encoding='utf-8')
            x_selectj = pd.DataFrame([str (x_selecttest[i]) for i in range(len(x_selecttest)) ], columns=['X_test'])
            y_taskj = pd.DataFrame([str (y_tasktest[i]) for i in range(len(y_tasktest)) ], columns=['y_test'])
            pd.concat([x_selectj, y_taskj], axis=1).to_csv(self.dataset+str(nflod)+'flod'+str(i)+'/test.csv', index=False, encoding='utf-8')
class kong():
    def __init__(self):
        self.all_tasks = ['lcn-dis']
        self.label_col_num = [1]
        self.sn = 7
        self.n_bfeat = 14  # default= 14
        self.n_afeat = 20  # default= 20
        self.type = 'class'

    def get_train_val_test_loader_dis(self, random_state, batch_size,dis):
        df_train = pd.read_csv(r"data"+dis+".csv", sep=',', encoding='utf-8')
        train_loader = construct_loader(df_train['X_train'].values.tolist(), df_train['y_train'].values.tolist(),
                                        batch_size=batch_size)
        return train_loader
    def get_train_val_test_loader_final(self, batch_size,dataset):
        df_train = pd.read_csv(dataset+"alldata.csv", sep=',', encoding='utf-8')
        train_loader = construct_loader(df_train['x_select'].values.tolist(), df_train['y_task'].values.tolist(),
                                        batch_size=batch_size)
        return train_loader
    def get_train_val_test_loader(self, random_state, batch_size,flodi,nflod,dataset):
        df_train = pd.read_csv(dataset+str(nflod)+'flod'+str(flodi)+'/train.csv', sep=',', encoding='utf-8')
        train_loader = construct_loader(df_train['X_train'].values.tolist(), df_train['y_train'].values.tolist(), batch_size=batch_size)
        df_test = pd.read_csv(dataset+str(nflod)+'flod'+str(flodi)+'/test.csv', sep=',', encoding='utf-8')
        test_loader = construct_loader(df_test['X_test'].values.tolist(), df_test['y_test'].values.tolist(), batch_size=batch_size)
        return(train_loader, test_loader)
def construct_loader(x, y, batch_size, shuffle=True):
    data_set = construct_dataset(x, y)    #变成类
    data_set = MolDataset(data_set)
    loader = torch.utils.data.DataLoader(dataset=data_set,
                                               batch_size=batch_size,
                                               collate_fn=mol_collate_func_class,
                                               shuffle=shuffle)
    return loader
def mol_collate_func_class(batch):

    adj_list = []
    afm_list =[]
    label_list = []
    size_list = []
    typeAtt_list = []
    orderAtt_list, aromAtt_list, conjAtt_list, ringAtt_list = [], [], [], []

    index_list = []

    for datum in batch:
        label_list.append(datum[7][0])
        size_list.append(datum[0].shape[0])
        index_list.append(datum[8])
    max_size = np.max(size_list) # max of batch    222 for hiv, 132 for tox21,
    btf_len = datum[2].shape[0]
    #max_size = max_molsize #max_molsize 132
    # padding
    for datum in batch:
        filled_adj = np.zeros((max_size, max_size), dtype=np.float32)
        filled_adj[0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[0]
        filled_afm = np.zeros((max_size, 20), dtype=np.float32)
        filled_afm[0:datum[0].shape[0], :] = datum[1]

        filled_typeAtt = np.zeros((2, max_size, max_size), dtype=np.float32)
        filled_typeAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[2]

        filled_orderAtt = np.zeros((2, max_size, max_size), dtype=np.float32)
        filled_orderAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[3]

        filled_aromAtt = np.zeros((2, max_size, max_size), dtype=np.float32)
        filled_aromAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[4]

        filled_conjAtt = np.zeros((2, max_size, max_size), dtype=np.float32)
        filled_conjAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[5]

        filled_ringAtt = np.zeros((2, max_size, max_size), dtype=np.float32)
        filled_ringAtt[:, 0:datum[0].shape[0], 0:datum[0].shape[0]] = datum[6]

        adj_list.append(filled_adj)
        afm_list.append(filled_afm)
        typeAtt_list.append(filled_typeAtt)
        orderAtt_list.append(filled_orderAtt)
        aromAtt_list.append(filled_aromAtt)
        conjAtt_list.append(filled_conjAtt)
        ringAtt_list.append(filled_ringAtt)

    if use_cuda:
        return ([torch.from_numpy(np.array(adj_list)).cuda(), torch.from_numpy(np.array(afm_list)).cuda(),
                 torch.from_numpy(np.array(typeAtt_list)).cuda(), torch.from_numpy(np.array(orderAtt_list)).cuda(),
                 torch.from_numpy(np.array(aromAtt_list)).cuda(), torch.from_numpy(np.array(conjAtt_list)).cuda(),
                 torch.from_numpy(np.array(ringAtt_list)).cuda(),FloatTensor(label_list),
                 torch.from_numpy(np.array(size_list)).cuda(), torch.from_numpy(np.array(index_list)).cuda()])
    else:
        return ([torch.from_numpy(np.array(adj_list)), torch.from_numpy(np.array(afm_list)),
             torch.from_numpy(np.array(typeAtt_list)),torch.from_numpy(np.array(orderAtt_list)),
                 torch.from_numpy(np.array(aromAtt_list)), torch.from_numpy(np.array(conjAtt_list)),
                 torch.from_numpy(np.array(ringAtt_list)),FloatTensor(label_list),
                 torch.from_numpy(np.array(size_list)), torch.from_numpy(np.array(index_list))])
def construct_dataset(x_all, y_all):
    output = []
    for i in range(len(x_all)):
        output.append(MolDatum(eval(x_all[i]), y_all[i], i))
    return(output)
class MolDatum():
    """
        Class that represents a train/validation/test datum
        - self.label: 0 neg, 1 pos -1 missing for different target.
    """
    def __init__(self, x, label, index):


        self.adj = np.array(x[1])
        self.afm =  np.array(x[0])
        self.bft =  np.array(x[2])
        self.orderAtt =  np.array(x[3])
        self.aromAtt = np.array (x[4])
        self.conjAtt =  np.array(x[5])
        self.ringAtt =   np.array(x[6])
        self.label =  ast.literal_eval(label)
        self.index = index
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

        label = self.data_list[key].label

        index = self.data_list[key].index
        return (adj, afm, TypeAtt, orderAtt, aromAtt, conjAtt, ringAtt, label,index)
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

   
