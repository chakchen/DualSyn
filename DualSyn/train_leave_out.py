import random
import torch.nn.functional as F
import torch.nn as nn
from models.dualsyn_leave_out import DualSyn
from utils_test import *
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
from sklearn import metrics
from creat_data_DC import creat_data
import pandas as pd
import datetime
import argparse

parser = argparse.ArgumentParser(description='Process some floats.')

parser.add_argument('--leave_type', type=str, help='The type of leaveout')
parser.add_argument('--dropping_method', type=str, help='The type of drop')
parser.add_argument('--dropout_rate', type=float, help='The dropout rate')
parser.add_argument('--device_num', type=int, help='The number of device')


args = parser.parse_args()
            

leave_type = args.leave_type
dropping_method = args.dropping_method
dropout_rate = args.dropout_rate
device_num = args.device_num


# print(f'leave_type:{leave_type}, dropout_method:{dropping_method}, dropout_rate: {dropout_rate}, device_num: {device_num}')


result_name = 'DualSyn_'+str(leave_type)+'_'+str(dropping_method)+"_drop_rate="+str(dropout_rate)

modeling = DualSyn


# training function at each epoch
def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch):
    print('Training on {} samples...'.format(len(drug1_loader_train.dataset)))
    model.train()
    # train_loader = np.array(train_loader)

    for batch_idx, data in enumerate(zip(drug1_loader_train, drug2_loader_train)): 
        data1 = data[0]
        data2 = data[1]
        data1 = data1.to(device)
        data2 = data2.to(device)
        #y = data[0].y.view(-1, 1).long().to(device)
        y = data[0].y.view(-1, 1).float().to(device)
        y = y.squeeze(1)
        optimizer.zero_grad()
        output = model(data1, data2)
        loss = loss_fn(output, y)
        # print('loss', loss)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data1.x),
                                                                           len(drug1_loader_train.dataset),
                                                                           100. * batch_idx / len(drug1_loader_train),
                                                                           loss.item()))


def predicting(model, device, drug1_loader_test, drug2_loader_test):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(drug1_loader_test.dataset)))
    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1 = data[0]
            data2 = data[1]
            data1 = data1.to(device)
            data2 = data2.to(device)
            output = model(data1, data2)
            #ys = F.softmax(output, 1).to('cpu').data.numpy()
            ys = output.to('cpu').data.numpy()
            #predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_labels = list(map(lambda x: int(x>0.5), ys))
            #predicted_scores = list(map(lambda x: x[1], ys))
            predicted_scores = list(map(lambda x: x, ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(len(dataset) * ratio)
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


# CPU or GPU

if torch.cuda.is_available():
    device = torch.device('cuda:'+str(device_num))
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')


TRAIN_BATCH_SIZE = 1024
TEST_BATCH_SIZE = 1024
LR = 0.001
LOG_INTERVAL = 20
NUM_EPOCHS = 200

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

cellfile = 'data/cell_features_954.csv'  
drug_smiles_file = 'data/smiles.csv'                  

if leave_type == 'leave_drug':
    train_datafile = ['data/leave_drug/leave_d00.csv',
                        'data/leave_drug/leave_d11.csv',
                        'data/leave_drug/leave_d22.csv',
                        'data/leave_drug/leave_d33.csv',
                        'data/leave_drug/leave_d44.csv']   
    train_pt_dataset = ['leave_drug_d00', 'leave_drug_d11', 'leave_drug_d22', 'leave_drug_d33', 'leave_drug_d44']   
    test_datafile = ['data/leave_drug/d00.csv',
                        'data/leave_drug/d11.csv',
                        'data/leave_drug/d22.csv',
                        'data/leave_drug/d33.csv',
                        'data/leave_drug/d44.csv']
    test_pt_result_dataset = ['drug_d00', 'drug_d11', 'drug_d22', 'drug_d33', 'drug_d44']
    fold_num = 5
elif leave_type == 'leave_comb':
    train_datafile = ['data/leave_comb/leave_c00.csv',
                        'data/leave_comb/leave_c11.csv',
                        'data/leave_comb/leave_c22.csv',
                        'data/leave_comb/leave_c33.csv',
                        'data/leave_comb/leave_c44.csv']    
    train_pt_dataset = ['leave_comb_c00', 'leave_comb_c11', 'leave_comb_c22', 'leave_comb_c33', 'leave_comb_c44']     
    test_datafile = ['data/leave_comb/c00.csv',
                        'data/leave_comb/c11.csv',
                        'data/leave_comb/c22.csv',
                        'data/leave_comb/c33.csv',
                        'data/leave_comb/c44.csv']
    test_pt_result_dataset = ['comb_c00', 'comb_c11', 'comb_c22', 'comb_c33', 'comb_c44']
    fold_num = 5
elif leave_type == 'leave_cell':
    train_datafile = ['data/leave_cell/leave_breast.csv',
                        'data/leave_cell/leave_colon.csv',
                        'data/leave_cell/leave_lung.csv',
                        'data/leave_cell/leave_melanoma.csv',
                        'data/leave_cell/leave_ovarian.csv',
                        'data/leave_cell/leave_prostate.csv']    
    train_pt_dataset = ['leave_cell_breast', 'leave_cell_colon', 'leave_cell_lung', 'leave_cell_melanoma', 'leave_cell_ovarian', 'leave_cell_prostate']   
    test_datafile = ['data/leave_cell/breast.csv',
                        'data/leave_cell/colon.csv',
                        'data/leave_cell/lung.csv',
                        'data/leave_cell/melanoma.csv',
                        'data/leave_cell/ovarian.csv',
                        'data/leave_cell/prostate.csv']
    test_pt_result_dataset = ['cell_breast', 'cell_colon', 'cell_lung', 'cell_melanoma', 'cell_ovarian', 'cell_prostate']
    fold_num = 6

for i in range(fold_num):

    train_drug1, train_drug2, train_cell, train_label, smile_graph, cell_features = creat_data(train_datafile[i], drug_smiles_file, cellfile)
    train_drug1_data = TestbedDataset(dataset=train_pt_dataset[i] + '_drug1', xd=train_drug1, xt=train_cell, y=train_label, smile_graph=smile_graph, xt_featrue=cell_features)
    train_drug2_data = TestbedDataset(dataset=train_pt_dataset[i] + '_drug2', xd=train_drug2, xt=train_cell, y=train_label, smile_graph=smile_graph, xt_featrue=cell_features)
    print('src_new_labels_0_10[0]', train_drug1_data[0])
    lenth = len(train_drug1_data)
    random_num = random.sample(range(0, lenth), lenth)
    drug1_data = train_drug1_data[random_num]
    drug2_data = train_drug2_data[random_num]


    test_drug1, test_drug2, test_cell, test_label, smile_graph, cell_features = creat_data(test_datafile[i], drug_smiles_file, cellfile)
    test_drug1_data = TestbedDataset(dataset=test_pt_result_dataset[i] + '_drug1', xd=test_drug1, xt=test_cell, y=test_label, smile_graph=smile_graph, xt_featrue=cell_features)
    test_drug2_data = TestbedDataset(dataset=test_pt_result_dataset[i] + '_drug2', xd=test_drug2, xt=test_cell, y=test_label, smile_graph=smile_graph, xt_featrue=cell_features)
    lenth = len(test_drug1_data)

    # build training set
    drug1_loader_train = DataLoader(drug1_data, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
    drug2_loader_train = DataLoader(drug2_data, batch_size=TRAIN_BATCH_SIZE, shuffle=None)

    # build test set
    drug1_loader_test = DataLoader(test_drug1_data, batch_size=TEST_BATCH_SIZE, shuffle=None)
    drug2_loader_test = DataLoader(test_drug2_data, batch_size=TEST_BATCH_SIZE, shuffle=None)

    model = modeling().to(device)
    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")

    folder_path = './result/' + result_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_AUCs = folder_path + '/'+ result_name + '_' + str(i) + '--AUCs--' + test_pt_result_dataset[i] + '_' + time_str + '.txt'
    AUCs = ('Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    best_auc = 0
    for epoch in range(NUM_EPOCHS):
        train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1)
        T, S, Y = predicting(model, device, drug1_loader_test, drug2_loader_test)
        # T is correct label
        # S is predict score
        # Y is predict label

        # compute preformence
        AUC = roc_auc_score(T, S)
        precision, recall, threshold = metrics.precision_recall_curve(T, S)
        PR_AUC = metrics.auc(recall, precision)
        BACC = balanced_accuracy_score(T, Y)
        tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
        TPR = tp / (tp + fn)
        PREC = precision_score(T, Y)
        ACC = accuracy_score(T, Y)
        KAPPA = cohen_kappa_score(T, Y)
        recall = recall_score(T, Y)


        # save data

        if best_auc < AUC:
            best_auc = AUC
            # torch.save(model.state_dict(), model_file_name)
            # independent_num = []
            # independent_num.append(test_num)
            # independent_num.append(T)
            # independent_num.append(Y)
            # independent_num.append(S)
            # txtDF = pd.DataFrame(data=independent_num)
            # txtDF.to_csv(result_file_name, index=False, header=False)
            AUCs = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA]
            save_AUCs(AUCs, file_AUCs)
        print('best_auc', best_auc)
    save_AUCs("best_auc:"+str(best_auc), file_AUCs)
