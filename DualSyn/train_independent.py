import random
import torch.nn.functional as F
import torch.nn as nn
from models.dualsyn_indepentent import DualSyn
from utils_test import *
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
from sklearn import metrics
from creat_data_DC import creat_data
import pandas as pd
import datetime
import argparse

parser = argparse.ArgumentParser(description='Process some floats.')

parser.add_argument('--dropping_method', type=str, help='The type of drop')
parser.add_argument('--dropout_rate', type=float, help='The dropout rate')
parser.add_argument('--device_num', type=int, help='The number of device')
parser.add_argument('--lr', type=float, help='The learning rate')

args = parser.parse_args()
            

dropping_method = args.dropping_method
dropout_rate = args.dropout_rate
device_num = args.device_num
lr = args.lr

# print(f'dropout_method:{dropping_method}, dropout_rate: {dropout_rate}, device_num: {device_num}')

result_name = 'DualSyn_independent_'+str(dropping_method)+"_drop_rate="+str(dropout_rate)+"_lr="+str(lr)

modeling = DualSyn

log_dir = "runs/"+result_name


# training function at each epoch
def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch):
    print("===============")
    print('Training on {} samples...'.format(len(drug1_loader_train.dataset)))
    model.train()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
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

        #loss = loss_fn(output, y)
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
        ys = output.to('cpu').data.numpy()
        #predicted_labels = list(map(lambda x: np.argmax(x), ys))
        predicted_labels = list(map(lambda x: int(x>0.5), ys))
        #predicted_scores = list(map(lambda x: x[1], ys))
        predicted_scores = list(map(lambda x: x, ys))
        total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
        total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
        total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()

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
LR = lr
LOG_INTERVAL = 20
NUM_EPOCHS = 300

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)


cellfile = 'data/cell_features_954.csv'  
drug_smiles_file = 'data/smiles.csv'                 
train_datafile = 'data/new_labels_0_10.csv'   
train_dataset = 'new_labels_0_10'
independent_datafile = 'data/independent/independent_input.csv'
datafile = 'independent_input'    


for i in range(5):
    train_drug1, train_drug2, train_cell, train_label, smile_graph, cell_features = creat_data(train_datafile, drug_smiles_file, cellfile)
    train_drug1_data = TestbedDataset(dataset=train_dataset + '_drug1', xd=train_drug1, xt=train_cell, y=train_label, smile_graph=smile_graph, xt_featrue=cell_features)
    train_drug2_data = TestbedDataset(dataset=train_dataset + '_drug2', xd=train_drug2, xt=train_cell, y=train_label, smile_graph=smile_graph, xt_featrue=cell_features)
    #print('src_new_labels_0_10[0]', train_drug1_data[0])
    lenth = len(train_drug1_data)
    random_num = random.sample(range(0, lenth), lenth)
    drug1_data = train_drug1_data[random_num]
    drug2_data = train_drug2_data[random_num]


    independent_drug1, independent_drug2, independent_cell, independent_label, smile_graph, cell_features = creat_data(
        independent_datafile, drug_smiles_file, cellfile)
    independent_drug1_data = TestbedDataset(dataset='independent_input_drug1', xd=independent_drug1, xt=independent_cell, y=independent_label, smile_graph=smile_graph, xt_featrue=cell_features)
    independent_drug2_data = TestbedDataset(dataset='independent_input_drug2', xd=independent_drug2, xt=independent_cell, y=independent_label, smile_graph=smile_graph, xt_featrue=cell_features)
    lenth = len(independent_drug1_data)

    drug1_loader_train = DataLoader(drug1_data, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
    drug2_loader_train = DataLoader(drug2_data, batch_size=TRAIN_BATCH_SIZE, shuffle=None)

    independent_drug1_loader_test = DataLoader(independent_drug1_data, batch_size=TEST_BATCH_SIZE, shuffle=None)
    independent_drug2_loader_test = DataLoader(independent_drug2_data, batch_size=TEST_BATCH_SIZE, shuffle=None)

    model = modeling().to(device)
    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")

    folder_path = './result/' + result_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_AUCs = folder_path + '/'+ result_name + '_' + str(i) + '--AUCs--' + datafile + '_' + time_str + '.txt'
    AUCs = ('Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')


    best_auc = 0
    for epoch in range(NUM_EPOCHS):
        train_T, train_S, train_Y = train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1)
        T, S, Y = predicting(model, device, independent_drug1_loader_test, independent_drug2_loader_test)
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

        train_AUC = roc_auc_score(train_T, train_S)
        train_precision, train_recall, train_threshold = metrics.precision_recall_curve(T, S)
        train_PR_AUC = metrics.auc(train_recall, train_precision)
        train_ACC = accuracy_score(train_T, train_Y)
        
        print("Train: AUC={}, PR_AUC={}, ACC={}".format(train_AUC, train_PR_AUC, train_ACC))
        print("Test: AUC={}, PR_AUC={}, ACC={}".format(AUC, PR_AUC, ACC))

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
        print("\n")
    save_AUCs("best_auc:"+str(best_auc), file_AUCs)
