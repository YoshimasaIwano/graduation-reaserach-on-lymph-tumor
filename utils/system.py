import numpy as np
import pandas as pd
import glob
from layers import resnet 
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms,datasets
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix,roc_curve,roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import scipy.stats as stats
from utils.data_load import TumorDataset, ImageTransform
cnt = 0



def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []
    
    # epoch loop
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        
        # train and val loop
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # train mode
            else:
                model.eval()   # val mode

            epoch_loss = 0.0  # sum of loss in each epoch
            epoch_corrects = 0  # number of correct in each epoch

            # mini batch learning
            for inputs, labels, ids, clinical_data in tqdm(dataloaders_dict[phase]):
                
                df_lb = []
                for i in range(len(labels)):
                    l = labels[i]
                    if l == 's':
                        df_lb.append(0)
                    elif l == 't':
                        df_lb.append(1)
                    # elif l == 'c':
                    #     df_lb.append(2)

                df_lbs = torch.from_numpy(np.asarray(df_lb))    # transform numpy to torch

                # send data to GPU if available
                inputs = inputs.to(device)
                clinical_data = clinical_data.to(device)
                df_lbs = df_lbs.to(device)

                # initialize optimizer
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, clinical_data)
                    loss = criterion(outputs, df_lbs)  # calculate loss
                    _, preds = torch.max(outputs, 1)  # predict label

                    # backpropagate if train
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # calculate results
                    epoch_loss += loss.item() * inputs.size(0)  # update sum of loss
                    # update accurate number of data
                    epoch_corrects += torch.sum(preds == df_lbs.data)

            # show loss and accuracy each epoch
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_acc_list.append(epoch_acc)
                train_loss_list.append(epoch_loss)
            else:
                val_acc_list.append(epoch_acc)
                val_loss_list.append(epoch_loss)

        # evaluate model function after the last train epoch
        if epoch == num_epochs-1:
            evaluate_model(model, dataloaders_dict['val'])

            #　graph for checking accuracy
            plt.figure()
            plt.plot(train_acc_list, label = 'train_acc' + str(cnt))
            plt.plot(val_acc_list, label = 'val_acc' + str(cnt))
            # plt.xlim(1,num_epochs)
            plt.title('learning rate')
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.legend()
            plt.savefig(f'outputs/learning_rate/exp{str(cnt)}.png')
            plt.close()

            #　graph for checking loss
            plt.figure()
            plt.plot(train_loss_list, label = 'train_loss' + str(cnt))
            plt.plot(val_loss_list, label = 'val_loss' + str(cnt))
            # plt.xlim(1,num_epochs)
            plt.title('loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.savefig(f'outputs/loss/exp{str(cnt)}.png')
            plt.close()

        
    
    return model

    
def evaluate_model(model, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_class = 2

    # val mode
    model.eval()

    # extract patient number and label ex) p_ids = 00001/label_t_00
    p_ids = []
    for inputs, labels, ids, clinical_data in tqdm(dataloader):
        p_ids.extend(ids)
    p_ids = sorted(list(set(p_ids)))

    pred_nums = []
    lb_nums = []
    pred_scores = []
    # wrong_id = []
    with torch.no_grad():
        for p_id in p_ids:
            path_list = sorted(glob.glob(f'./data/tumor_-150_150/{p_id}/*.npy'))
            val_dataset = TumorDataset(path_list, ImageTransform(), 'val')
            val_dataloader = DataLoader(val_dataset)
    
            pred_num = []
            lb_num = []

            # Initialize the prediction and label lists(tensors) 
            pred_lists = torch.zeros(0, dtype = torch.float, device = 'cpu')
            lb_lists = torch.zeros(0, dtype = torch.long, device = 'cpu')

            for i, (inputs, labels, ids, clinical_data) in enumerate(val_dataloader):
                df_lb = []
                for i in range(len(labels)):
                    l = labels[i]
                    if l == 's':
                        df_lb.append(0)
                    elif l == 't':
                        df_lb.append(1)
                    # elif l == 'c':
                    #     df_lb.append(2)

                # transform label into tensor and send it to device(gpu)
                df_lbs = torch.from_numpy(np.asarray(df_lb))
                clinical_data = clinical_data.to(device)
                inputs = inputs.to(device)
                df_lbs = df_lbs.to(device)

                # outputs is probability after softmax
                outputs = model(inputs, clinical_data)
                # print(outputs)

                # in majority decision, deicide 0 or 1 on each image
                # _, preds = torch.max(outputs, 1)
                # summarize (0, 1) on ROI
                # pred_lists = torch.cat([pred_lists, preds.view(-1).cpu()])

                # log function
                outputs_l = torch.log(outputs)

                # add list output_l after returning cpu
                pred_lists = torch.cat([pred_lists, outputs_l.cpu()])

                # accumulate answer
                lb_lists = torch.cat([lb_lists,df_lbs.view(-1).cpu()])


            # sum of log
            pred_list_sum = torch.sum(pred_lists, 0, keepdim = True)
            # decide 0 or 1 based on the bigger log value
            _, pred_num = torch.max(pred_list_sum, 1)
            
            # mode in patient (majority decision)   
            # pred_num = stats.mode(pred_lists.to('cpu').detach().numpy().copy())

            # make label answer
            lb_num = stats.mode(lb_lists.to('cpu').detach().numpy().copy()) 

            # predict probability on ROI (majority dicision)
            # pred_score = np.count_nonzero(pred_lists.to('cpu').detach().numpy().copy() == 1) / len(pred_lists.to('cpu').detach().numpy().copy()) 

            # TODO: something wrong there
            # predict probability on ROI 
            soft = nn.Softmax(dim =1)
            pred_soft = soft(pred_list_sum)
            # print(pred_soft[0].numpy())
            pred_score = pred_soft[0].numpy()[1]
            # print(pred_score)

            # convert value of mode into DataFrame ex) ModeResult(mode=array([0]), count=array([27]))
            # pred_num = pd.DataFrame(pred_num[1].numpy())

            # create DataFrame predict and answer(label)
            pred_num = pd.DataFrame(pred_num.numpy())
            lb_num = pd.DataFrame(lb_num[0])
            
            # add pred and label on ROI into list 
            pred_nums.extend(pred_num.values)
            lb_nums.extend(lb_num.values)

            # record mistook ROI number
            # if pred_num.values != lb_num.values:
            #     wrong_id.append(p_id)

            # for ROC curve
            pred_scores.append(pred_score)

            
    # Confusion matrix
    conf_mat=confusion_matrix(np.array(lb_nums), np.array(pred_nums))
    print(conf_mat)

    # Per-class accuracy
    class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
    print('class_accuracy', class_accuracy)

    # Per_pred_class accuracy
    pred_class_acc = 100*conf_mat.diagonal()/conf_mat.sum(0)
    print('prediction_class_accuracy', pred_class_acc)

    # DICE score
    dice_score = 2 * conf_mat[1][1] / (2 * conf_mat[1][1] + conf_mat[1][0] + conf_mat[0][1]) 
    print('DICE_SCORE', dice_score)

    # ROI accuracy
    p_acc = 100 * conf_mat.diagonal().sum() / conf_mat.sum()
    print('ROI_accuracy', p_acc)

    # wrong id and feature
    # print('Wrong cases', wrong_id)

    # ROC curve
    global cnt
    fpr, tpr, thresholds = roc_curve(lb_nums, pred_scores)
    label_name = 'ROC_Curve_' + str(cnt)
    plt.figure()
    plt.plot(fpr, tpr, marker='o', label = label_name)
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.grid()
    plt.legend()
    plt.savefig(f'outputs/ROC/exp_5/sklearn_roc_curve_50layer_trial_{str(cnt)}.png')
    plt.close()

    auc_score = roc_auc_score(lb_nums, pred_scores)
    print('AUC_SCORE', auc_score)
    cnt += 1

