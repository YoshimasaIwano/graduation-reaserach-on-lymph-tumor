import torch
from utils.system import train_model, evaluate_model
from utils.data_load import val_dataloader, train_dataloader, TumorDataset, ImageTransform, tumor_list
from utils.make_model import  model_resnet

def conduct_train():
    
    train_data_1 = train_dataloader(1)
    val_data_1 = val_dataloader(1)
    train_data_2 = train_dataloader(2)
    val_data_2 = val_dataloader(2)
    train_data_3 = train_dataloader(3)
    val_data_3 = val_dataloader(3)


    # dictionary ofject
    dataloaders_dict_1 = {"train": train_data_1, "val": val_data_1}
    dataloaders_dict_2 = {"train": train_data_2, "val": val_data_2}
    dataloaders_dict_3 = {"train": train_data_3, "val": val_data_3}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model, criterion, optimizer = model_resnet()
    # dataloader_dict = dataloaders_dict_1
    # model_ft = train_model(model, dataloader_dict, criterion, optimizer, num_epochs = 2)

    for i in range(3):
        model, criterion, optimizer = model_resnet()
        print('start' + str(i+1))
        if i == 0:
            dataloader_dict = dataloaders_dict_1
        elif i == 1:
            dataloader_dict = dataloaders_dict_2
        elif i == 2:
            dataloader_dict = dataloaders_dict_3
        model_ft = train_model(model, dataloader_dict, criterion, optimizer, num_epochs = 20)
        save_path = f'./outputs/exp_5/trial_{i}.pth'
        torch.save(model_ft.state_dict(), save_path)

if __name__ == '__main__':
    conduct_train()
    