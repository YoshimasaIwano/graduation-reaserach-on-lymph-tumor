
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from layers import resnet 

def model_resnet():
    #TODO: standarization.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # information of device(cpu / gpu)
    # print(device)
    model = resnet.resnet50(pretrained = False)

    # send model to gpu 
    model.to(device)

    # configuration of model 
    # print(model)

    # add coefficient before each class in loss function 
    # weights = torch.tensor([0.453, 0.547]).cuda()
    # weights = torch.tensor([1.829700, 2.205255]).cuda()weight = weights
    criterion = nn.CrossEntropyLoss()

    # check parameters
    # for para in model.parameters():
    #     print(para)

    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

    return model, criterion, optimizer