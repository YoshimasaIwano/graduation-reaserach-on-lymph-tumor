
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from layers import resnet 

def model_resnet():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet.resnet50(pretrained = False)

    # send model to gpu 
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

    return model, criterion, optimizer