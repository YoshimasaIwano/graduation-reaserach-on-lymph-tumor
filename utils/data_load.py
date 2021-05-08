import numpy as np
import pandas as pd
import glob
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms,datasets
import torch
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

val_num_1 = ['00001','00002','00003','00004','00005','00007','00008','00009','00010','00011','00012','00013','00014','00015','00016','00017','00018','00019','00020','00021','00025','00028']
train_num_1 = ['00022','00023','00024','00026','00027','00029','00030','00031','00032','00033','00034','00035','00036','00037','00038','00039','00040','00041','00042','00043','00044','00045','00046','00047','00048','00049','00050','00051','00052','00053','00054','00055','00056','00057','00058','00059','00060','00061','00062','00063','00064','00065','00066']

val_num_2 = ['00022','00023','00024','00026','00027','00029','00030','00031','00032','00033','00034','00035','00036','00037','00038','00039','00041','00049','00051']
train_num_2 = ['00001','00002','00003','00004','00005','00007','00008','00009','00010','00011','00012','00013','00014','00015','00016','00017','00018','00019','00020','00021','00025','00028','00040','00042','00043','00044','00045','00046','00047','00048','00050','00052','00053','00054','00055','00056','00057','00058','00059','00060','00061','00062','00063','00064','00065','00066']

val_num_3 = ['00040','00042','00043','00044','00045','00046','00047','00048','00050','00052','00053','00054','00055','00056','00057','00058','00059','00060','00061','00062','00063','00064','00065','00066']
train_num_3 = ['00001','00002','00003','00004','00005','00007','00008','00009','00010','00011','00012','00013','00014','00015','00016','00017','00018','00019','00020','00021','00022','00023','00024','00025','00026','00027','00028','00029','00030','00031','00032','00033','00034','00035','00036','00037','00038','00039','00041','00049','00051']

# clinical data
primary_lesions_s = {'00001': 1,'00002': 0,'00003': 1,'00004': 0,'00005': 0,'00007': 0,'00008': 0,'00009': 0,'00010': 0,'00011': 0,'00012': 0,'00013': 0,'00014': 1,'00015': 0,'00016': 1,'00017': 0,'00018': 0,'00019': 1,'00020': 1,'00021': 0,'00022': 0,'00023': 1,'00024': 1,'00025': 0,'00026': 1,'00027': 1,'00028': 0,'00029': 1,'00030': 1,'00031': 1,'00032': 1,'00033': 1,'00034': 0,'00035': 1,'00036': 0,'00037': 0,'00038': 0,'00039': 1,'00040': 0,'00041': 1,'00042': 0,'00043': 1,'00044': 0,'00045': 0,'00046': 1,'00047': 1,'00048': 0,'00049': 0,'00050': 0,'00051': 0,'00052': 0,'00053': 1,'00054': 0,'00055': 1,'00056': 1,'00057': 1,'00058': 0,'00059': 1,'00060': 0,'00061': 1,'00062': 0,'00063': 0,'00064': 0,'00065': 1,'00066': 0}
primary_lesions_e = {'00001': 1,'00002': 1,'00003': 0,'00004': 1,'00005': 1,'00007': 0,'00008': 1,'00009': 0,'00010': 0,'00011': 1,'00012': 1,'00013': 1,'00014': 1,'00015': 1,'00016': 1,'00017': 0,'00018': 1,'00019': 0,'00020': 0,'00021': 1,'00022': 0,'00023': 0,'00024': 0,'00025': 1,'00026': 1,'00027': 0,'00028': 0,'00029': 0,'00030': 0,'00031': 1,'00032': 1,'00033': 1,'00034': 0,'00035': 1,'00036': 0,'00037': 1,'00038': 0,'00039': 0,'00040': 1,'00041': 1,'00042': 0,'00043': 0,'00044': 1,'00045': 0,'00046': 0,'00047': 0,'00048': 1,'00049': 1,'00050': 0,'00051': 1,'00052': 0,'00053': 1,'00054': 1,'00055': 1,'00056': 0,'00057': 1,'00058': 0,'00059': 1,'00060': 0,'00061': 0,'00062': 1,'00063': 1,'00064': 0,'00065': 0,'00066': 0}
primary_lesions_ys = {'00001': 0,'00002': 1,'00003': 1,'00004': 1,'00005': 1,'00007': 0,'00008': 1,'00009': 0,'00010': 1,'00011': 1,'00012': 1,'00013': 0,'00014': 0,'00015': 1,'00016': 1,'00017': 0,'00018': 1,'00019': 0,'00020': 1,'00021': 0,'00022': 0,'00023': 1,'00024': 0,'00025': 1,'00026': 0,'00027': 0,'00028': 1,'00029': 0,'00030': 0,'00031': 0,'00032': 0,'00033': 1,'00034': 0,'00035': 0,'00036': 0,'00037': 0,'00038': 0,'00039': 0,'00040': 0,'00041': 0,'00042': 0,'00043': 0,'00044': 0,'00045': 0,'00046': 0,'00047': 0,'00048': 0,'00049': 0,'00050': 1,'00051': 0,'00052': 1,'00053': 0,'00054': 1,'00055': 1,'00056': 0,'00057': 0,'00058': 0,'00059': 0,'00060': 0,'00061': 1,'00062': 1,'00063': 1,'00064': 0,'00065': 0,'00066': 1}
primary_lesions_ch = {'00001': 0,'00002': 0,'00003': 0,'00004': 0,'00005': 0,'00007': 0,'00008': 0,'00009': 0,'00010': 0,'00011': 0,'00012': 0,'00013': 0,'00014': 0,'00015': 0,'00016': 0,'00017': 0,'00018': 0,'00019': 0,'00020': 0,'00021': 1,'00022': 0,'00023': 0,'00024': 0,'00025': 0,'00026': 0,'00027': 0,'00028': 0,'00029': 0,'00030': 0,'00031': 0,'00032': 0,'00033': 1,'00034': 0,'00035': 0,'00036': 0,'00037': 0,'00038': 0,'00039': 0,'00040': 0,'00041': 0,'00042': 0,'00043': 0,'00044': 0,'00045': 1,'00046': 0,'00047': 0,'00048': 0,'00049': 1,'00050': 0,'00051': 1,'00052': 1,'00053': 0,'00054': 0,'00055': 1,'00056': 1,'00057': 0,'00058': 0,'00059': 0,'00060': 0,'00061': 0,'00062': 0,'00063': 0,'00064': 0,'00065': 0,'00066': 0}
primary_lesions_t = {'00001': 1,'00002': 1,'00003': 0,'00004': 1,'00005': 1,'00007': 1,'00008': 1,'00009': 1,'00010': 0,'00011': 0,'00012': 0,'00013': 0,'00014': 0,'00015': 0,'00016': 0,'00017': 1,'00018': 0,'00019': 0,'00020': 1,'00021': 0,'00022': 0,'00023': 1,'00024': 0,'00025': 1,'00026': 0,'00027': 0,'00028': 1,'00029': 0,'00030': 0,'00031': 1,'00032': 0,'00033': 0,'00034': 0,'00035': 0,'00036': 0,'00037': 0,'00038': 1,'00039': 0,'00040': 0,'00041': 0,'00042': 0,'00043': 0,'00044': 0,'00045': 0,'00046': 0,'00047': 0,'00048': 0,'00049': 1,'00050': 1,'00051': 1,'00052': 1,'00053': 0,'00054': 1,'00055': 1,'00056': 0,'00057': 0,'00058': 0,'00059': 1,'00060': 0,'00061': 0,'00062': 0,'00063': 0,'00064': 0,'00065': 0,'00066': 0}
primary_lesions_b = {'00001': 0,'00002': 0,'00003': 0,'00004': 0,'00005': 0,'00007': 0,'00008': 0,'00009': 0,'00010': 0,'00011': 0,'00012': 0,'00013': 0,'00014': 0,'00015': 0,'00016': 0,'00017': 0,'00018': 0,'00019': 0,'00020': 0,'00021': 0,'00022': 1,'00023': 0,'00024': 0,'00025': 0,'00026': 0,'00027': 0,'00028': 0,'00029': 0,'00030': 0,'00031': 0,'00032': 0,'00033': 0,'00034': 1,'00035': 0,'00036': 1,'00037': 0,'00038': 0,'00039': 0,'00040': 0,'00041': 0,'00042': 1,'00043': 0,'00044': 0,'00045': 0,'00046': 0,'00047': 0,'00048': 0,'00049': 0,'00050': 0,'00051': 0,'00052': 0,'00053': 0,'00054': 0,'00055': 0,'00056': 0,'00057': 0,'00058': 1,'00059': 0,'00060': 1,'00061': 0,'00062': 0,'00063': 0,'00064': 1,'00065': 0,'00066': 0}

LDHs = {'00001': 473,'00002': 172,'00003': 2667,'00004': 429,'00005': 199,'00007': 1025,'00008': 130,'00009': 931,'00010': 150,'00011': 1249,'00012': 1117,'00013': 935,'00014': 2122,'00015': 947,'00016': 1860,'00017': 405,'00018': 154,'00019': 668,'00020': 942,'00021': 3155,'00022': 711,'00023': 312,'00024': 815,'00025': 442,'00026': 440,'00027': 3228,'00028': 180,'00029': 550,'00030': 440,'00031': 563,'00032': 349,'00033': 650,'00034': 371,'00035': 436,'00036': 399,'00037': 1086,'00038': 169,'00039': 4039,'00040': 407,'00041': 239,'00042': 340,'00043': 1175,'00044': 863,'00045': 709,'00046': 340,'00047': 367,'00048': 629,'00049': 821,'00050': 346,'00051': 355,'00052': 159,'00053': 457,'00054': 725,'00055': 828,'00056': 619,'00057': 758,'00058': 827,'00059': 179,'00060': 1032,'00061': 596,'00062': 25,'00063': 379,'00064': 266,'00065': 637,'00066': 449}
AFPs = {'00001': 471,'00002': 119,'00003': 2754,'00004': 6748,'00005': 216,'00007': 71,'00008': 20,'00009': 18995,'00010': 4,'00011': 90,'00012': 57,'00013': 57,'00014': 3,'00015': 24,'00016': 3092,'00017':11,'00018': 52,'00019': 180,'00020': 590,'00021': 79,'00022': 22580,'00023':2918,'00024': 12,'00025': 104,'00026': 2,'00027': 0.9,'00028': 5193,'00029': 15,'00030': 1346.7,'00031': 7907,'00032': 5,'00033': 40376,'00034': 2,'00035': 214,'00036': 52,'00037': 7.4,'00038': 401,'00039': 4.3,'00040': 6.2,'00041': 181,'00042': 1.2,'00043': 3.3,'00044': 31,'00045': 1.8,'00046': 2,'00047': 1.2,'00048': 5.2,'00049': 6000,'00050': 11000,'00051': 2.4,'00052': 73,'00053': 3,'00054': 46,'00055': 765,'00056': 2,'00057': 21198,'00058': 376,'00059': 139,'00060': 77.3,'00061': 492,'00062': 24.3,'00063': 17147,'00064': 3.7,'00065': 30.4,'00066': 48000}
HCGs = {'00001': 3220,'00002': 0.6,'00003': 89,'00004': 640,'00005': 0.4,'00007': 1100000,'00008': 3.9,'00009': 410,'00010': 0.4,'00011': 68,'00012': 280,'00013': 9035,'00014': 64,'00015': 510000,'00016': 10000,'00017': 0.5,'00018': 899,'00019': 12.8,'00020': 7900,'00021': 7139,'00022': 20507.8,'00023': 22.1,'00024': 5,'00025': 7668,'00026': 5606,'00027': 8.6,'00028': 2045,'00029': 253860,'00030': 0.7,'00031': 257000,'00032': 107,'00033': 1545,'00034': 224762,'00035': 0.7,'00036': 177937,'00037': 0.5,'00038': 1.3,'00039': 123000,'00040': 19639,'00041': 21,'00042': 73680,'00043': 29.8,'00044': 0.5,'00045': 8230000,'00046': 0.1,'00047': 4,'00048': 0.5,'00049': 68970,'00050': 0.5,'00051': 2.4,'00052': 113,'00053': 0.5,'00054': 223,'00055': 399000,'00056': 46656,'00057': 18026,'00058': 5515,'00059': 6051,'00060': 104108,'00061': 0.8,'00062': 0.1,'00063': 0.5,'00064': 275000,'00065': 643,'00066':0.5}

b_lymphs = {'00001': 40,'00002': 30,'00003': 100,'00004': 100,'00005': 30,'00007': 100,'00008': 10,'00009': 50,'00010': 10,'00011': 115,'00012': 55,'00013': 20,'00014': 55,'00015': 115,'00016': 55,'00017': 65,'00018': 40,'00019': 50,'00020': 30,'00021': 40,'00022': 150,'00023': 30,'00024': 70,'00025': 35,'00026': 80,'00027': 160,'00028': 130,'00029': 55,'00030': 70,'00031': 170,'00032': 20,'00033': 105,'00034': 50,'00035': 50,'00036': 75,'00037': 35,'00038': 29,'00039': 95,'00040': 40,'00041': 20,'00042': 55,'00043': 140,'00044': 35,'00045': 70,'00046': 50,'00047': 70,'00048': 30,'00049': 60,'00050': 30,'00051': 25,'00052': 25,'00053': 70,'00054': 10,'00055': 70,'00056': 105,'00057': 50,'00058': 110,'00059': 80,'00060': 110,'00061': 100,'00062': 20,'00063': 60,'00064': 65,'00065': 95,'00066': 80}
a_lymphs = {'00001': 30,'00002': 20,'00003': 30,'00004': 50,'00005': 10,'00007': 40,'00008': 10,'00009': 25,'00010': 10,'00011': 100,'00012':30,'00013': 10,'00014': 10,'00015': 20,'00016': 10,'00017': 35,'00018': 40,'00019': 30,'00020': 10,'00021': 25,'00022': 60,'00023': 10,'00024': 10,'00025': 20,'00026': 35,'00027': 60,'00028': 110,'00029': 20,'00030': 30,'00031': 220,'00032': 20,'00033': 60,'00034': 25,'00035': 20,'00036': 50,'00037': 20,'00038': 40,'00039': 10,'00040': 10,'00041': 10,'00042': 10,'00043': 10,'00044': 20,'00045': 45,'00046': 25,'00047': 50,'00048': 10,'00049': 40,'00050': 10,'00051': 10,'00052': 10,'00053': 35,'00054': 10,'00055': 65,'00056': 20,'00057': 25,'00058': 20,'00059': 50,'00060': 60,'00061': 50,'00062': 10,'00063': 20,'00064': 30,'00065': 30,'00066': 30}



def tumor_list(version):
    """
    version: cross validation version and train or val
    """
    path_list = []
    for i in version:
        paths = sorted(glob.glob(f'./data/tumor_-150_150/{i}/label_*/*.npy'))
        path_list.extend(paths)

    # number of benign or malignant
    # s_list = [s for s in path_list if 'label_s' in s]
    # print(len(s_list))
    # t_list = [s for s in path_list if 'label_t' in s]
    # print(len(t_list))

    # number of data
    # print(len(path_list))  
    
    return path_list

class ImageTransform():
    def __init__(self):
        self.data_transform = {
            'train': transforms.Compose([
                # transforms.RandomResizedCrop(
                #     resize, scale=(0.5, 1.0)),  # augmentation
                # transforms.RandomHorizontalFlip(),  # augmentation
                
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=1),
                # transforms.Resize(256),
                transforms.ToTensor(),  # convert into tensor [0, 1] (regularization)
                # transforms.Normalize(mean = [0.640], std = [0.255])  # standarlization convert into [-1, 1]
            ]),
            'val': transforms.Compose([
                
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=1),
                # transforms.Resize(256),
                transforms.ToTensor(),  # convert into tensor
                # transforms.Normalize(mean = [0.640], std = [0.255])  # standarlization
            ])
        }

    def __call__(self, data, phase):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            designate train or val mode
        """
        return self.data_transform[phase](data)

class TumorDataset(Dataset):
    """
    data preprocessing
    """
    def __init__(self, path_list, transform, phase):
        self.path_list = path_list
        self.transform = transform
        self.phase = phase
        

    def __len__(self):
        # number of images
        return len(self.path_list)

    def __getitem__(self, index):
        path = self.path_list[index]
        tmp_data = np.load(path)

        # iamge size 100×100
        # if index == 0:
        #     print(len(tmp_data))

        # min of pixel value
        # print(tmp_data.min())

        # preprocessing
        transformed_data = self.transform(tmp_data, self.phase)

        if 'label_s' in path:
            lb = 's'
        elif 'label_t' in path:
            lb = 't'

        # patient id + type of tumor
        p_num = path.split('/')[-3]
        p_fea = path.split('/')[-2]
        p_id = p_num + '/' + p_fea

        # add clinical data
        primary_lesion_s = primary_lesions_s[p_num]
        primary_lesion_e = primary_lesions_e[p_num]
        primary_lesion_ys = primary_lesions_ys[p_num]
        primary_lesion_ch = primary_lesions_ch[p_num]
        primary_lesion_t = primary_lesions_t[p_num]
        primary_lesion_b = primary_lesions_b[p_num]

        LDH = LDHs[p_num]
        AFP = AFPs[p_num]
        HCG = HCGs[p_num]

        b_lymph = b_lymphs[p_num]
        a_lymph = a_lymphs[p_num]

        clinical_data = [primary_lesion_s, primary_lesion_e, primary_lesion_ys, primary_lesion_ch, primary_lesion_t, primary_lesion_b, LDH, AFP, HCG, b_lymph, a_lymph]
        
        # #  to standarlize,  numpy&reshape
        # clinical_data = np.array(clinical_data)
        # clinical_data = clinical_data.reshape(-1,1)
        
        # # standarlization
        # scaler = StandardScaler()
        # clinical_data = scaler.fit_transform(clinical_data)

        # # return tensor
        # clinical_data = clinical_data.reshape(-1)
        clinical_data = torch.tensor(clinical_data).float()


        # image size after transforming
        # if index == 0:
        #     print(len(transformed_data[0]))
        
        return transformed_data, lb, p_id, clinical_data
                
def train_dataloader(version):
    """
    train のdataloader
    """
    phase = 'train'
    batch_size = 32
    if version == 1:
        path_list = tumor_list(train_num_1)
    elif version == 2:
        path_list = tumor_list(train_num_2)
    elif version == 3:
        path_list = tumor_list(train_num_3)
    
    train_dataset = TumorDataset(path_list, ImageTransform(), phase)
    # ？
    # print(train_dataset.__getitem__(0)[0])
    train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)
    
    return train_dataloader

def val_dataloader(version):
    """
    val のdataloader
    """
    phase = 'val'
    batch_size = 32
    if version == 1:
        path_list = tumor_list(val_num_1)
    elif version == 2:
        path_list = tumor_list(val_num_2)
    elif version == 3:
        path_list = tumor_list(val_num_3)
    
    val_dataset = TumorDataset(path_list, ImageTransform(), phase)
    val_dataloader = DataLoader(val_dataset, shuffle = True, batch_size = batch_size)
    
    return val_dataloader