import numpy as np
import SimpleITK as sitk
import glob
import cv2
import os
from PIL import Image

def extract_tumor(im, lb):
    # min and max range of a tumor
    min_row = np.min(lb.nonzero()[0])
    max_row = np.max(lb.nonzero()[0])
    min_col = np.min(lb.nonzero()[1])
    max_col = np.max(lb.nonzero()[1])
    
    row_lng = max_row - min_row
    col_lng = max_col - min_col

    # TODO: how to deal with or resize the larger ones
    # if row_lng > 100 or col_lng > 100:
    #     im_tmp = im[min_row: max_row, min_col: max_col]
    #     im_tmp = Image.fromarray(np.uint8(im_tmp))
    #     im_tumor = np.asarray(im_tmp.resize((100,100), Image.LANCZOS))
    # else:

    # the center of a tumor
    cen_row = int((max_row + min_row) / 2)
    cen_col = int((max_col + min_col) / 2)
    
    # image size is 100, so clip image with center +-50
    row_s = max(0,cen_row - 50)
    row_e = min(512,cen_row + 50)
    col_s = max(0,cen_col - 50)
    col_e = min(512,cen_col + 50)
    
    im_tumor = im[row_s: row_e, col_s: col_e]

    return im_tumor

def make_one_patient(num_dir):
    patient_num = num_dir.split('/')[-2]
    lb_dirs = glob.glob(num_dir + 'label_*/')
    im_dir = num_dir + 'image/'

    # 0006 is cancer, cannot classify either t and s
    if patient_num == '00006': 
        return 

    for i, lb_dir in enumerate(lb_dirs):
        lb_slice = sorted(glob.glob(lb_dir + '*.npy'))
        for j, lb_path in enumerate(lb_slice):
            im_path = im_dir + f'{str(j).zfill(3)}.npy'
            
            im = np.load(im_path)
            lb = np.load(lb_path)
            if np.all(lb == 0):
                continue

            # clip -150 ~ 150 on HU
            for i in range(len(im)):
                im[im < -150] = -150
                im[im > 150] = 150
            
            
            im_tumor = extract_tumor(im, lb)
            
            # lb_type eg)label_t_1, label_t_2
            lb_type = lb_dir.split('/')[-2]
            tumor_dir = f'./data/tumor_-150_150/{patient_num}/{lb_type}'
            os.makedirs(tumor_dir, exist_ok = True)
            tumor_npy = tumor_dir + '/' + f'{str(j).zfill(3)}.npy'
            np.save(tumor_npy, im_tumor)


def make_patch():
    num_dirs = sorted(glob.glob('./data/slices/*/'))

    for i, num_dir in enumerate(num_dirs):
        make_one_patient(num_dir)

