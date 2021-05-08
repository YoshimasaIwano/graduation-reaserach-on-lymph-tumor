import numpy as np
import SimpleITK as sitk
import glob
import cv2
import os

def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def load(path):
    im = sitk.ReadImage(path)
    im_arr = sitk.GetArrayFromImage(im)
    return im, im_arr

def get_data():
    data_dir = '../testicular/data/testicular_v2/'
    pathlist = sorted(glob.glob(data_dir + '*/DICOMDAT/*.nii.gz'))  # sort
    pathlist_add = glob.glob(data_dir + '*/DICOMDAT/*.nrrd')    # to add .nrrd file
    pathlist.extend(pathlist_add)
    pathlist = sorted(pathlist)
    # for i in range(len(pathlist)):
    #     print(pathlist[i])
    datalist = []
    cnt = 0
    tumor_cnt = 0
    
    """
    cnt: number of patient
    tumor_cnt: number of tumor
    """
    for i,path in enumerate(pathlist):
        im, imarr = load(path)
        if 'Segmentation' in path:
            for j, imslice in enumerate(imarr):
                dir_path = f'../testicular/data/slices/{cnt:05}/label_s_{tumor_cnt:02}'
                makedirs(dir_path)
                new_save_path = f'../testicular/data/slices/{cnt:05}/label_s_{tumor_cnt:02}/{str(j).zfill(3)}.npy'
                np.save(new_save_path, imslice)
            tumor_cnt += 1       
        elif 'teratoma' in path:
            for j, imslice in enumerate(imarr):
                dir_path = f'../testicular/data/slices/{cnt:05}/label_t_{tumor_cnt:02}'
                makedirs(dir_path)
                new_save_path = f'../testicular/data/slices/{cnt:05}/label_t_{tumor_cnt:02}/{str(j).zfill(3)}.npy'
                np.save(new_save_path, imslice)       
            tumor_cnt += 1
        elif 'cancer' in path:
            for j, imslice in enumerate(imarr):
                dir_path = f'../testicular/data/slices/{cnt:05}/label_c_{tumor_cnt:02}'
                makedirs(dir_path)
                new_save_path = f'../testicular/data/slices/{cnt:05}/label_c_{tumor_cnt:02}/{str(j).zfill(3)}.npy'
                np.save(new_save_path, imslice)       
            tumor_cnt += 1
        else:
            cnt += 1
            tumor_cnt = 0
            dir_path = f'../testicular/data/slices/{cnt:05}'
            makedirs(dir_path)
            for j, imslice in enumerate(imarr):
                dir_path = f'../testicular/data/slices/{cnt:05}/image'
                makedirs(dir_path)
                new_save_path = f'../testicular/data/slices/{cnt:05}/image/{str(j).zfill(3)}.npy'
                np.save(new_save_path, imslice) 
        
        

    
