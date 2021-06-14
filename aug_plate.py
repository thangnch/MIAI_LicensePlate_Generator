import random
from PIL import Image 
import cv2
import numpy as np
from shutil import copyfile
import glob
import shutil
import os
import gc
import multiprocessing

rect_num_noise=10
rect_r_noise = 5
rect_skew_amt = 8

square_num_noise=10
square_r_noise=5
square_skew_amt = 8

num_noise = 0
r_noise = 0
skew_amt = 0

CTC_w= 256
CTC_h= 32


def split_plate(img):
    height = img.shape[0]
    divide_height = int(height // 2) # 440/2 = 220

    img_up = img[0:divide_height,:]
    img_dw = img[divide_height:,:]

    img = np.concatenate((img_up,img_dw),axis=1)
    return img

def augmention(img):

    if random.random() > 0.7:
        img = np.array(img)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = Image.fromarray(img)

    return img

def copy_org(image_org_filepath, image_aug_filepath, plate_class):

    print("Copy:", image_org_filepath)

    org_img = cv2.imread(image_org_filepath, cv2.IMREAD_GRAYSCALE)

    ret, org_img = cv2.threshold(org_img, 127, 255, cv2.THRESH_BINARY)

    org_img = randomNoise(org_img, num_noise, r_noise)

    #if plate_class != "R":
    #    org_img = split_plate(org_img)

    #org_img = cv2.resize(org_img, (CTC_w, CTC_h))

    cv2.imwrite(image_aug_filepath, org_img)
    return

def randomNoise(img_bg,num_dot,r):

    width, height = img_bg.shape[1],img_bg.shape[0]
    if random.random()>0.7:
        for i in range(random.randint(0,num_dot)):
            if random.random()>0.5:
                cv2.circle(img_bg,(random.randint(0,width),random.randint(0,height)),random.randint(0,r),(0,0,0),-1)
            else:
                x,y= random.randint(0, width), random.randint(0, height)
                cv2.line(img_bg,(x,y),(x+random.randint(0,r*2),y+random.randint(0,r*2)),(0,0,0),2)
    return img_bg

def randomNoise_white(img_bg,num_dot,r):

    width, height = img_bg.shape[1],img_bg.shape[0]
    if random.random()>0.6:
        for i in range(random.randint(0,num_dot)):
            if random.random()>0.5:
                cv2.circle(img_bg,(random.randint(0,width),random.randint(0,height)),random.randint(0,r),(255,255,255),-1)
            else:
                x,y= random.randint(0, width), random.randint(0, height)
                cv2.line(img_bg,(x,y),(x+random.randint(0,r*2),y+random.randint(0,r*2)),(255,255,255),2)
    return img_bg

def aug_skew(aug_id, aug_num, org_image, image_org_filepath, image_aug_path, label_org_path, label_aug_path, plate_class):
    idx = 0
    # Copy image
    copy_org(image_org_filepath, image_aug_path + os.path.basename(image_org_filepath).split(".")[0] + ".png",plate_class)

    #Copy label
    image_org_filename = os.path.basename(image_org_filepath).split(".")[0]
    copyfile(label_org_path + image_org_filename + ".txt", label_aug_path + image_org_filename + ".txt")

    while idx < aug_num:
        value = skew_amt
        # ----------------- Wrap image
        # Build org_corner
        org_corner = np.float32([[0, 0], [0, org_image.shape[0]], [org_image.shape[1], org_image.shape[0]], [org_image.shape[1], 0]]) # 4 point

        # Build new_corner
        top_left = [random.uniform(-value, value), random.uniform(-value, value)]
        top_right = [org_image.shape[1] - random.uniform(-value, value), random.uniform(-value, value)]
        bottom_left = [random.uniform(-value, value), org_image.shape[0] - random.uniform(-value, value)]
        bottom_right = [org_image.shape[1] + random.uniform(-value, value), org_image.shape[0] + random.uniform(-value, value)]

        aug_corner = np.float32([top_left, bottom_left, bottom_right, top_right])

        # Wrap image
        M = cv2.getPerspectiveTransform(org_corner, aug_corner)
        aug_img = cv2.warpPerspective(org_image, M, (org_image.shape[1], org_image.shape[0]),borderValue=255)

        # Set file name
        image_aug_filename = image_org_filename + "_" + str(aug_id) + "_" + str(idx) + ".png"
        label_aug_filename = image_org_filename + "_" + str(aug_id) + "_" + str(idx) + ".txt"


        copyfile(label_org_path + image_org_filename + ".txt", label_aug_path + label_aug_filename)

        ret, aug_img = cv2.threshold(aug_img, 127, 255, cv2.THRESH_BINARY)

        # noise
        aug_img = randomNoise(aug_img,num_noise,r_noise)

        # Split and write

        #if plate_class!="R":
        #    aug_img = split_plate(aug_img)

        #aug_img = cv2.resize(aug_img,(CTC_w,CTC_h))
        cv2.imwrite(image_aug_path + image_aug_filename, aug_img)

        idx = idx + 1

    del aug_img, aug_corner, ret
    gc.collect()

    return


def my_rmtree(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
    os.rmdir(directory)
    return

def aug_plate(start_idx, end_idx, image_org_path, image_aug_path, label_org_path,label_aug_path,plate_class):

    for image_id in range(start_idx,end_idx):
        img_org_filepath  =  image_org_path + "syn_" + str(image_id) + ".png"
        org_img = cv2.imread(img_org_filepath, 0)

        aug_id = "skew"
        aug_num = 10
        aug_skew(aug_id, aug_num, org_img, img_org_filepath, image_aug_path, label_org_path, label_aug_path,plate_class)

        print("Aug ",image_id)
        # if count==2:
        #   return
        del org_img

    #del org_img
    gc.collect()
    return


def data_augment(type, total_plate, plate_per_process , num_process, plate_class):
    #type=train or test

    global num_noise, r_noise, skew_amt

    org_path = None
    aug_path = None

    if plate_class=="R":
        org_path = "data_rectangle/org/" + type + "/"
        aug_path = "data_rectangle/aug/" + type + "/"
        num_noise = rect_num_noise
        r_noise = rect_r_noise
        skew_amt = rect_skew_amt

    if plate_class=="S":
        org_path = "data_square/org/" + type + "/"
        aug_path = "data_square/aug/" + type + "/"
        num_noise = square_num_noise
        r_noise = square_r_noise
        skew_amt = square_skew_amt

    if plate_class=="M":
        org_path = "data_motobike/org/" + type + "/"
        aug_path = "data_motobike/aug/" + type + "/"
        num_noise = square_num_noise
        r_noise = square_r_noise
        skew_amt = square_skew_amt

    image_org_path = org_path + "images/"
    label_org_path = org_path + "labels/"

    image_aug_path = aug_path  + "images/"
    label_aug_path = aug_path  + "labels/"

    print("Remove ",image_aug_path)

    try:
        my_rmtree(image_aug_path)
    except:
        pass

    print("Remove ", label_aug_path)
    try:
        my_rmtree(label_aug_path)
    except:
        pass

    print('Aug Removed !')

    if not os.path.exists(image_aug_path):
        os.mkdir(image_aug_path)
    if not os.path.exists(label_aug_path):
        os.mkdir(label_aug_path)

    print("Total plate=", total_plate)
    print("Num process = ", num_process)

    p_list = []
    for idx in range(0, num_process):

        start_idx = idx * plate_per_process
        end_idx = start_idx + plate_per_process

        if end_idx > total_plate:
            end_idx = total_plate

        p = multiprocessing.Process(target=aug_plate, args=(start_idx, end_idx, image_org_path, image_aug_path, label_org_path,label_aug_path, plate_class, ))
        p.start()

        p_list.append(p)

    for p in p_list:
        p.join()

    return

#data_augment()