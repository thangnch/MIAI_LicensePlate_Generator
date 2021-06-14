import cv2
import glob
import gc
import os
import random
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import progressbar
import math
from generate_image import *
from aug_plate import augmention,data_augment,split_plate
import multiprocessing
import argparse



available_number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
available_char = ['A', 'B', 'C', 'D','E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']
available_template_m = ['NN-CN/NNNN', 'NN-CC/NNNN','NN-CC/NNN.NN','NN-CN/NNN.NN']
available_template_s = ['NNC/NNNN', 'NNC/NNN.NN','NNCC/NNN.NN', 'NNCC/NNNN', 'NNC/NNNN', 'NNC/NNN.NN', 'NN-CC/NNN.NN', 'NNCC/NNNN','NNCC/NNN.NN', 'CC/NN.NN']
available_template_r = ['NNC-NNNN', 'NNC-NNN.NN','NN-CC-NNN-NN','NNCC-NNNN', 'NNC-NNNN', 'NNC-NNN.NN', 'NN-CC-NNN-NN', 'NNCC-NNNN','NNCC-NNN.NN', 'CC-NN-NN']

available_square_bg = glob.glob('background/square*.jpg')
available_rec_bg = glob.glob('background/rec*.jpg')

total_number = len(available_number)
total_char = len(available_char)




def generate_sample(template):
    count_numb = template.count('N')
    count_char = template.count('C')
    for i in range(count_numb):
        idx_numb = random.randint(0, total_number - 1)
        template = template.replace('N', available_number[idx_numb], 1)
    for i in range(count_char):
        idx_char = random.randint(0, total_char - 1)
        template = template.replace('C', available_char[idx_char], 1)
    return template

def generate_plate(template):
    if '/' in template:
        bg = available_square_bg[random.randint(0, len(available_square_bg) - 1)]
        return generate_2lines_images(template, bg)
    else:
        bg = available_rec_bg[random.randint(0, len(available_rec_bg) - 1)]
        return generate_1lines_image(template, bg)


def make_plate(start_idx, end_idx, available_template, plate_class ):
    for i in range(start_idx, end_idx):
        # try:
        print("Make plate", i)
        filename = os.path.join(args.output_dir, 'syn_{}.png'.format(i))
        labelname = os.path.join(args.label_dir, 'syn_{}.txt'.format(i))

        idx = random.randint(0, total_template - 1)
        template = available_template[idx]

        sample = generate_sample(template)
        img, _ = generate_plate(sample)
        img = augmention(img)

        #img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # Split if need
        #if plate_class!="R":
        #    img = split_plate(img)

        img = img.resize((256, 32), Image.ANTIALIAS)
        img.save(filename)

        labels = sample.replace('-', '').replace('.', '').replace('/', '')
        #cv2.imwrite(filename, img)

        with open(labelname, 'a') as f:
            f.write(labels)

    del img
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

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Vietnamese Synthesis License Plate.')

    parser.add_argument('-id', required=True,
                       help='S or R or M')
    parser.add_argument('-numb', required=True,
                       help='Total number of Synthesis images')
    parser.add_argument('-mode', required=True,
                        help='Total number of Synthesis images')
    parser.add_argument('-output_dir', default='',
                       help='Output directory')
    parser.add_argument('-label_dir', default='',
                       help='Output directory')

    args = parser.parse_args()

    if args.id == 'R':
        total_template = len(available_template_r)
    if args.id == 'S':
        total_template = len(available_template_s)
    if args.id == 'M':
        total_template = len(available_template_m)

    available_template = None


    # Set output dir
    if args.id == "R": # Bien dai
        args.output_dir = "data_rectangle/org/" + args.mode + "/images"
        args.label_dir = "data_rectangle/org/" + args.mode + "/labels"
        available_template = available_template_r

    if args.id == "S": # Bien vuong oto
        args.output_dir = "data_square/org/" + args.mode + "/images"
        args.label_dir = "data_square/org/" + args.mode + "/labels"
        available_template = available_template_s

    if args.id == "M": # Bien dai
        args.output_dir = "data_motobike/org/" + args.mode + "/images"
        args.label_dir = "data_motobike/org/" + args.mode + "/labels"
        available_template = available_template_m

    print(args)
    print("Remove ",args.output_dir)


    try:
        my_rmtree(args.output_dir)
    except:
        pass

    print("Remove ", args.label_dir)

    try:
        my_rmtree(args.label_dir)
    except:
        pass

    print('Removed!')

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.exists(args.label_dir):
        os.mkdir(args.label_dir)

    # Num process
    total_plate = int(args.numb)
    if args.mode == "train":
        plate_per_process = 5000
    else:
        plate_per_process = 2500

    num_process = total_plate // plate_per_process

    if total_plate % plate_per_process != 0:
        num_process = num_process + 1

    print("Total plate=", total_plate)
    print("Num process = ", num_process)

    p_list = []
    for idx in range(0, num_process):

        start_idx = idx * plate_per_process
        end_idx = start_idx + plate_per_process

        if end_idx > total_plate:
            end_idx = total_plate

        p = multiprocessing.Process(target=make_plate, args=(start_idx, end_idx,available_template,args.id,))
        p.start()

        p_list.append(p)

    for p in p_list:
        p.join()

    # Now augment
    print('Start augment !')
    data_augment(args.mode, total_plate, plate_per_process ,num_process, args.id)

    print('Completed !')
