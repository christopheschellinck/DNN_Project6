import numpy as np
import os
import pandas as pd
import pdb
import shutil
import random
import glob
import matplotlib.pyplot as plt

# Organize data into train, valid, test dirs
#we have 629 images in the Dangerous folder, and 2271 in the not-Dangerous folder
#first: I want to create blanced classes, so I will take only 629 from the not-Dangerous folder
#second: I will move 440 images from Danerous and not-Dangerous to the train folder
#        I will move 126 images to the valid folder, and
#        I will move 63 images to the test folder



############
## Working on the Dangerous Folder
############
#move images 440 randomly from the Dangerous folder to the train/Dangerous folder
to_be_moved_train = random.sample(glob.glob("/home/saba/Downloads/skin cancer/Dangerous/*.BMP"), 440)
for f in enumerate(to_be_moved_train, 1):
   # print(f)
    dest = "/home/saba/Downloads/skin cancer/train/Dangerous"
    if not os.path.exists(dest):
        os.makedirs(dest)
    shutil.move(f[1], dest) #move an image

#move 126 images to the valid
to_be_moved_valid = random.sample(glob.glob("/home/saba/Downloads/skin cancer/Dangerous/*.BMP"), 126)
for f in enumerate(to_be_moved_valid, 1):
    dest = "/home/saba/Downloads/skin cancer/valid/Dangerous"
    if not os.path.exists(dest):
        os.makedirs(dest)
    shutil.move(f[1], dest) #move an image

# move 63 images to the tset
to_be_moved_test = random.sample(glob.glob("/home/saba/Downloads/skin cancer/Dangerous/*.BMP"), 63)
for f in enumerate(to_be_moved_test, 1):
    dest = "/home/saba/Downloads/skin cancer/test/Dangerous"
    if not os.path.exists(dest):
        os.makedirs(dest)
    shutil.move(f[1], dest)  # move an image




############
## Working on the not-Dangerous Folder
############
#move images 440 randomly from the Dangerous folder to the train/Dangerous folder
to_be_moved_train_not = random.sample(glob.glob("/home/saba/Downloads/skin cancer/not_Dangerous/*.BMP"), 440)
for f in enumerate(to_be_moved_train_not, 1):
   # print(f)
    dest = "/home/saba/Downloads/skin cancer/train/not_Dangerous"
    if not os.path.exists(dest):
        os.makedirs(dest)
    shutil.move(f[1], dest) #move an image

#move 126 images to the valid
to_be_moved_valid_not = random.sample(glob.glob("/home/saba/Downloads/skin cancer/not_Dangerous/*.BMP"), 126)
for f in enumerate(to_be_moved_valid_not, 1):
    dest = "/home/saba/Downloads/skin cancer/valid/not_Dangerous"
    if not os.path.exists(dest):
        os.makedirs(dest)
    shutil.move(f[1], dest) #move an image

# move 63 images to the tset
to_be_moved_test_not = random.sample(glob.glob("/home/saba/Downloads/skin cancer/not_Dangerous/*.BMP"), 63)
for f in enumerate(to_be_moved_test_not, 1):
    dest = "/home/saba/Downloads/skin cancer/test/not_Dangerous"
    if not os.path.exists(dest):
        os.makedirs(dest)
    shutil.move(f[1], dest)  # move an image


