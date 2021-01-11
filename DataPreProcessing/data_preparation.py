"""
create a new folders Dangerous and not-Dangerous
Check the image according to csv file, if the image as class 0 or 1, then move this image to not-Danger
"""
import numpy as np
import os
import pandas as pd
import pdb
import shutil

url='/home/saba/Downloads/sc.csv'
df=pd.read_csv(url)

def Upper(x):
    try:
        x=str(x)
        return x.upper()
    except ValueError:
        return None

def to_cat(x):
    try:
        x=int(x)
        if x <= 1:
            return ('not_dangerous')
        elif x>=2:
            return ('dangerous')
    except ValueError:
        return None

# print(df['kat.Diagnose'].value_counts())
# pdb.set_trace()

#convert the name of image file to upp to match the images in the directory
df['id']=df['id'].apply(lambda x: Upper(x))
# 0,1 classes are not dangerous detection, 2, and 3 are dangerous
df['kat.Diagnose']=df['kat.Diagnose'].apply(lambda x: to_cat(x))
#drop duplicated
df.dropna(subset=['id','kat.Diagnose'], inplace=True)

# print(df['id'].value_counts())
# print(df['kat.Diagnose'].value_counts())
# print(df.info())

df_sub=df[['id','kat.Diagnose']]
#print(df_sub.head(20))


#extract the images from the
dir1='/home/saba/Downloads/skin cancer/SET_D'
dir2='/home/saba/Downloads/skin cancer/SET_E'
dir3='/home/saba/Downloads/skin cancer/SET_F'

#create a directory if it does not exist
if os.path.isdir('/home/saba/Downloads/skin cancer/Dangerous') is False:
    os.makedirs('/home/saba/Downloads/skin cancer/Dangerous')
if os.path.isdir('/home/saba/Downloads/skin cancer/not_Dangerous') is False:
    os.makedirs('/home/saba/Downloads/skin cancer/not_Dangerous')

##dir1
for img in os.listdir(dir1): #iterate over all images
    #get the image name without extension
    img_name=os.path.splitext(os.path.basename(img))[0]
    #CHECK if we have a class for the img_name
    if img_name in df_sub['id'].values:
        #get the class of img_name
        get_class_type=df_sub.loc[df_sub['id'] == img_name, 'kat.Diagnose'].iloc[0]
        #get a complete path with the image
        from_dir=os.path.join(dir1,img)
        #move to specific new class
        if get_class_type=='dangerous':
            shutil.move(from_dir, '/home/saba/Downloads/skin cancer/Dangerous')
        elif get_class_type=='not_dangerous':
            shutil.move(from_dir, '/home/saba/Downloads/skin cancer/not_Dangerous')
        else:
            continue
    else:
        print("not here in dir1")
        continue

##dir2, do the same for dir2
for img in os.listdir(dir2): #iterate over all images
    img_name=os.path.splitext(os.path.basename(img))[0]
    if img_name in df_sub['id'].values:
        get_class_type=df_sub.loc[df_sub['id'] == img_name, 'kat.Diagnose'].iloc[0]
        from_dir=os.path.join(dir2,img)
        if get_class_type=='dangerous':
            shutil.move(from_dir, '/home/saba/Downloads/skin cancer/Dangerous')
        elif get_class_type=='not_dangerous':
            shutil.move(from_dir, '/home/saba/Downloads/skin cancer/not_Dangerous')
        else:
            continue
    else:
        print("no")
        continue


#dir3
for img in os.listdir(dir3): #iterate over all images
    img_name=os.path.splitext(os.path.basename(img))[0]
    if img_name in df_sub['id'].values:
        get_class_type=df_sub.loc[df_sub['id'] == img_name, 'kat.Diagnose'].iloc[0]
        from_dir=os.path.join(dir3,img)
        if get_class_type=='dangerous':
            shutil.move(from_dir, '/home/saba/Downloads/skin cancer/Dangerous')
        elif get_class_type=='not_dangerous':
            shutil.move(from_dir, '/home/saba/Downloads/skin cancer/not_Dangerous')
        else:
            continue
    else:
        print("not in ")
        continue