#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
Created on  Feb 2th,2018

@author: hongrui
用来对图片进行整理重命名


"""

import cv2
import os
import numpy as np
import tensorflow as tf
import sys

N_CLASSES = 2  # 2个输出神经元，［1，0］ 或者 ［0，1］猫和狗的概率
IMG_W = 208  # 重新定义图片的大小，图片如果过大则训练比较慢  
IMG_H = 208  
BATCH_SIZE = 4  #每批数据的大小
CAPACITY = 256

# def get_batch(image,label,image_W,image_H,batch_size,capacity):
#     # 转换数据为 ts 能识别的格式
#     image = tf.cast(image,tf.string)
#     label = tf.cast(label, tf.int32)

#     # 将image 和 label 放倒队列里 
#     input_queue = tf.train.slice_input_producer([image,label])
#     label = input_queue[1]
#     # 读取图片的全部信息
#     image_contents = tf.read_file(input_queue[0])
#     # 把图片解码，channels ＝3 为彩色图片, r，g ，b  黑白图片为 1 ，也可以理解为图片的厚度
#     image = tf.image.decode_jpeg(image_contents,channels =3)
#     # 将图片以图片中心进行裁剪或者扩充为 指定的image_W，image_H
#     image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
#     # 对数据进行标准化,标准化，就是减去它的均值，除以他的方差
#     image = tf.image.per_image_standardization(image)

#     # 生成批次  num_threads 有多少个线程根据电脑配置设置  capacity 队列中 最多容纳图片的个数  tf.train.shuffle_batch 打乱顺序，
#     image_batch, label_batch = tf.train.batch([image, label],batch_size = batch_size, num_threads = 64, capacity = capacity)
    
#     # 重新定义下 label_batch 的形状
#     label_batch = tf.reshape(label_batch , [batch_size])
#     # 转化图片
#     image_batch = tf.cast(image_batch,tf.float32)
#     return  image_batch, label_batch


def rename(path):
    cnt = 0
    imgs = os.listdir(path)
    label_name = path.split('/')[-2]
    path_ = path[0:len(path)-len(label_name)-1]
    if os.path.exists(path_ + 'imgs') == False:
        os.mkdir(path_ + 'imgs')
    new_path = path_ + 'imgs/' 
    for img in imgs:
        temp=cv2.imread(path + img)
        # os.remove(path + img)
        cv2.imwrite(new_path +  label_name + str(cnt)+".jpg",temp)
        print "renamed "+ img +" to "+label_name + str(cnt)+".jpg"
        cnt += 1
    return new_path

if __name__=='__main__':
    path = os.getcwd() + '/'
    print path

    cat_path = path + 'cat/'
    dot_path = path + 'dog/'
    datasets_path = rename(cat_path)
    datasets_path = rename(dot_path)
    imgs = os.listdir(datasets_path)


#     cats = []
#     label_cats = []
#     dogs = []
#     label_dogs = []
#     for img in imgs:
#             print 'img',img
#             name = img.split('.')
#             print 'name',name
#             if 'cat' in name[0]:
#                 cats.append(path + img)
#                 label_cats.append(0)
#             else:
#                 if 'dog' in name[0]:
#                     dogs.append(path + img)
#                     label_dogs.append(1)           
#             # print 'cats',cats
#             # print 'dogs',dogs
            
#             image_list = np.hstack((cats,dogs))
#             label_list = np.hstack((label_cats,label_dogs))

#     print 'label_cats',label_cats,len(label_cats)
#     print 'label_dogs',label_dogs,len(label_dogs)
#     print 'image_list',image_list,len(image_list)
#     print 'label_list',label_list,len(label_list)
#     print '===================='
#     train = tf.cast(image_list,tf.string)
#     train_label = tf.cast(label_list,tf.string)
#     print 'image',train
#     print 'label',train_label
#     train_batch, train_label_batch = get_batch(train,  
#                                                       train_label,  
#                                                       IMG_W,  
#                                                       IMG_H,  
#                                                       BATCH_SIZE,   
#                                                       CAPACITY)

#     print 'train_batch', train_batch
#     print 'train_label_batch', train_label_batch
# # return image_list,label_list