#!/usr/bin/env python3
#===============================================================================
#
#         FILE: data_explore.py
#
#        USAGE: ./data_explore.py
#
#  DESCRIPTION: 根据用户配置规则，计算SKU维度的数据详情。
#
#      OPTIONS: ---
# REQUIREMENTS: ---
#         BUGS: ---
#        NOTES: ---
#       AUTHOR: fansiguo
#      COMPANY: jd.com
#      VERSION: 1.0
#      CREATED: 2018/5/3 12:09
#     REVIEWER: 
#     REVISION: ---
#    SRC_TABLE: 
#         
#    TGT_TABLE: 
#===============================================================================
import numpy
import matplotlib.pyplot as plt
import scipy.misc

data_file = open('C:\\Users\\fansiguo\\Desktop\\data\\mnist_train_100.csv','r')
data_list = data_file.readlines()
data_file.close()

all_values = data_list[1].split(',')
#image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
#plt.imshow(image_array,cmap='Greys',interpolation='None')
#plt.show()

scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
print(scaled_input)

img_array = scipy.misc.imread('C:\\Users\\fansiguo\\Desktop\\data\\img\\7.png',flatten=True)
img_data =  255.0 - img_array.reshape(784)
