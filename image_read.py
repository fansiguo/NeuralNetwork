#!/usr/bin/env python3
#===============================================================================
#
#         FILE: image_read.py
#
#        USAGE: ./image_read.py
#
#  DESCRIPTION: 
#
#      OPTIONS: ---
# REQUIREMENTS: ---
#         BUGS: ---
#        NOTES: ---
#       AUTHOR: fansiguo
#      COMPANY: jd.com
#      VERSION: 1.0
#      CREATED: 2018/5/3 16:49
#     REVIEWER: 
#     REVISION: ---
#    SRC_TABLE: 
#         
#    TGT_TABLE: 
#===============================================================================
import scipy.misc

img_array = scipy.misc.imread('C:\\Users\\fansiguo\\Desktop\\data\\img\\0.jpeg',flatten=True)
img_data =  255.0 - img_array.reshape(784)
