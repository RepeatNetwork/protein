# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import io

content_dir = './zf_data/zf.content'
new_content_dir = './zf_data/zf.content.norm'
num = 0
id_dict = {}
label_dict = {}
threshold = 0.2
### 统计feature数量
with io.open(content_dir,'r',encoding='utf-8') as fi:
    for i,line1 in enumerate(fi):
        line = line1.strip().split('\t')
        if i==0:
            num = len(line) - (1+12)
            print("feature cols's num is: ", num)
            break
fea_min = [0 for i in range(num)]
fea_max = [1 for i in range(num)]
### 统计每列特征的最大值，最小值
with io.open(content_dir,'r',encoding='utf-8') as fi:
    for i,line1 in enumerate(fi):
        line = line1.strip().split('\t')
        for j in range(num):
            cur_fea = float(line[j+1])
            if cur_fea < fea_min[j]:
                fea_min[j] = cur_fea
            if cur_fea > fea_max[j]:
                fea_max[j] = cur_fea
### 计算归一化特征
with io.open(content_dir,'r',encoding='utf-8') as fi, io.open(new_content_dir,'w',encoding='utf-8') as fo:
    for i,line1 in enumerate(fi):
        line = line1.strip().split('\t')
        fo.write(line[0])
        for j in range(num):
            old_fea = float(line[j+1])
            new_fea = (old_fea - fea_min[j]) / (fea_max[j] - fea_min[j])
            fo.write(u'\t'+str(new_fea))
        for j in range(-12, 0):
            fo.write(u'\t'+line[j])
        fo.write(u'\n')
print("Finished!")
