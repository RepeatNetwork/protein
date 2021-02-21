# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import io

fea_dir = './fea.tsv'
adj_dir = './adj.tsv'
label_dir = './label.tsv'
cite_dir = './zf_data/zf.cites'        
##从序号1开始编码，1，2，...1776
content_dir = './zf_data/zf.content'
num = 0
id_dict = {}
label_dict = {}
threshold = 0.4
effect_node = set()
with io.open(adj_dir,'r',encoding='utf-8') as fi, io.open(cite_dir,'w',encoding='utf-8') as fo:
    for i,line1 in enumerate(fi):
        line = line1.strip().split('\t')
        if i==0:
            num = len(line)
            for i in range(num):
                id_dict[line[i]] = i+1      
            continue
        repeat = line[0]
        for j in range(num):
            index = j+1
            if float(line[index]) < threshold and i != index:
                effect_node.add(i)
                effect_node.add(index)
                fo.write(str(i)+u' '+str(index)+u'\n')

with io.open(label_dir,'r',encoding='utf-8') as fi:
    for i,line1 in enumerate(fi):
        line = line1.strip().split('\t')
        repeat = line[0]
        label = line[1:]
        label_dict[repeat] = label
with io.open(fea_dir,'r',encoding='utf-8') as fi, io.open(content_dir,'wb') as fo:
    for i,line1 in enumerate(fi):
        line = line1.strip().split('\t')
        repeat = line[0]
        index = id_dict[repeat]
        ### 去除孤立结点 ###
        if index not in effect_node:
            continue
        fo.write(str(index))
        for j in range(len(line)-1):  
            fo.write('\t'+line[j+1])
        for j in range(len(label_dict[repeat])):
            fo.write(u'\t'+label_dict[repeat][j])
        fo.write(u'\n')
print("Finished!")
