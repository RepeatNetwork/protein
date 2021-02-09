# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import io

fea_dir = './fea.tsv'
adj_dir = './adj.tsv'
label_dir = './label.tsv'
cite_dir = './tale_data/tale.cites'        
##从序号1开始编码，1，2，...1776
content_dir = './tale_data/tale.content'
num = 0
id_dict = {}
label_dict = {}
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
            if float(line[index])<0.05:
                fo.write(str(i)+u' '+str(index)+u'\n')

with io.open(label_dir,'r',encoding='utf-8') as fi:
    for i,line1 in enumerate(fi):
        line = line1.strip().split('\t')
        repeat = line[0]
        label = line[1] 
        ## A1, C2, G3, T4
        if label=='A':
            label_dict[repeat] = '1'
        elif label=='C':
            label_dict[repeat] = '2'
        elif label=='G':
            label_dict[repeat] = '3'
        elif label=='T':
            label_dict[repeat] = '4'
with io.open(fea_dir,'r',encoding='utf-8') as fi, io.open(content_dir,'wb') as fo:
    for i,line1 in enumerate(fi):
        line = line1.strip().split('\t')
        repeat = line[0]
        index = id_dict[repeat]
        # fo.write(repeat+'\t'+str(index))
        fo.write(str(index))
        for j in range(len(line)-1):  
            fo.write('\t'+line[j+1])
        fo.write(u'\t'+label_dict[repeat]+u'\n')
print("Finished!")
