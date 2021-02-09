import pandas as pd
import numpy as np
import io

label_dir = './orig_data/TALE_label.txt'
feature_dir = './orig_data/TALE_feature_merge.txt'
adj_dir = './orig_data/TALE_repeat_distance.txt'
output_feature_dir = './fea.tsv'
output_adj_dir = './adj.tsv'
tale_set = set()
save_fea_list = []
with io.open(label_dir,'r',encoding='utf-8') as fi:
    for i,line1 in enumerate(fi):
        line = line1.strip().split('\t')
        repeat = line[0]
        tale_set.add(repeat)
print("num of tale_set is: ", len(tale_set))  

with io.open(feature_dir,'r',encoding='utf-8') as fi, io.open(output_feature_dir, 'w', encoding='utf-8') as fo:
    for i,line1 in enumerate(fi):
        line = line1.strip().split('\t')
        repeat = line[0]
        if repeat in tale_set:
            fo.write(line1)
with io.open(adj_dir,'r',encoding='utf-8') as fi, io.open(output_adj_dir, 'w', encoding='utf-8') as fo:
    for i,line1 in enumerate(fi):
        line = line1.strip().split(' ')
        if i==0:
            num = len(line)
            for j in range(num):
                repeat = line[j].strip().split('"')[1]
                if repeat in tale_set:
                    fo.write('\t'+repeat)
                    save_fea_list.append(j)
            fo.write(u'\n')
        else:
            repeat = line[0].strip().split('"')[1]
            if repeat in tale_set:
                fo.write(repeat)
                for j in save_fea_list:
                    fo.write('\t'+line[j+1])
                fo.write(u'\n')
print("Finised!")
