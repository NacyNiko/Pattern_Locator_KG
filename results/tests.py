# -*- coding: utf-8 -*- 
# @Time : 2023/2/22 15:27 
# @Author : Yinan 
# @File : tests.py
import pandas as pd

df = pd.read_table('../results/icews14/statistics/con_pre_pair/evolve.txt', names=['head', 'relation', 'tail', 'time'])

df_ori = pd.read_table('../temporal_pattern/data/icews14/train2id.txt', names=['head', 'relation', 'tail', 'time'])
df_int = pd.merge(df_ori, df, how='inner')
pass