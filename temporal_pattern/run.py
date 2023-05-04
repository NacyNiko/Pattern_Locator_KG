# -*- coding: utf-8 -*- 
# @Time : 2023/2/7 14:27 
# @Author : Yinan 
# @File : run.py

import argparse

import evaluation
import temporal_pattern_lookout
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='icews14', choices=['icews14', 'wikidata', 'icews05-15', 'yago15k'], help='Knowledge graph dataset')

parser.add_argument('--threshold', default=0.5, type=float)
args = parser.parse_args()

temporal_pattern_lookout.main(args.dataset)

for p in ['temporal implication', 'evolve', 'temporal inverse', 'implication'
    , 'symmetric', 'temporal symmetric', 'inverse']:
    evaluation.main(args.dataset, p, args.threshold)




