# -*- coding: utf-8 -*- 
# @Time : 2023/1/24 19:24 
# @Author : Yinan 
# @File : utilities.py
import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticke
import numpy as np
from itertools import combinations, permutations

# a) in training set: freq of symmetric grounding per relation
# b) total freq of samples for each relations
# c) compute percentage
# d) distubution of training and teat set
# e) put a threshold to select symmetric etc
# f) for static symmetric: count how many samples by that relation exist in test
# another: for dynamic symmetric: select top k dynamic symmetric relations based on trainig you already got,
# then count how many of test samples their Premise exist in train  ,premise->conclusion


class AnalysisTools:
    def __init__(self, dataset, pattern):
        self.dataset = dataset
        self.pattern = pattern

        if self.pattern == 'evolve' or self.pattern.split()[0] == 'temporal':
            self.temporal = True
        else:
            self.temporal = False

        self.save_path = '../results/{}/statistics/'.format(self.dataset)
        self.static_rel = pd.read_csv('../results/{}/pattern sets/train/stat_t_rel.csv'.format(self.dataset))
        self.temporal_rel = pd.read_csv('../results/{}/pattern sets/train/stat_t_rel.csv'.format(self.dataset))
        if not os.path.exists('../results/{}/summary/{}'.format(self.dataset, self.pattern)):
            os.makedirs('../results/{}/summary/{}'.format(self.dataset, self.pattern))

    """ occurrence of certain pattern for each relation in train set
        INPUT: target_pattern: pattern set,  a subset of original training set and obey this pattern.
                stat_t_rel: name of relations and their number.
                pattern_name: on which pattern working on. 
        OUTPUT: Frequency of each relation for certain pattern in training set, sorted by descending.
    """
    @ staticmethod
    def occurrence_pattern(target_pattern: pd.DataFrame, stat_t_rel: pd.DataFrame, pattern_name: str):
        counts_num = stat_t_rel.copy().set_index('relation')
        counts_num.sort_index(inplace=True)
        pattern_set = pd.DataFrame(target_pattern.value_counts('relation')).rename(columns={0: 'number'})
        stat_t_rel = stat_t_rel.set_index('relation', inplace=False)
        stat = (pattern_set / stat_t_rel).fillna(0).rename(columns={'number': 'percentage'})
        stat.insert(0, 'number of relation', counts_num.loc[:, 'number'])
        stat.insert(1, 'number of {}'.format(pattern_name), pattern_set.loc[:, 'number'])
        stat = stat.fillna(0)
        stat.reset_index(inplace=True)
        stat['number of {}'.format(pattern_name)] = stat['number of {}'.format(pattern_name)].astype(int)
        return stat.sort_values(by='number of relation', ascending=False)

    def pattern_frequency_analyse(self):
        for by in ['train', 'test']:
            save_path = self.save_path + by
            path_list = {'s_%s' % self.pattern: save_path + '/static'
                , 's_rel': save_path + '/static'
                , 'd_%s' % self.pattern: save_path + '/dynamic'
                , 'd_rel': save_path + '/dynamic'
                              }
            for p in path_list.values():
                if not os.path.exists(p):
                    os.makedirs(p)

            if self.temporal:
                set_t_pattern = pd.read_csv(r'../results/{}/pattern sets/{}/set {}.csv'
                                            .format(self.dataset, by, self.pattern))
                stat_t_rel = pd.read_csv(r'../results/{}/pattern sets/{}/stat_t_rel.csv'
                                         .format(self.dataset, by))
                stat_t_pattern = self.occurrence_pattern(set_t_pattern, stat_t_rel, '%s' % self.pattern)

                stat_t_pattern.reset_index(drop=True).to_csv(
                    '{}/freq_{}.csv'.format(path_list['d_%s' % self.pattern], self.pattern),
                    index=False)


            else:
                set_pattern = pd.read_csv(r'../results/{}/pattern sets/{}/set {}.csv'
                                            .format(self.dataset, by, self.pattern))
                stat_t_rel = pd.read_csv(r'../results/{}/pattern sets/{}/stat_t_rel.csv'
                                       .format(self.dataset, by))
                stat_pattern = self.occurrence_pattern(set_pattern, stat_t_rel, '%s' % self.pattern)

                stat_pattern.reset_index(drop=True).to_csv('{}/freq_{}.csv'.format(path_list['s_%s' % self.pattern], self.pattern),
                                                       index=False)


    def pattern_dictionary(self):
        def cal_comb(relations: pd.Series) -> list:
            return list(combinations([i for i in relations], 2))

        def pattern_pair(pattern_set: pd.DataFrame) -> pd.DataFrame:
            relations_stat = self.temporal_rel if self.temporal else self.static_rel
            dic = pd.DataFrame(columns=['relation i', 'relation j', '#%s' % self.pattern, '#rel_i'
                , 'P one direction', 'P two direction'])
            check = set()
            rel_list = cal_comb(pattern_set.loc[:, 'relation'].drop_duplicates().reset_index(drop=True))
            if self.pattern in ['symmetric', 'temporal symmetric']:
                for rel in relations_stat['relation']:
                    if rel not in check:
                        check.add(rel)
                        rel1 = pattern_set.copy()[pattern_set['relation'] == rel]
                        num_rel1 = int(relations_stat[relations_stat['relation'] == rel]['number'].values)
                        temp = pd.DataFrame([['{}'.format(rel), '{}'.format(rel), rel1.shape[0], num_rel1
                                                 , rel1.shape[0]/num_rel1, rel1.shape[0]/num_rel1]]
                                , columns=['relation i', 'relation j', '#%s' % self.pattern, '#rel_i', 'P one direction', 'P two direction'])
                        dic = pd.concat([dic, temp], axis=0)
            else:
                for rel in rel_list:
                    if rel not in check and rel[-1::-1] not in check:
                        check.add(rel)
                        check.add(rel[-1::1])
                        rel1 = pattern_set.copy()[pattern_set['relation'] == rel[0]]
                        rel2 = pattern_set.copy()[pattern_set['relation'] == rel[1]]
                        if self.pattern in ['inverse', 'temporal inverse']:
                            rel2.loc[:, 'head'], rel2.loc[:, 'tail'] = rel2.loc[:, 'tail'].copy(), rel2.loc[:, 'head'].copy()

                        if self.pattern == 'evolve':
                            ans = pd.merge(rel1, rel2, how='inner', on=['head', 'tail'])
                            ans1 = ans[ans['time_x'] <= ans['time_y']].drop_duplicates(subset=['time_x', 'head', 'tail'])
                            ans2 = ans[ans['time_x'] >= ans['time_y']].drop_duplicates(subset=['time_y', 'head', 'tail'])
                            p_rel1 = ans1.shape[0]
                            p_rel2 = ans2.shape[0]
                        elif self.pattern in ['implication', 'temporal implication', 'inverse', 'temporal inverse']:
                            if self.temporal:
                                ans = pd.merge(rel1, rel2, how='inner',
                                       on=['head', 'tail', 'time'])
                                p_rel1 = p_rel2 = ans.shape[0]
                            else:
                                ans = pd.merge(rel1, rel2, how='inner',
                                               on=['head', 'tail'])
                                ans1 = ans.drop_duplicates(subset=['time_x', 'head', 'tail'], inplace=False)
                                ans2 = ans.drop_duplicates(subset=['time_y', 'head', 'tail'], inplace=False)
                                p_rel1 = ans1.shape[0]
                                p_rel2 = ans2.shape[0]
                        else:
                            raise 'Undefined pattern!'
                        num_rel1 = int(relations_stat[relations_stat['relation'] == rel[0]]['number'].values)
                        num_rel2 = int(relations_stat[relations_stat['relation'] == rel[1]]['number'].values)
                        temp1 = pd.DataFrame([[rel[0], rel[1], p_rel1, num_rel1, p_rel1 / num_rel1, (p_rel1+p_rel2)/(num_rel1+num_rel2)]]
                                             , columns=['relation i', 'relation j', '#%s' % self.pattern, '#rel_i',
                                                        'P one direction', 'P two direction'])
                        temp2 = pd.DataFrame([[rel[1], rel[0], p_rel2, num_rel2, p_rel2 / num_rel2, (p_rel1+p_rel2)/(num_rel1+num_rel2)]]
                                             , columns=['relation i', 'relation j', '#%s' % self.pattern, '#rel_i',
                                                        'P one direction', 'P two direction'])
                        dic = pd.concat([dic, temp1, temp2], axis=0)
            return dic

        def transformation(df: pd.DataFrame):
            relations = df.drop_duplicates(subset=['relation i']).sort_values(by='#rel_i', ascending=False).loc[:, 'relation i']
            temp_res = pd.DataFrame(index=relations, columns=relations)
            for index, row in df.iterrows():
                temp_res.loc[row['relation i'], row['relation j']] = row['P one direction']
            temp_res.fillna(0, inplace=True)
            return temp_res

        save_path = self.save_path + 'train'
        path_list = {'s_%s' % self.pattern: save_path + '/static'
            , 's_rel': save_path + '/static'
            , 'd_%s' % self.pattern: save_path + '/dynamic'
            , 'd_rel': save_path + '/dynamic'
                              }
        for p in path_list.values():
            if not os.path.exists('{}/pair/'.format(p)):
                os.makedirs('{}/pair/'.format(p))
        set_pattern = pd.read_csv(r'../results/{}/pattern sets/train/set {}.csv'
                                  .format(self.dataset, self.pattern))
        res = pattern_pair(set_pattern)
        res_2d = transformation(res)

        if self.temporal:
            res.to_csv('{}/pair/pair_{}.csv'.format(path_list['d_%s' % self.pattern], self.pattern),
                    index=False)
            res_2d.to_csv('{}/pair/pair2d_{}.csv'.format(path_list['d_%s' % self.pattern], self.pattern),
                    index=True)
        else:
            res.to_csv('{}/pair/pair_{}.csv'.format(path_list['s_%s' % self.pattern], self.pattern),
                           index=False)
            res_2d.to_csv('{}/pair/pair2d_{}.csv'.format(path_list['s_%s' % self.pattern], self.pattern),
                          index=True)

        res.to_csv('../results/{}/summary/{}/pair_{}.csv'.format(self.dataset, self.pattern, self.pattern),
                   index=False)
        res_2d.to_csv('../results/{}/summary/{}/pair2d_{}.csv'.format(self.dataset, self.pattern, self.pattern),
                      index=True)

    def conclusion_premise_paar(self, threshold):
        save_path = '../results/{}/statistics/con_pre_pair'.format(self.dataset)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        test_set = pd.read_table('./data/{}/test2id.txt'.format(self.dataset),
                                 header=None, names=['head', 'relation', 'tail', 'time'], index_col=False)
        train_set = pd.read_table('./data/{}/train2id.txt'.format(self.dataset),
                                  header=None, names=['head', 'relation', 'tail', 'time'], index_col=False)
        test_reversed = test_set.copy()
        test_reversed.loc[:, 'head'], test_reversed.loc[:, 'tail'] = \
            test_reversed.loc[:, 'tail'].copy(), test_reversed.loc[:, 'head'].copy()
        test_vc = test_set.value_counts('relation')
        subset = pd.DataFrame()
        if self.pattern in ['symmetric', 'temporal symmetric']:
            relations = pd.read_csv('../results/{}/statistics/train/{}/freq_{}.csv'.format(
                self.dataset, 'static' if self.pattern == 'symmetric' else 'dynamic', self.pattern
            ))
            relations = relations[relations['percentage'] >= threshold]
            temp = set(relations.loc[:, 'relation'])
            test_reversed = test_reversed.loc[test_reversed['relation'].isin(temp)]
            intersected = pd.merge(train_set, test_reversed, how='inner'
                                   , on=['head', 'relation', 'tail'] if self.pattern == 'symmetric' else ['head', 'relation', 'tail', 'time'])
            subset = pd.concat([subset, intersected.iloc[:, :4].drop_duplicates()])
            subset.rename(columns={'time_x': 'time'}, inplace=True)
            if self.pattern == 'symmetric':
                intersected.drop_duplicates(subset=['time_x'], inplace=True)

            int_vc = intersected.value_counts('relation')

            stat = pd.DataFrame((int_vc / test_vc))

            stat.insert(0, 'number in test set', test_vc)
            stat.insert(0, 'number of {} in train set'.format(self.pattern), int_vc)
            stat.insert(0, 'frequency', relations.set_index('relation').loc[:, 'percentage'])
            stat.fillna(0, inplace=True)
            stat.rename(columns={0: 'percentage'}, inplace=True)
            stat['number of {} in train set'.format(self.pattern)] = stat[
                'number of {} in train set'.format(self.pattern)].astype(int)
            stat.sort_values(by='number in test set', inplace=True, ascending=False)

        else:
            stat = pd.DataFrame(
                columns=['relation test', 'relation train', '#%s' % self.pattern, '#rel_test', 'percentage'])
            if self.pattern in ['temporal implication', 'temporal inverse']:
                relations = pd.read_csv('../results/{}/statistics/train/dynamic/pair/pair_{}.csv'.format(
                    self.dataset, self.pattern))
                relations = relations[relations['P one direction'] >= threshold]
                if self.pattern == 'temporal inverse':
                    temp = relations.loc[:, 'relation j']
                    test_reversed = test_reversed.loc[test_reversed['relation'].isin(temp)]
                    intersected = pd.merge(train_set, test_reversed, how='inner',
                                           on=['head', 'tail', 'time'])

                    for index, row in relations.iterrows():
                        if row['relation j'] in set(test_reversed.loc[:, 'relation']):
                            rel_i = row['relation i']
                            rel_j = row['relation j']
                            intersected_temp = intersected[(intersected['relation_x'] == rel_i)
                                                           & (intersected['relation_y'] == rel_j)]
                            temp = pd.DataFrame([[rel_j, rel_i, intersected_temp.shape[0], test_vc[rel_j]
                                                    , intersected_temp.shape[0] / test_vc[rel_j]]]
                                                 , columns=['relation test', 'relation train', '#%s' % self.pattern,
                                                            '#rel_test', 'percentage'])
                            subset = pd.concat([subset, intersected_temp.rename({'relation_x': 'relation'}
                                                                                , inplace=True).iloc[:, :4].drop_duplicates()])
                            stat = pd.concat([stat, temp], axis=0)
                else:
                    temp = relations.loc[:, 'relation j']
                    test_set = test_set.loc[test_set['relation'].isin(temp)]
                    intersected = pd.merge(train_set, test_set, how='inner',
                                           on=['head', 'tail', 'time'])
                    for index, row in relations.iterrows():
                        if row['relation j'] in set(test_reversed.loc[:, 'relation']):
                            rel_i = row['relation i']
                            rel_j = row['relation j']
                            intersected_temp = intersected[(intersected['relation_x'] == rel_i)
                                                       & (intersected['relation_y'] == rel_j)]
                            temp = pd.DataFrame([[rel_i, rel_j, intersected_temp.shape[0], test_vc[rel_j]
                                                     , intersected_temp.shape[0] / test_vc[rel_j]]]
                                                , columns=['relation test', 'relation train', '#%s' % self.pattern,
                                                           '#rel_test', 'percentage'])
                            subset = pd.concat([subset, intersected_temp.iloc[:, :4].drop_duplicates()])
                            stat = pd.concat([stat, temp], axis=0)

            else:
                if self.pattern == 'evolve':
                    relations = pd.read_csv('../results/{}/statistics/train/dynamic/pair/pair_{}.csv'.format(
                        self.dataset, self.pattern))
                else:
                    relations = pd.read_csv('../results/{}/statistics/train/static/pair/pair_{}.csv'.format(
                        self.dataset, self.pattern))
                relations = relations[relations['P one direction'] >= threshold]
                if self.pattern == 'inverse':
                    temp = relations.loc[:, 'relation j']
                    test_reversed = test_reversed.loc[test_reversed['relation'].isin(temp)]
                    intersected = pd.merge(train_set, test_reversed, how='inner', on=['head', 'tail'])
                    for index, row in relations.iterrows():
                        if row['relation j'] in set(test_reversed.loc[:, 'relation']):
                            rel_i = row['relation i']
                            rel_j = row['relation j']
                            intersected_temp = intersected[(intersected['relation_x'] == rel_i)
                                                           & (intersected['relation_y'] == rel_j)]
                            temp = pd.DataFrame([[rel_j, rel_i, intersected_temp.drop_duplicates(subset=['head', 'tail', 'relation_y', 'time_y']).shape[0], test_vc[rel_j]
                                                     , intersected_temp.drop_duplicates(subset=['head', 'tail', 'relation_y', 'time_y']).shape[0] / test_vc[rel_j]]]
                                                , columns=['relation test', 'relation train', '#%s' % self.pattern,
                                                           '#rel_test', 'percentage'])
                            stat = pd.concat([stat, temp], axis=0)
                            subset = pd.concat([subset, intersected_temp.iloc[:, :4].drop_duplicates()])
                else:
                    temp = relations.loc[:, 'relation j']
                    test_set = test_set.loc[test_set['relation'].isin(temp)]
                    intersected = pd.merge(train_set, test_set, how='inner', on=['head', 'tail'])
                    for index, row in relations.iterrows():
                        if row['relation j'] in set(test_reversed.loc[:, 'relation']):
                            rel_i = row['relation i']
                            rel_j = row['relation j']
                            if self.pattern == 'evolve':
                                intersected_temp = intersected[(intersected['relation_x'] == rel_i)
                                                           & (intersected['relation_y'] == rel_j)
                                                           & (intersected['time_y'] >= intersected['time_x'])]
                            else:
                                intersected_temp = intersected[(intersected['relation_x'] == rel_i)
                                                           & (intersected['relation_y'] == rel_j)]
                            temp = pd.DataFrame([[rel_j, rel_i, intersected_temp.shape[0], test_vc[rel_j]
                                                     , intersected_temp.shape[0] / test_vc[rel_j]]]
                                                , columns=['relation test', 'relation train', '#%s' % self.pattern,
                                                           '#rel_test', 'percentage'])
                            stat = pd.concat([stat, temp], axis=0)
                            subset = pd.concat([subset, intersected_temp.iloc[:, :4].drop_duplicates()])

        stat.to_csv(save_path + '/{}.csv'.format(self.pattern), index=False)
        subset.to_csv(save_path + '/{}.txt'.format(self.pattern), sep='\t', index=False)

        stat.to_csv('../results/{}/summary/{}/{}.csv'.format(self.dataset, self.pattern, self.pattern), index=False)
        subset.to_csv('../results/{}/summary/{}/{}.txt'.format(self.dataset, self.pattern, self.pattern), sep='\t', index=False)

    def find_missing_instances(self, length=3):
        set_path = '../results/{}/pattern sets/train'.format(self.dataset)
        test_set = pd.read_table('./data/{}/test2id.txt'.format(self.dataset)
                                 , header=None, names=['head', 'relation', 'tail', 'time'], index_col=False)
        with open(set_path + '/set_entity_relation.pkl', 'rb') as f:
            entity_relation = pickle.load(f)
        selected_keys = [key for key, value in entity_relation.items() if len(value) >= length]
        col1, col2 = zip(*selected_keys)
        data_dict = {'head': col1, 'relation': col2}

        head_relation = pd.DataFrame(data_dict)
        selected_instances = pd.merge(test_set, head_relation, how='inner', on=['head', 'relation'])
        selected_instances.to_csv('../results/{}/summary/missing_instances.txt'.format(self.dataset), sep='\t', index=False)

    # def temporal_symmetric_finding(self, threshold_num, threshold_percentage):
    #     if not os.path.exists('../results/{}/summary/temporal_symmetric_instances_selected.txt'.format(self.dataset)):
    #         freq_table = pd.read_csv(r'../results/{}/statistics/train/dynamic/freq_temporal symmetric.csv'.format(self.dataset))
    #         selected_rel = freq_table[(freq_table['number of temporal symmetric'] >= threshold_num) & (freq_table['percentage'] >= threshold_percentage)]
    #         test_set = pd.read_table(r'./data/{}/test2id.txt'.format(self.dataset), header=None
    #                                  , names=['head', 'relation', 'tail', 'time'], index_col=False)
    #         train_set = pd.read_table(r'./data/{}/train2id.txt'.format(self.dataset), header=None
    #                                  , names=['head', 'relation', 'tail', 'time'], index_col=False)
    #
    #         instance_in_test = test_set[test_set['relation'].isin(selected_rel['relation'])]
    #         inverse_instance_in_test = instance_in_test.copy()
    #         inverse_instance_in_test['head'], inverse_instance_in_test['tail'] = inverse_instance_in_test['tail'].copy()\
    #             , inverse_instance_in_test['head'].copy()
    #
    #         overlapping_train_set = pd.merge(train_set, inverse_instance_in_test, how='inner')
    #         overlapping_train_set.to_csv('../results/{}/summary/temporal_symmetric_instances_selected.txt'.format(self.dataset), sep='\t', index=False)

    def pattern_finding(self, pattern, threshold_num, threshold_percentage):
        temporal = True if (pattern[:8] == 'temporal' or pattern == 'evolve') else False
        if not os.path.exists(f'../results/{self.dataset}/summary/{pattern}_instances_selected.txt'):
            freq_table = pd.read_csv(f'../results/{self.dataset}/statistics/train/{"dynamic" if temporal else "static"}/freq_{pattern}.csv')
            selected_rel = freq_table[(freq_table[f'number of {pattern}'] >= threshold_num) & (freq_table['percentage'] >= threshold_percentage)]
            test_set = pd.read_table(r'./data/{}/test2id.txt'.format(self.dataset), header=None
                                     , names=['head', 'relation', 'tail', 'time'], index_col=False)
            train_set = pd.read_table(r'./data/{}/train2id.txt'.format(self.dataset), header=None
                                     , names=['head', 'relation', 'tail', 'time'], index_col=False)

            instance_in_test = test_set[test_set['relation'].isin(selected_rel['relation'])]
            inverse_instance_in_test = instance_in_test.copy()
            inverse_instance_in_test['head'], inverse_instance_in_test['tail'] = inverse_instance_in_test['tail'].copy()\
                    , inverse_instance_in_test['head'].copy()

            if pattern in ['symmetric', 'temporal symmetric']:
                overlapping_train_set = pd.merge(inverse_instance_in_test, train_set, how='inner'
                            , on=['head', 'relation', 'tail', 'time'] if temporal else ['head', 'relation', 'tail'])
                if not temporal:
                    overlapping_train_set = overlapping_train_set.iloc[:, :4].rename(columns={'time_x':  'time'}).drop_duplicates()

            elif pattern in ['inverse', 'temporal inverse', 'implication', 'temporal implication', 'evolve']:
                if pattern in ['inverse', 'temporal inverse']:
                    overlapping_train_set = pd.merge(inverse_instance_in_test, train_set, how='inner',
                                                 on=['head', 'tail', 'time'] if temporal else ['head', 'tail'])
                elif pattern in ['implication', 'temporal implication', 'evolve']:
                    overlapping_train_set = pd.merge(instance_in_test, train_set, how='inner',
                                                 on=['head', 'tail', 'time'] if temporal else ['head', 'tail'])
                if not temporal:
                    overlapping_train_set.drop(columns='time_y', inplace=True)
                    overlapping_train_set = overlapping_train_set.rename(columns={'time_x':  'time'}).drop_duplicates()
                overlapping_train_set = overlapping_train_set[overlapping_train_set['relation_x'] != overlapping_train_set['relation_y']]
                overlapping_train_set.drop(columns='relation_y', inplace=True)
                overlapping_train_set = overlapping_train_set.rename(columns={'relation_x':  'relation'}).drop_duplicates()

            if pattern in ['symmetric', 'temporal symmetric', 'inverse', 'temporal inverse']:
                overlapping_train_set['head'], overlapping_train_set['tail'] = overlapping_train_set['tail'].copy() \
                , overlapping_train_set['head'].copy()
            temp = pd.merge(overlapping_train_set, test_set, how='inner')
            overlapping_train_set.to_csv(f'../results/{self.dataset}/summary/{pattern}_instances_selected.txt',
                                         sep='\t', index=False)


class PlotTools:
    def __init__(self, dataset, pattern):
        self.dataset = dataset
        self.pattern = pattern

    @staticmethod
    def plot_distribution_rel(train_set, test_set, showall=False, dynamic=False, on='symmetric', save_path=None):
        if not showall:
            train_set = train_set[~train_set['freq'].isin([0])]
            test_set = test_set[~test_set['freq'].isin([0])]
        train_set.reset_index(inplace=True)
        test_set.reset_index(inplace=True)
        fig, ax = plt.subplots(1, 1, figsize=(24, 6))
        if on == 'relations':
            ax.bar(train_set.loc[:, 'relation'].apply(str), train_set.loc[:, 'freq'] / train_set.loc[:, 'freq'].sum(),
                   label='Train', color='#f9766e', edgecolor='grey', alpha=0.5)
            ax.bar(test_set.loc[:, 'relation'].apply(str), test_set.loc[:, 'freq'] / test_set.loc[:, 'freq'].sum(),
                   label='Test', color='#00bfc4', edgecolor='grey', alpha=0.5)
        elif on == 'symmetric':
            ax.bar(train_set.loc[:, 'relation'].apply(str), train_set.loc[:, 'freq'],
                   label='Train', color='#f9766e', edgecolor='grey', alpha=0.5)
            ax.bar(test_set.loc[:, 'relation'].apply(str), test_set.loc[:, 'freq'],
                   label='Test', color='#00bfc4', edgecolor='grey', alpha=0.5)
        ax.set_xlabel('relations', fontsize=12)
        ax.set_ylabel('percentage', fontsize=12)
        ax.tick_params(axis='x', length=0, rotation=30)
        ax.grid(axis='y', alpha=0.5, ls='--')
        ax.legend(frameon=False)
        ax.set_title('{} {}'.format('Dynamic' if dynamic else 'Static', on))

        if on == 'relations':
            ax.xaxis.set_major_locator(ticke.MultipleLocator(base=5))

        if save_path:
            plt.savefig(save_path + 'Distribution.png', dpi=300)
        # plt.show()

    def plot_train_test_distribution(self):
        train_set = pd.read_csv('../results/%s/pattern sets/train/stat_t_rel.csv' % self.dataset)
        test_set = pd.read_csv('../results/%s/pattern sets/test/stat_t_rel.csv' % self.dataset)
        train_t_set = pd.read_csv('../results/%s/pattern sets/train/stat_t_rel.csv' % self.dataset)
        test_t_set = pd.read_csv('../results/%s/pattern sets/test/stat_t_rel.csv' % self.dataset)

        for s in ['temporal', '']:
            data_train = train_t_set if s == 'temporal' else train_set
            data_test = test_t_set if s == 'temporal' else test_set
            rel_num_tr = data_train.loc[:, 'number'].sum()
            rel_num_te = data_test.loc[:, 'number'].sum()

            data_train.set_index('relation', inplace=True)
            data_test.set_index('relation', inplace=True)
            fig, ax = plt.subplots(2, 1, figsize=(24, 6), sharex=True)
            ax[0].bar(data_train.index, data_train.loc[:, 'number'],
                      label='number of %s relation in train set' % s, color='#f9766e', edgecolor='grey', alpha=0.5)
            ax[0].bar(data_test.index, data_test.loc[:, 'number'],
                      label='number of %s in test set' % s, color='#00bfc4', edgecolor='grey', alpha=0.5)

            ax[1].bar(data_train.index, data_train.loc[:, 'number'] / rel_num_tr,
                      label='percentage of %s relation in train set' % s, color='#f9766e', edgecolor='grey', alpha=0.5)
            ax[1].bar(data_test.index, data_test.loc[:, 'number'] / rel_num_te,
                      label='percentage of %s in test set' % s, color='#00bfc4', edgecolor='grey', alpha=0.5)

            ax[0].set_xlabel('relations', fontsize=12)
            ax[1].set_xlabel('relations', fontsize=12)
            ax[0].set_ylabel('number', fontsize=12)
            ax[1].set_ylabel('percentage', fontsize=12)
            for i in range(2):
                ax[i].tick_params(axis='x', length=0, rotation=30)
                ax[i].grid(axis='y', alpha=0.5, ls='--')
                ax[i].legend(frameon=False)

            fig.suptitle('({}) {} relation distribution'.format(self.dataset, s))
            plt.xticks([])

            save_path = '../results/{}/statistics/'.format(self.dataset)
            plt.savefig(save_path + '%s_Distribution.png' % s, dpi=300)
            plt.savefig('../results/{}/summary/{}/{}_Distribution.png'.format(self.dataset, self.pattern, self.pattern), dpi=300)
            # plt.show()

    # def plot_pair(self):
    #     load_path = '../results/{}/statistics/train/{}/'.format(self.dataset
    #                                                             , 'dynamic' if (self.pattern == 'evolve' or self.pattern.split[0] == 'temporal') else 'static'
    #     load_path += 'pair_%s.csv' % self.pattern


class FolderChange:
    def __init__(self, dataset, pattern):
        self.pattern = pattern
        self.dataset = dataset



