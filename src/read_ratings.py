# -*- coding: utf-8 -*-
'''
Data pre process

@author:
Liu Yun

@ created:
07/8/2019
@references:
'''
import os
import pandas as pd
import dill as pickle
import numpy as np
import itertools
from collections import Counter

TPS_DIR = '../data/imdb/10core'
PRE_TPS_DIR = '../data/imdb'
np.random.seed(2020)


def get_entity_index(reviews):  # reviews: all the reviews of dataset
    review_list = []
    for i in range(len(reviews)):
        review_list.append(reviews[i].split('\t'))
    entity_counts = Counter(itertools.chain(*review_list))
    # Mapping from index to review entities
    # most_common entities sorted by desc
    entity_sorted = [x[0] for x in entity_counts.most_common()]
    entity_sorted = list(sorted(entity_sorted))
    # Mapping from review entities to index, start=1 because we pad the triple used 0.
    entity_index = {x: i for i, x in enumerate(entity_sorted, start=1)}

    return entity_index


def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index=True)
    count = playcount_groupbyid.size()
    return count


def numerize(tp, user2id, item2id):
    uid = list(map(lambda x: user2id[x], tp['user_id']))
    sid = list(map(lambda x: item2id[x], tp['item_id']))
    tp['user_id'] = uid
    tp['item_id'] = sid
    return tp


def compute_len(user_reviews, item_reviews):
    # the number of reviews of users
    review_num_u = np.array([len(x) for x in user_reviews.values()])
    x = np.sort(review_num_u)
    # only covering 80% numbers of reviews, the max number of reviews
    u_len = x[int(0.8 * len(review_num_u)) - 1]
    # the number of review entities of each review
    review_len_u = np.array([len(j) for i in user_reviews.values() for j in i])
    x2 = np.sort(review_len_u)
    # only covering 90% numbers of review entities, the max number of review entities
    u2_len = x2[int(0.9 * len(review_len_u)) - 1]

    review_num_i = np.array([len(x) for x in item_reviews.values()])
    y = np.sort(review_num_i)
    i_len = y[int(0.8 * len(review_num_i)) - 1]
    review_len_i = np.array([len(j) for i in item_reviews.values() for j in i])
    y2 = np.sort(review_len_i)
    i2_len = y2[int(0.9 * len(review_len_i)) - 1]
    print("u_len:", u_len)
    print("i_len:", i_len)
    print("u2_len:", u2_len)
    print("i2_len:", i2_len)
    user_num = len(user_reviews)
    item_num = len(item_reviews)
    print("user_num:", user_num)
    print("item_num:", item_num)

    return u_len, i_len, u2_len, i2_len


def pad_reviews(user_reviews, u_len, u2_len, padding_word=0):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    review_num = u_len  # the max number of reviews per user
    review_len = u2_len  # the max length of review entities of each review

    u_text2 = {}
    for i in user_reviews.keys():  # for the current user i
        u_reviews = user_reviews[i]  # the list of reviews of user i
        padded_u_train = []
        for ri in range(review_num):  # for the current review of user i
            if ri < len(u_reviews):  # the current review belongs to the user
                sentence = u_reviews[ri]
                if review_len > len(sentence):
                    num_padding = review_len - len(sentence)
                    new_sentence = sentence + [padding_word] * num_padding
                    padded_u_train.append(new_sentence)
                else:
                    new_sentence = sentence[:review_len]
                    padded_u_train.append(new_sentence)
            else:  # pad new review of user i
                new_sentence = [padding_word] * review_len
                padded_u_train.append(new_sentence)
        u_text2[i] = padded_u_train

    return u_text2


if __name__ == '__main__':
    data = pd.read_csv(os.path.join(TPS_DIR, 'imdb.csv'))
    usercount, itemcount = get_count(
        data, 'user_id'), get_count(data, 'item_id')
    unique_uid = usercount.index
    unique_sid = itemcount.index
    item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
    data = numerize(data, user2id, item2id)
    tp_rating = data[['user_id', 'item_id', 'ratings']]

    n_ratings = tp_rating.shape[0]
    test = np.random.choice(n_ratings, size=int(
        0.20 * n_ratings), replace=False)
    test_idx = np.zeros(n_ratings, dtype=bool)
    test_idx[test] = True

    tp_1 = tp_rating[test_idx]
    tp_train = tp_rating[~test_idx]

    data2 = data[test_idx]
    data = data[~test_idx]

    n_ratings = tp_1.shape[0]
    test = np.random.choice(n_ratings, size=int(
        0.50 * n_ratings), replace=False)

    test_idx = np.zeros(n_ratings, dtype=bool)
    test_idx[test] = True

    tp_test = tp_1[test_idx]
    tp_valid = tp_1[~test_idx]

    tp_train.to_csv(os.path.join(TPS_DIR, 'imdb_train.csv'),
                    index=False, header=None)
    tp_valid.to_csv(os.path.join(TPS_DIR, 'imdb_valid.csv'),
                    index=False, header=None)
    tp_test.to_csv(os.path.join(TPS_DIR, 'imdb_test.csv'),
                   index=False, header=None)
    # deal with reviews and replace triples in reviews as index
    entity_index = os.path.join(PRE_TPS_DIR, 'entity_index')
    with open(entity_index, "rb") as tf:
        entity_index = pickle.load(tf)
    # key: user_id, value: review indices
    user_reviews = {}
    item_reviews = {}
    # key: user_id, value: item_ids
    user_rid = {}
    item_rid = {}
    # train dataset
    for i in data.values:
        review_index = []
        reviews = i[3].split('\t')
        review_list = sorted(set(reviews), key=reviews.index)
        for entity in review_list:
            if entity in entity_index:
                review_index.append(entity_index[entity])

        if user_reviews.__contains__(int(i[0])):
            user_reviews[int(i[0])].append(review_index)
            user_rid[int(i[0])].append(int(i[1]))
        else:
            user_rid[int(i[0])] = [int(i[1])]
            user_reviews[int(i[0])] = [review_index]
        if item_reviews.__contains__(int(i[1])):
            item_reviews[int(i[1])].append(review_index)
            item_rid[int(i[1])].append(int(i[0]))
        else:
            item_reviews[int(i[1])] = [review_index]
            item_rid[int(i[1])] = [int(i[0])]

    # test dataset,
    for i in data2.values:
        if user_reviews.__contains__(int(i[0])):
            l_content = 1
        else:
            user_rid[int(i[0])] = []
            user_reviews[int(i[0])] = []
        if item_reviews.__contains__(int(i[1])):
            l_content = 1
        else:
            item_reviews[int(i[1])] = []
            item_rid[int(i[1])] = []

    u_len, i_len, u2_len, i2_len = compute_len(user_reviews, item_reviews)

    # the padded triple is 0
    user_reviews = pad_reviews(user_reviews, u_len, u2_len)
    item_reviews = pad_reviews(item_reviews, i_len, i2_len)

    with open(os.path.join(TPS_DIR, 'user_review'), 'wb') as f1:
        pickle.dump(user_reviews, f1, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(TPS_DIR, 'item_review'), 'wb') as f2:
        pickle.dump(item_reviews, f2, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(TPS_DIR, 'user_rid'), 'wb') as f3:
        pickle.dump(user_rid, f3, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(TPS_DIR, 'item_rid'), 'wb') as f4:
        pickle.dump(item_rid, f4, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(os.path.join(PRE_TPS_DIR, 'entity_index'), 'wb') as f_ti:
    #     pickle.dump(entity_index, f_ti, protocol=pickle.HIGHEST_PROTOCOL)

    usercount, itemcount = get_count(
        data, 'user_id'), get_count(data, 'item_id')

    print(np.sort(np.array(usercount.values)))

    print(np.sort(np.array(itemcount.values)))
