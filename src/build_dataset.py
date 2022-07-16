# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os

TPS_DIR = '../data/imdb/10core'
valid_data = os.path.join(TPS_DIR, "imdb_valid.csv")
test_data = os.path.join(TPS_DIR, "imdb_test.csv")
train_data = os.path.join(TPS_DIR, "imdb_train.csv")
user_review = os.path.join(TPS_DIR, "user_review")
item_review = os.path.join(TPS_DIR, "item_review")
user_review_id = os.path.join(TPS_DIR, "user_rid")
item_review_id = os.path.join(TPS_DIR, "item_rid")


# reid_user_train, reid_user_valid, u_len, item_num + 1
def pad_reviewid(u_train, u_valid, u_test, u_len, num):
    pad_u_train = []

    for i in range(len(u_train)):
        x = u_train[i]  # is a list of item_ids which user i commented.
        while u_len > len(x):
            x.append(num)
        if u_len < len(x):
            x = x[:u_len]
        pad_u_train.append(x)
    pad_u_valid = []

    for i in range(len(u_valid)):
        x = u_valid[i]
        while u_len > len(x):
            x.append(num)
        if u_len < len(x):
            x = x[:u_len]
        pad_u_valid.append(x)

    pad_u_test = []
    for i in range(len(u_test)):
        x = u_test[i]
        while u_len > len(x):
            x.append(num)
        if u_len < len(x):
            x = x[:u_len]
        pad_u_test.append(x)
    return pad_u_train, pad_u_valid, pad_u_test


def load_data(train_data, valid_data, user_rid, item_rid, u_len, i_len, user_num, item_num):
    # Load and preprocess data  u_text, y_train, y_valid, u_len, u2_len, uid_train, iid_train, uid_valid, iid_valid,
    # user_num, reid_user_train, reid_user_valid
    y_train, y_valid, y_test, uid_train, iid_train, uid_valid, iid_valid,\
        uid_test, iid_test, reid_user_train, reid_item_train, reid_user_valid, reid_item_valid,\
        reid_user_test, reid_item_test = \
        load_data_and_labels(train_data, valid_data, user_rid, item_rid)
    print("load data done")
    # u_len: the length of reviews. u2_len: the length of entities.

    reid_user_train, reid_user_valid, reid_user_test = pad_reviewid(
        reid_user_train, reid_user_valid, reid_user_test, u_len, item_num + 1)
    print("pad user done")
    reid_item_train, reid_item_valid, reid_item_test = pad_reviewid(
        reid_item_train, reid_item_valid, reid_item_test, i_len, user_num + 1)
    print("pad item done")

    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    y_test = np.array(y_test)
    uid_train = np.array(uid_train)
    uid_valid = np.array(uid_valid)
    uid_test = np.array(uid_test)
    iid_train = np.array(iid_train)
    iid_valid = np.array(iid_valid)
    iid_test = np.array(iid_test)
    reid_user_train = np.array(reid_user_train)
    reid_user_valid = np.array(reid_user_valid)
    reid_user_test = np.array(reid_user_test)
    reid_item_train = np.array(reid_item_train)
    reid_item_valid = np.array(reid_item_valid)
    reid_item_test = np.array(reid_item_test)

    return [y_train, y_valid, y_test, uid_train, iid_train, uid_valid, iid_valid,
            uid_test, iid_test, reid_user_train, reid_item_train, reid_user_valid, reid_item_valid,
            reid_user_test, reid_item_test]


def load_data_and_labels(train_data, valid_data, user_rid, item_rid):
    # Load data from files
    f_train = open(train_data, "rb")

    with open(user_rid, "rb") as f3:
        user_rids = pickle.load(f3)
    with open(item_rid, "rb") as f4:
        item_rids = pickle.load(f4)

    reid_user_train = []
    reid_item_train = []
    uid_train = []
    iid_train = []
    y_train = []

    i = 0
    for line in f_train:
        i = i + 1
        line = line.decode().split(',')
        uid_train.append(int(line[0]))
        iid_train.append(int(line[1]))
        reid_user_train.append(user_rids[int(line[0])])
        reid_item_train.append(item_rids[int(line[1])])
        y_train.append(float(line[2]))

    print("valid")
    reid_user_valid = []
    reid_item_valid = []

    uid_valid = []
    iid_valid = []
    y_valid = []
    f_valid = open(valid_data)
    for line in f_valid:
        line = line.split(',')
        uid_valid.append(int(line[0]))
        iid_valid.append(int(line[1]))
        reid_user_valid.append(user_rids[int(line[0])])
        reid_item_valid.append(item_rids[int(line[1])])
        y_valid.append(float(line[2]))

    print("test")
    reid_user_test = []
    reid_item_test = []

    uid_test = []
    iid_test = []
    y_test = []
    f_test = open(test_data)
    for line in f_test:
        line = line.split(',')
        uid_test.append(int(line[0]))
        iid_test.append(int(line[1]))
        reid_user_test.append(user_rids[int(line[0])])
        reid_item_test.append(item_rids[int(line[1])])
        y_test.append(float(line[2]))

    return [y_train, y_valid, y_test, uid_train,
            iid_train, uid_valid, iid_valid, uid_test, iid_test, reid_user_train, reid_item_train, reid_user_valid,
            reid_item_valid, reid_user_test, reid_item_test]


if __name__ == '__main__':
    with open(user_review, "rb") as f1:
        user_review = pickle.load(f1)
    with open(item_review, "rb") as f2:
        item_review = pickle.load(f2)
    u_len = len(user_review[0])
    i_len = len(item_review[0])
    user_num = len(user_review)
    item_num = len(item_review)
    y_train, y_valid, y_test, uid_train, iid_train, uid_valid, iid_valid,\
        uid_test, iid_test, reid_user_train, reid_item_train,\
        reid_user_valid, reid_item_valid, reid_user_test, reid_item_test = \
        load_data(train_data, valid_data, user_review_id, item_review_id, u_len, i_len, user_num, item_num)

    np.random.seed(2020)

    shuffle_indices = np.random.permutation(np.arange(len(y_train)))

    userid_train = uid_train[shuffle_indices]
    itemid_train = iid_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    reid_user_train = reid_user_train[shuffle_indices]
    reid_item_train = reid_item_train[shuffle_indices]

    y_train = y_train[:, np.newaxis]
    y_valid = y_valid[:, np.newaxis]
    y_test = y_test[:, np.newaxis]

    userid_train = userid_train[:, np.newaxis]
    itemid_train = itemid_train[:, np.newaxis]
    userid_valid = uid_valid[:, np.newaxis]
    itemid_valid = iid_valid[:, np.newaxis]
    userid_test = uid_test[:, np.newaxis]
    itemid_test = iid_test[:, np.newaxis]

    batches_train = list(
        zip(userid_train, itemid_train, reid_user_train, reid_item_train, y_train))  # align several arrays by
# column and become matrix
    batches_valid = list(
        zip(userid_valid, itemid_valid, reid_user_valid, reid_item_valid, y_valid))
    batches_test = list(zip(userid_test, itemid_test, reid_user_test, reid_item_test, y_test))
    print('write begin')
    output = open(os.path.join(TPS_DIR, 'imdb.train'), 'wb')
    pickle.dump(batches_train, output)
    output = open(os.path.join(TPS_DIR, 'imdb.valid'), 'wb')
    pickle.dump(batches_valid, output)
    output = open(os.path.join(TPS_DIR, 'imdb.test'), 'wb')
    pickle.dump(batches_test, output)

    para = {}
    para['user_num'] = user_num
    para['item_num'] = item_num
    para['review_num_u'] = u_len
    para['review_num_i'] = i_len
    para['review_len_u'] = len(user_review[0][1])
    para['review_len_i'] = len(item_review[0][1])

    para['train_length'] = len(y_train)
    para['valid_length'] = len(y_valid)
    para['test_length'] = len(y_test)

    output = open(os.path.join(TPS_DIR, 'imdb.para'), 'wb')

    # Pickle dictionary using protocol 0.
    pickle.dump(para, output)
