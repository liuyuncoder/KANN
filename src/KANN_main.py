import os
import numpy as np
import tensorflow as tf
import pickle
import logging
import time
from KANN import KA
from CustomSchedule import CustomSchedule
from sklearn.metrics import roc_auc_score
logging.basicConfig(level="ERROR")

np.random.seed(2020)
random_seed = 2020

# print("The number of available GPU is: ", len(tf.config.experimental.list_physical_devices()))
# Obtain all physical gpu.
# physical_devices = tf.config.experimental.list_physical_devices('GPU') 
# Setting available GPU.
# tf.config.experimental.set_visible_devices(physical_devices[1:], 'GPU')

def loss_function(real, pred):
    loss_object = tf.keras.losses.MeanSquaredError()
    loss = loss_object(real, pred)

    return loss
# for auc acc metrics.
def classification_loss(real, pred):
    class_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(real, tf.float32), logits=pred))
    return class_loss

def class_evaluate(real_eval, pred_eval):
    auc = roc_auc_score(y_true=real_eval, y_score=pred_eval)
    predictions = [1 if i >= 0.5 else 0 for i in pred_eval]
    acc = np.mean(np.equal(predictions, real_eval))
    return auc, acc
# end

# for rmse evaluation.
def evaluate_metric(real_eval, pred_eval):
    rmse = tf.sqrt(tf.reduce_mean(
        tf.square(tf.subtract(pred_eval, real_eval))))
    return rmse


def create_padding_mask(seq):
    # padding mask is used to change 0 in index sequences into 1.
    mask = tf.cast(tf.equal(seq, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]  # 　broadcasting


@tf.function  # Using TensorFlow to optimize eager code and speed the computing.
def train_step(u_batch, i_batch, uid, iid, reuid, reiid, y_batch):
    # The input inp.shape == (batch_size, input_seq_len)
    # The output of each layer is (batch_size, input_seq_len, d_model)
    input_seq_len = tf.shape(u_batch)[1]
	# The mask of input and output.
    inp_padding_mask = create_padding_mask(u_batch)
    tar_padding_mask = create_padding_mask(i_batch)

    # Record all the computing process of KANN for doing gradient decent later.
    with tf.GradientTape() as tape:
        scores, attn_weights = KA(
            u_batch, i_batch, True, uid, iid, reuid, reiid, inp_padding_mask, tar_padding_mask)
        loss = loss_function(y_batch, scores)

	# updating the trainable parameters of KANN.
    gradients = tape.gradient(loss, KA.trainable_variables)
    optimizer.apply_gradients(zip(gradients, KA.trainable_variables))

    # Recording loss into TensorBoard.
    train_loss(loss)
    # train_accuracy(tar_real, scores)


def evaluate(u_valid, i_valid, userid_valid, itemid_valid, reuid_eval, reiid_eval, y_valid):

    inpv_padding_mask = create_padding_mask(u_valid)
    tarv_padding_mask = create_padding_mask(i_valid)
    scores_evaluate, attn_weights_evaluate = KA(
        u_valid, i_valid, False, userid_valid, itemid_valid, reuid_eval, reiid_eval,
        inpv_padding_mask, tarv_padding_mask)
    # evaluate_loss = classification_loss(y_valid, scores_evaluate)
    evaluate_loss = loss_function(y_valid, scores_evaluate)
    # print("valid loss is: ", evaluate_loss)
    valid_loss(evaluate_loss)
    rmse = evaluate_metric(y_valid, scores_evaluate)
    # print("valid rmse is: ", rmse)
    valid_rmse(rmse)
    # return evaluate_loss, rmse, attn_weights_evaluate


def test(u_test, i_test, userid_test, itemid_test, reuid_test, reiid_test, y_test):
    inptest_padding_mask = create_padding_mask(u_test)
    tartest_padding_mask = create_padding_mask(i_test)
    scores_test, attn_weights_test = KA(
        u_test, i_test, False, userid_test, itemid_test, reuid_test, reiid_test,
        inptest_padding_mask, tartest_padding_mask)
    loss_test = loss_function(y_test, scores_test)
    # print("test loss is: ", loss_test)
    test_loss(loss_test)
    rmse_test = evaluate_metric(y_test, scores_test)
    # print("test rmse is: ", rmse_test)
    test_rmse(rmse_test)


output_dir = '../output/amazon/10core'
data_dir = '../data/amazon/10core'

checkpoint_path = os.path.join(output_dir, "checkpoints")
log_dir = os.path.join(output_dir, 'logs')

print("Loading data...")
tain_data_dir = os.path.join(data_dir, 'amazon.train')
valid_data_dir = os.path.join(data_dir, 'amazon.valid')
test_data_dir = os.path.join(data_dir, 'amazon.test')
utext_dir = os.path.join(data_dir, 'user_review')
itext_dir = os.path.join(data_dir, 'item_review')
para_dir = os.path.join(data_dir, 'amazon.para')

tain_pkl = open(tain_data_dir, 'rb')
valid_pkl = open(valid_data_dir, 'rb')
test_pkl = open(test_data_dir, 'rb')
utext_pkl = open(utext_dir, 'rb')
itext_pkl = open(itext_dir, 'rb')
para_pkl = open(para_dir, 'rb')

train_data = pickle.load(tain_pkl)
valid_data = pickle.load(valid_pkl)
test_data = pickle.load(test_pkl)
u_text = pickle.load(utext_pkl)
i_text = pickle.load(itext_pkl)
para = pickle.load(para_pkl)
user_num = para['user_num']
item_num = para['item_num']
review_num_u = para['review_num_u']
review_num_i = para['review_num_i']
review_len_u = para['review_len_u']
review_len_i = para['review_len_i']

train_length = para['train_length']
valid_length = para['valid_length']
test_length = para['test_length']

train_data = np.array(train_data)
valid_data = np.array(valid_data)
test_data = np.array(test_data)

# Initialize KANN.
num_layers = 2
d_model = 50
num_heads = 4
dff = 1024
batch_size = 64
EPOCHS = 38
KA = KA(user_num, item_num, num_layers, d_model, num_heads, dff)

# Custom learning rate.
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

train_loss = tf.keras.metrics.Mean(name='train_loss')
valid_loss = tf.keras.metrics.Mean(name='valid_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')
valid_rmse = tf.keras.metrics.Mean(name='valid_rmse')
test_rmse = tf.keras.metrics.Mean(name='test_rmse')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
#     name='train_accuracy')


# Compare the results of different parameters.
run_id = f"wordemb_{num_layers}layers_{d_model}d_{num_heads}heads_{dff}dff_head"
checkpoint_path = os.path.join(checkpoint_path, run_id)
log_dir = os.path.join(log_dir, run_id)

# Using tf.train.Checkpoint to intergrate the savings, easy saving and reading.
# Save the model of KANN and the status of the optimizer.
ckpt = tf.train.Checkpoint(KA=KA, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(
    ckpt, checkpoint_path, max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)

    # Confirm the number of epoches.
    last_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])
    print(f'have read the newest checkpoint，and the model has trained {last_epoch} epochs。')
else:
    last_epoch = 0
    print("haven't found checkpoint，training from start。")


print(f"KANN has trained {last_epoch} epochs。")
print(f"Left epochs：{min(0, last_epoch - EPOCHS)}")

# write to TensorBoard.
summary_writer = tf.summary.create_file_writer(log_dir)
data_size_train = len(train_data)
data_size_valid = len(valid_data)
data_size_test = len(test_data)

ll = int(data_size_train / batch_size)
ll_valid = int(data_size_valid / batch_size)
ll_test = int(data_size_test / batch_size)
# Compare the setting of `EPOCHS` and trained `last_epoch` to determine the left epochs.
for epoch in range(last_epoch, EPOCHS):
    start = time.time()
    # Shuffle the data at each epoch
    shuffle_indices = np.random.permutation(
        np.arange(data_size_train))
    shuffled_data = train_data[shuffle_indices]
    # rerecord metrics of TensorBoard.
    train_loss.reset_states()
    valid_loss.reset_states()
    valid_rmse.reset_states()
    test_loss.reset_states()
    test_rmse.reset_states()
#    train_accuracy.reset_states()

    # Dealing with one epoch batch by batch.
    for batch_num in range(ll):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size_train)
        data_train = shuffled_data[start_index:end_index]
        uid, iid, reuid, reiid, y_batch = zip(*data_train)
        u_batch = []
        i_batch = []
        # len(uid)
        for i in range(len(uid)):
            u_review = u_text[uid[i][0]]
            i_review = i_text[iid[i][0]]
            # input_reuid = reuid[i]
            # input_reiid = reiid[i]
            u_review = np.reshape(u_review, (-1))
            i_review = np.reshape(i_review, (-1))
            u_batch.append(u_review)
            i_batch.append(i_review)
        u_batch = np.array(u_batch)
        i_batch = np.array(i_batch)
        y_batch = np.array(y_batch)
        # print("u_batch", u_batch.shape)
        # For each time, one step input parameters into KANN，generate prediction results and compute minimized gradient loss.
        train_step(u_batch, i_batch, uid, iid, reuid, reiid, y_batch)

    print("\nEvaluation:")
    print("the batch number is: ", batch_num)
    # shuffle_indices_valid = np.random.permutation(np.arange(data_size_valid))
    # valid_data = valid_data[shuffle_indices_valid]
    for valid_batch in range(ll_valid):
        start_index = valid_batch * batch_size
        end_index = min((valid_batch + 1) *
                        batch_size, data_size_valid)
        data_valid = valid_data[start_index:end_index]
        userid_valid, itemid_valid, reuid_valid, reiid_valid, y_valid = zip(
            *data_valid)
        u_valid = []
        i_valid = []
        # len(uid)
        for i in range(len(userid_valid)):
            u_review_valid = u_text[userid_valid[i][0]]
            i_review_valid = i_text[itemid_valid[i][0]]
            # input_reuid = reuid[i]
            # input_reiid = reiid[i]
            u_review_valid = np.reshape(u_review_valid, (-1))
            i_review_valid = np.reshape(i_review_valid, (-1))
            u_valid.append(u_review_valid)
            i_valid.append(i_review_valid)
        u_valid = np.array(u_valid)
        i_valid = np.array(i_valid)
        # print("u_valid", u_valid.shape)
        evaluate(u_valid, i_valid, userid_valid, itemid_valid,
                 reuid_valid, reiid_valid, y_valid)
    print("\ntest:")
    # shuffle_indices_test = np.random.permutation(np.arange(data_size_test))
    # test_data = test_data[shuffle_indices_test]
    for test_batch in range(ll_test):
        start_index = test_batch * batch_size
        end_index = min((test_batch + 1) *
                        batch_size, data_size_test)
        data_test = test_data[start_index:end_index]
        userid_test, itemid_test, reuid_test, reiid_test, y_test = zip(
            *data_test)
        u_test = []
        i_test = []
        # len(uid)
        for i in range(len(userid_test)):
            u_review_test = u_text[userid_test[i][0]]
            i_review_test = i_text[itemid_test[i][0]]
            # input_reuid = reuid[i]
            # input_reiid = reiid[i]
            u_review_test = np.reshape(u_review_test, (-1))
            i_review_test = np.reshape(i_review_test, (-1))
            u_test.append(u_review_test)
            i_test.append(i_review_test)
        u_test = np.array(u_test)
        i_test = np.array(i_test)
        # print("u_valid", u_valid.shape)
        test(u_test, i_test, userid_test, itemid_test, reuid_test, reiid_test, y_test)

    # each epoch do once saving.
    if (epoch + 1) % 1 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                            ckpt_save_path))

	# write train_loss and valid_loss into TensorBoard.
    with summary_writer.as_default():
        tf.summary.scalar(
            "train_loss", train_loss.result(), step=epoch + 1)
        tf.summary.scalar(
            "valid_loss", valid_loss.result(), step=epoch + 1)
        tf.summary.scalar("valid_rmse", valid_rmse.result(), step=epoch + 1)
        tf.summary.scalar(
            "test_loss", test_loss.result(), step=epoch + 1)
        tf.summary.scalar("test_rmse", test_rmse.result(), step=epoch + 1)
        # tf.summary.scalar(
        #     "train_acc", train_accuracy.result(), step=epoch + 1)
    print('Epoch {} Train Loss {:.4f}'.format(epoch + 1, train_loss.result()))
    print('Epoch {} Valid Loss {:.4f}'.format(epoch + 1, valid_loss.result()))
    print('Epoch {} Valid RMSE {:.4f}'.format(epoch + 1, valid_rmse.result()))
    print('Epoch {} Test Loss {:.4f}'.format(epoch + 1, test_loss.result()))
    print('Epoch {} Test RMSE {:.4f}'.format(epoch + 1, test_rmse.result()))
    # print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
    #                                                     train_loss.result(),
    #                                                     train_accuracy.result()))
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
