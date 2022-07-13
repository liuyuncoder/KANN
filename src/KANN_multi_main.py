import os
import numpy as np
import tensorflow as tf
import pickle
import logging
import time
from KANN import KA
# from tensorflow.keras.utils import multi_gpu_model
# from multi_gpu import ParallelModel
from CustomSchedule import CustomSchedule
logging.basicConfig(level="ERROR")

np.random.seed(2020)
random_seed = 2020

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

uidt, iidt, _, _, yt_batch = zip(*train_data)
uidt = np.array(uidt)
iidt = np.array(iidt)

yt_batch = np.array(yt_batch)
u_batch = []
i_batch = []
for i in range(train_length):
    u_review = u_text[uidt[i][0]]
    i_review = i_text[iidt[i][0]]
    u_review = np.reshape(u_review, (-1))
    i_review = np.reshape(i_review, (-1))
    u_batch.append(u_review)
    i_batch.append(i_review)
u_batch = np.array(u_batch)
i_batch = np.array(i_batch)

uidt_valid, iidt_valid, _, _, yt_valid = zip(*valid_data)
uidt_valid = np.array(uidt_valid)
iidt_valid = np.array(iidt_valid)
yt_valid = np.array(yt_valid)
u_batch_valid = []
i_batch_valid = []
for j in range(len(uidt_valid)):
    u_review_v = u_text[uidt_valid[j][0]]
    i_review_v = i_text[iidt_valid[j][0]]
    u_review_v = np.reshape(u_review_v, (-1))
    i_review_v = np.reshape(i_review_v, (-1))
    u_batch_valid.append(u_review_v)
    i_batch_valid.append(i_review_v)
u_batch_valid = np.array(u_batch_valid)
i_batch_valid = np.array(i_batch_valid)

uidt_test, iidt_test, _, _, yt_test = zip(*test_data)
uidt_test = np.array(uidt_test)
iidt_test = np.array(iidt_test)
yt_test = np.array(yt_test)
u_batch_test = []
i_batch_test = []
for k in range(test_length):
    u_review_t = u_text[uidt_test[k][0]]
    i_review_t = i_text[iidt_test[k][0]]
    u_review_t = np.reshape(u_review_t, (-1))
    i_review_t = np.reshape(i_review_t, (-1))
    u_batch_test.append(u_review_t)
    i_batch_test.append(i_review_t)
u_batch_test = np.array(u_batch_test)
i_batch_test = np.array(i_batch_test)

strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
BUFFER_SIZE = train_length

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = 50


train_dataset = tf.data.Dataset.from_tensor_slices((uidt, iidt, yt_batch, u_batch, i_batch)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
valid_dataset = tf.data.Dataset.from_tensor_slices((uidt_valid, iidt_valid, yt_valid, u_batch_valid, i_batch_valid)).batch(GLOBAL_BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((uidt_test, iidt_test, yt_test, u_batch_test, i_batch_test)).batch(GLOBAL_BATCH_SIZE)
train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
valid_dist_dataset = strategy.experimental_distribute_dataset(valid_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)
# Initialize KANN.
num_layers = 2
d_model = 50
num_heads = 4
dff = 1024
# Easy to compare the different experimental results with different parameters.
run_id = f"new_{num_layers}layers_{d_model}d_{num_heads}heads_{dff}dff_128batch_size"
checkpoint_path = os.path.join(checkpoint_path, run_id)
log_dir = os.path.join(log_dir, run_id)
def evaluate_metric(real_eval, pred_eval):
    real_eval = tf.cast(real_eval, tf.float32)
    rmse = tf.sqrt(tf.reduce_mean(
            tf.square(tf.subtract(pred_eval, real_eval))))
    return rmse

def create_padding_mask(seq):
    # padding mask is used to change 0 in index sequences into 1.
    mask = tf.cast(tf.equal(seq, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]  # 　broadcasting

with strategy.scope():
    loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    def loss_function(real, pred):
        per_sample_loss = loss_object(real, pred)
        return tf.nn.compute_average_loss(per_sample_loss, global_batch_size=GLOBAL_BATCH_SIZE)

with strategy.scope():
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    valid_rmse = tf.keras.metrics.Mean(name='valid_rmse')
    test_rmse = tf.keras.metrics.Mean(name='test_rmse')

with strategy.scope():
    KA = KA(user_num, item_num, num_layers, d_model, num_heads, dff)

    # Custom learning rate.
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    ckpt = tf.train.Checkpoint(KA=KA, optimizer=optimizer)

with strategy.scope():
    def train_step(x):
        # Record all the computing process of KANN for doing gradient decent later.
        uid, iid, y_batch, u_batch, i_batch = x
        inp_padding_mask = create_padding_mask(u_batch)
        tar_padding_mask = create_padding_mask(i_batch)
        with tf.GradientTape() as tape:
            scores, attn_weights= KA(
                u_batch, i_batch, True, uid, iid, inp_padding_mask, tar_padding_mask)
    #        isarr = ParallelModel(isarr, NUM_GPU)
            loss = loss_function(y_batch, scores)

        # updating the trainable parameters of KANN.
        gradients = tape.gradient(loss, KA.trainable_variables)
        optimizer.apply_gradients(zip(gradients, KA.trainable_variables))

        # Recording loss into TensorBoard.
        return loss
        # train_accuracy(tar_real, scores)


    def evaluate(x_valid):
        uid_valid, iid_valid, y_valid, u_valid, i_valid = x_valid
        inpv_padding_mask = create_padding_mask(u_valid)
        tarv_padding_mask = create_padding_mask(i_valid)
        scores_evaluate, attn_weights_evaluate = KA(
            u_valid, i_valid, False, uid_valid, iid_valid,
            inpv_padding_mask, tarv_padding_mask)
        evaluate_loss = loss_object(y_valid, scores_evaluate)
        # print("valid loss is: ", evaluate_loss)
        valid_loss.update_state(evaluate_loss)
        rmse = evaluate_metric(y_valid, scores_evaluate)
        # print("valid rmse is: ", rmse)
        valid_rmse.update_state(rmse)
        # return evaluate_loss, rmse, attn_weights_evaluate


    def test(x_test):
        uid_test, iid_test, y_test, u_test, i_test = x_test
        inptest_padding_mask = create_padding_mask(u_test)
        tartest_padding_mask = create_padding_mask(i_test)
        scores_test, attn_weights_test = KA(
            u_test, i_test, False, uid_test, iid_test,
            inptest_padding_mask, tartest_padding_mask)
        loss_test = loss_object(y_test, scores_test)
        # print("test loss is: ", loss_test)
        test_loss.update_state(loss_test)
        rmse_test = evaluate_metric(y_test, scores_test)
        # print("test rmse is: ", rmse_test)
        test_rmse.update_state(rmse_test)

with strategy.scope():
    # experimental_run_v2 replicate the computation and run it with distributed mode.
    @tf.function
    def distributed_train_step(x):
        # per_replica_losses = strategy.experimental_run_v2(train_step, args=(x,))
        per_replica_losses = strategy.run(train_step, args=(x,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    @tf.function
    def distributed_valid_step(x_valid):
        # return strategy.experimental_run_v2(evaluate, args=(x_valid,))
        return strategy.run(evaluate, args=(x_valid,))

    @tf.function
    def distributed_test_step(x_test):
        # return strategy.experimental_run_v2(test, args=(x_test,))
        return strategy.run(test, args=(x_test,))

    # Using tf.train.Checkpoint to intergrate the savings, easy saving and reading.
    # Save the model of KANN and the status of the optimizer.
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
    for epoch in range(last_epoch, EPOCHS):
        start = time.time()
        total_loss = 0.0
        num_batches = 0
        valid_loss.reset_states()
        valid_rmse.reset_states()
        test_loss.reset_states()
        test_rmse.reset_states()

        for x in train_dist_dataset:
            # print("u_batch", u_batch.shape)
            # For each time, one step input parameters into KANN，generate prediction results and compute minimized gradient loss.
            total_loss += distributed_train_step(x)
            num_batches += 1
        train_loss = total_loss / num_batches
        print("\nEvaluation:")
        print("the batch number is: ", num_batches)
        for x_valid in valid_dist_dataset:
            distributed_valid_step(x_valid)

        print("\ntest:")
        # shuffle_indices_test = np.random.permutation(np.arange(data_size_test))
        # test_data = test_data[shuffle_indices_test]
        for x_test in test_dist_dataset:
            distributed_test_step(x_test)

        # each epoch do once saving.
        if (epoch + 1) % 1 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                ckpt_save_path))

        # write train_loss and valid_loss into TensorBoard.
        with summary_writer.as_default():
            tf.summary.scalar(
                "train_loss", train_loss, step=epoch + 1)
            tf.summary.scalar(
                "valid_loss", valid_loss.result(), step=epoch + 1)
            tf.summary.scalar("valid_rmse", valid_rmse.result(), step=epoch + 1)
            tf.summary.scalar(
                "test_loss", test_loss.result(), step=epoch + 1)
            tf.summary.scalar("test_rmse", test_rmse.result(), step=epoch + 1)

        print('Epoch {} Train Loss {:.4f}'.format(epoch + 1, train_loss))
        print('Epoch {} Valid Loss {:.4f}'.format(epoch + 1, valid_loss.result()))
        print('Epoch {} Valid RMSE {:.4f}'.format(epoch + 1, valid_rmse.result()))
        print('Epoch {} Test Loss {:.4f}'.format(epoch + 1, test_loss.result()))
        print('Epoch {} Test RMSE {:.4f}'.format(epoch + 1, test_rmse.result()))
        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
