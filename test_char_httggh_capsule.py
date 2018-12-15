#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 18:57:59 2018

@author: huihsuan
"""
from dataset import MultiNli
from tqdm import tqdm

import tensorflow as tf
from time import ctime
import numpy as np


from rnn_capsule import RNN_Capsule
#from tensorflow.python.ops import rnn_cell
#from tensorflow.contrib import legacy_seq2seq as seq2seq
#from tensorflow.contrib import grid_rnn

#from tensorflow.contrib.rnn import GridLSTMCell

from tensorflow.contrib.rnn import UGRNNCell
##argument
#import dataset 


######parameters


keep_prob = 1
learning_rate = 0.0004#0.5 #peter: 0.001 -> 0.5 -> 0.00005 (20180628)
batch_num = 16
max_len = 100
num_heads = 5 #for transformer
hidden_dim = 200 #a dim reduction after highway network

##############################




mnli = MultiNli("glove.txt.gz", "./DIIN/data/multinli_0.9",
                max_len = max_len,
                batch=batch_num,
                train_epoch=1,
                dev_epoch=1,
                char_emb_dim=8,
                #trainfile="multinli_0.9_train_5000.jsonl",
)

#word to index
#get glove data to tensorflow embedding layer
#word2idx = {}
#weights = []
#with open('myglove.txt','r') as file:
#    for index, line in tqdm(enumerate(file)):
#        values = line.split() # Word and weights separated by space
#        vocab = values[0] # Word is first symbol on each line
#        embd = np.asarray(values[1:], dtype=np.float32) # Remainder of line is weights for word
        #if vocab in word2idx.keys():
        #    print(vocab)
        #else:
#        word2idx[vocab] = index #+ 2 # PAD is our zeroth index so shift by one
#        weights.append(embd)
        # if index > 100:
        #     break

weights =mnli.embedding #np.array(weights)

sentence1 = mnli.sentence1#tf.placeholder(name='EmbeddingInput1',dtype = tf.int64, shape=(None, None))
sentence2 = mnli.sentence2#tf.placeholder(name='EmbeddingInput2',dtype = tf.int64, shape=(None, None))



sent1_mask = tf.cast(tf.sign(sentence1), dtype=tf.float32)
sent2_mask = tf.cast(tf.sign(sentence2), dtype=tf.float32)
sent1_len = tf.reduce_sum(sent1_mask, -1)
sent2_len = tf.reduce_sum(sent2_mask, -1)

#labels = mnli.label

#######################################
#20180612 add: take shared data
antonym1  = tf.expand_dims(mnli.antonym1, -1)
antonym2  = tf.expand_dims(mnli.antonym2, -1)
exact1to2 = tf.expand_dims(mnli.exact1to2, -1)
exact2to1 = tf.expand_dims(mnli.exact2to1, -1)
synonym1  = tf.expand_dims(mnli.synonym1, -1)
synonym2  = tf.expand_dims(mnli.synonym2, -1)
#20180630 add char
sent1char = mnli.sent1char
sent2char = mnli.sent2char
#######################################
###helper function
def embedded(weights, name="", trainable=True):
    weight_init = tf.constant_initializer(weights)
    embedding_weights = tf.get_variable(
        name=f'{name + "_" if name else ""}embedding_weights', shape=weights.shape,
        initializer=weight_init,
        trainable=trainable)

    def lookup(x):
        nonlocal embedding_weights
        return tf.nn.embedding_lookup(embedding_weights, x)

    return lookup


def char_conv(inp,
              filter_size=5,
              channel_out=100,
              strides=[1, 1, 1, 1],
              padding="SAME",
              dilations=[1, 1, 1, 1]):
    inc = inp.get_shape()[-1]
    filts = tf.get_variable("char_filter", shape=(1, filter_size, inc, channel_out), dtype=tf.float32)
    bias = tf.get_variable("char_bias", shape=(channel_out,), dtype=tf.float32)
    conv = tf.nn.conv2d(inp, filts,
                        strides=strides,
                        padding=padding,
                        dilations=dilations) + bias
    out = tf.reduce_max(tf.nn.relu(conv), 2)
    return out


##########################################

with tf.variable_scope("word_embedding"):
    glove_embedding = embedded(mnli.embedding)
    embedding_pre = glove_embedding(sentence1)
    embedding_hyp = glove_embedding(sentence2)

with tf.variable_scope("char_embedding"):
    char_embedding = embedded(mnli.char_embedding, name="char")
    char_embedding_pre = char_embedding(sent1char)
    char_embedding_hyp = char_embedding(sent2char)

    with tf.variable_scope("conv") as scope:
        conv_pre = char_conv(char_embedding_pre)
        scope.reuse_variables()
        conv_hyp = char_conv(char_embedding_hyp)

# 20180612 add: concat the information from shared.json
# peter: change [-1] -> -1
embed_pre = tf.concat((embedding_pre, antonym1, exact1to2, synonym1, conv_pre), -1)
embed_hyp = tf.concat((embedding_hyp, antonym2, exact2to1, synonym2, conv_hyp), -1)


def mask(x, x_mask=None):
    if x_mask is None:
        return x

    dim = x.get_shape()[-1]
    mask = tf.tile(tf.expand_dims(x_mask, -1), [1, 1, dim])
    return x * mask


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def highway(x, activation):
    size = x.get_shape()[-1]
    W = tf.get_variable("W", (size, size),
                        initializer=tf.random_normal_initializer())
    b = tf.get_variable("b", size,
                        initializer=tf.constant_initializer(0.0))

    Wt = tf.get_variable("Wt", (size, size),
                         initializer=tf.random_normal_initializer())
    bt = tf.get_variable("bt", size,
                         initializer=tf.constant_initializer(0.0))

    T = tf.sigmoid(tf.tensordot(x, Wt, 1) + bt, name="transform_gate")
    H = activation(tf.tensordot(x, W, 1) + b, name="activation")
    C = tf.subtract(1.0, T, name="carry_gate")

    y = tf.add(tf.multiply(H, T), tf.multiply(x, C), "y")
    return y


def highway_network(x, num, activation, name, reuse=None):
    for i, a in zip(range(num), activation):
        with tf.variable_scope(f"{name}_highway{i+1}", reuse=reuse):
            x = highway(x, a)
    return x




#peter: change embedding_xxx to embed_xxx
hout_pre = highway_network(embed_pre, 2, [tf.nn.sigmoid] * 2, "premise")
hout_hyp = highway_network(embed_hyp, 2, [tf.nn.sigmoid] * 2, "hypothesis")

#peter: dim reduction
hout_pre = normalize(tf.layers.dense(hout_pre, hidden_dim, activation=tf.nn.sigmoid))
hout_hyp = normalize(tf.layers.dense(hout_hyp, hidden_dim, activation=tf.nn.sigmoid))

hout_pre = mask(hout_pre, sent1_mask)
hout_hyp = mask(hout_hyp, sent2_mask)




####attention





def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.softmax) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.softmax) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.softmax) # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs) # (N, T_q, C)

    return outputs



pre_atten = multihead_attention(queries=hout_pre,
                        keys = hout_pre,
                        num_units= hidden_dim,#20180612 300->303
                        num_heads= num_heads, #peter: change 10 -> 3
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="pre_multihead_attention",
                        reuse=None)

hyp_atten = multihead_attention(queries=hout_hyp,
                        keys = hout_hyp,
                        num_units= hidden_dim,#20180612 300->303
                        num_heads=num_heads, #peter: change 10 -> 3
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="hyp_multihead_attention",
                        reuse=None)

p2h_atten = multihead_attention(queries=hout_pre,
                        keys = hout_hyp,
                        num_units= hidden_dim,#20180612 300->303
                        num_heads= num_heads, #peter: change 10 -> 3
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="p2h_multihead_attention",
                        reuse=None)
h2p_atten = multihead_attention(queries=hout_hyp,
                        keys = hout_pre,
                        num_units= hidden_dim,#20180612 300->303
                        num_heads= num_heads, #peter: change 10 -> 3
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="h2p_multihead_attention",
                        reuse=None)
##concat the output of hw &attention

#[B, L, 300+300]
concatP =tf.concat(values = [hout_pre, pre_atten],axis = 2, name='concatP')
concatH =tf.concat(values = [hout_hyp, hyp_atten],axis = 2, name='concatH')

#[B, L, 300]
mulP =tf.multiply(hout_pre, pre_atten)
mulH =tf.multiply(hout_hyp, hyp_atten)

#[B, L, 300]
subP =tf.abs(tf.subtract(hout_pre, pre_atten))
subH =tf.abs(tf.subtract(hout_hyp, hyp_atten))

#[B, L, 600+300+300]
P_ = tf.concat([concatP, mulP, subP, p2h_atten], axis=2)
H_ = tf.concat([concatH, mulH, subH, h2p_atten], axis=2)

P_ = mask(P_)
H_ = mask(H_)

#[B, L, 1200]
#ph = tf.multiply(P_,H_)

#####multiply(P_,H_) ,  [N, PL, HL, 2d]
def mulph(p, h):  # [b, L, d]

    PL = tf.shape(p)[1]
    HL = tf.shape(h)[1]
    p_aug = tf.tile(tf.expand_dims(p, 2), [1, 1, HL, 1])
    h_aug = tf.tile(tf.expand_dims(h, 1), [1, PL, 1, 1])  # [N, PL, HL, 2d]    h_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(h_mask, 1), [1, PL
    ph = p_aug * h_aug

    return ph

ph_ = mulph(P_,H_)
#with tf.Session() as sess:
#    sess.run(hout, feed_dict = {Input1: sentence1, Input2: sentence2})

#[B, pl, hl, d]
pl_sum_pool = tf.reduce_sum(ph_, 1) #[B, hl, d]
hl_sum_pool = tf.reduce_sum(ph_, 2) #[B, pl, d]
pl_ave_pool = tf.reduce_mean(ph_, 1)
hl_ave_pool = tf.reduce_mean(ph_, 2)
pl_max_pool = tf.reduce_max(ph_, 1)
hl_max_pool = tf.reduce_max(ph_, 2)

#1
#ph = tf.concat([pl_sum_pool, hl_sum_pool, pl_ave_pool, hl_ave_pool, pl_max_pool, hl_max_pool], axis=2)

#2
pl_ = tf.concat([pl_sum_pool, pl_ave_pool, pl_max_pool, P_], axis = 2)
hl_ = tf.concat([hl_sum_pool, hl_ave_pool, hl_max_pool, H_], axis = 2)

pl_ = tf.layers.dense(pl_, hidden_dim)
hl_ = tf.layers.dense(hl_, hidden_dim)


ph = tf.concat([pl_, hl_], axis = 2)

ph = tf.layers.dense(ph, hidden_dim)


#num_layers = 1
#num_frequency_blocks = [60, 60]


###UGRNNCell
rnn_cell = UGRNNCell(num_units=128)
# pl_outputs, pl_state = tf.nn.dynamic_rnn(rnn_cell, pl_,
#                                     dtype=tf.float32)
# hl_outputs, hl_state = tf.nn.dynamic_rnn(rnn_cell, hl_,
#                                     dtype=tf.float32)
ph_outputs, ph_state = tf.nn.dynamic_rnn(rnn_cell, ph,
                                    dtype=tf.float32)


# p_outputs = tf.concat([pl_state, ph_state], axis = 1)
# h_outputs = tf.concat([hl_state, ph_state], axis = 1)

outputs = ph_outputs#tf.concat([pl_outputs, hl_outputs], axis = 1)

labels = mnli.label
onehotlabel = tf.one_hot(labels, 3)

rnn_capsule = RNN_Capsule(3, labels)

ps, rs = rnn_capsule(outputs)
loss = rnn_capsule.loss(outputs)
y = ps
with tf.variable_scope("final") as scope:
    final = highway_network(rs, 2, [tf.nn.sigmoid] * 2, "reconstruct")
    scope.reuse_variables()
    ph_final = highway_network(ph_state, 2, [tf.nn.sigmoid] * 2, "reconstruct", reuse=True)

_y = tf.layers.dense(final, 1) #[B, 3, 1]
_y = tf.squeeze(_y, -1) #[B, 3] 
_ph = tf.layers.dense(ph_final, 3) #[B, 3]
loss += tf.reduce_mean(tf.reduce_sum(tf.square(_y  - _ph), -1))
                                                                     




#outputs = ph_state
#final = highway_network(outputs, 2, [256] * 2, [tf.nn.sigmoid] * 2, "final")
#y = tf.layers.dense(final, 3)#, activation=tf.nn.relu)

# create a BasicRNNCell
#GRU input: [B, 1200+ 128]
#GRU output: [B, 128]
#cell = GridLSTMCell(num_units=128,  num_frequency_blocks = num_frequency_blocks,start_freqindex_list=[0,0],
#               end_freqindex_list=[60,60], couple_input_forget_gates=True,
#              )


#####gridLSTM from github example
#cell_fn = grid_rnn.Grid2LSTMCell(num_units=128)
#cell = rnn_cell.MultiRNNCell([cell_fn] * num_layers)
#initial_state = cell.zero_state(tf.shape(ph)[0], dtype=tf.float32)
#outputs, state =seq2seq.rnn_decoder(ph, initial_state, cell)
#output = tf.reshape(tf.concat(outputs, axis=1), [-1,128])



###baseline:dynamic_rnn
#rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=128)
#_outputs, state = tf.nn.dynamic_rnn(rnn_cell, ph,
           #                         dtype=tf.float32)
#outputs = state

#rnn_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers,)



# rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
#         rnn_cell, output_keep_prob=keep_prob)




# 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]

# defining initial state
#initial_state = cell.zero_state(tf.shape(ph)[0], dtype=tf.float32)

#RNN hidden impl
# wx * x + wh * h + b
# w * [x ; h] + b

# 'state' is a tensor of shape [batch_size, cell_state_size]
#dynmic_rnn output: [B, L, 128] 



###Decoder

#GRU final output [B, 128]



#final = highway_network(outputs, 1, [128], [tf.nn.relu], "final")
#y = tf.layers.dense(final, 3, activation=tf.nn.relu)
#y = tf.layers.dense(outputs, 3, tf.nn.tanh)

###labels
# labels = mnli.label

# # training
# onehotlabel = tf.one_hot(labels, 3)
# loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
#                                                                      logits=y))  # tf.losses.softmax_cross_entropy
#(onehotlabel, y) #/ tf.cast(tf.shape(y)[0], dtype=tf.float32)
# l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
# loss += l2_loss * 9e-5
# loss = tf.losses.mean_squared_error(labels=onehotlabel, predictions=y)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

##evaluate

# current accuracy
predictlabel = tf.argmax(y, axis=1)
correctlabel = tf.cast(tf.equal(predictlabel, labels), dtype=tf.float32)
correctnumber = tf.reduce_sum(correctlabel)
# bn = tf.cast(tf.shape(labels)[0], dtype=tf.float32)
correntPred = tf.reduce_mean(correctlabel)

# total accuracy
accu_op, accuracy = tf.metrics.accuracy(
    labels,
    predictlabel,
    weights=None,
    metrics_collections=None,
    updates_collections=None,
    name=None
)

# sess = tf.train.MonitoredTrainingSession()# config=tf.ConfigProto(log_device_placement=True))
# sess.run(tf.global_variables_initializer(), {embedding_init: weights})




####big_loop for total epoch
# global_step = tf.train.get_or_create_global_step()
# saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir="test",save_secs=600)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
sess.run(init)

# saver.save(sess, "model/basemodel_v1")
#saver.restore(sess, "model/baseline-v2")


para_num = sum([np.prod(sess.run(tf.shape(v))) for v in tf.trainable_variables()])
print(f"parameters num : {para_num}")

def run(init, e=1, train=False, name=""):
    for epoch in range(e):
        total_loss = 0.
        batch_number = 0
        total_pred = 0.  # total_pred for one epoch
        local_pred = 0.
        local_loss = 0.

        # init_trainset
        init(sess)
        while True:
            try:
                if train:
                    _, loss_value, pred = sess.run((train_op, loss, correntPred))
                else:
                    loss_value, pred = sess.run((loss, correntPred))
                total_loss += loss_value
                local_loss += loss_value
                total_pred += pred
                local_pred += pred
                batch_number += 1
                # bc+=8
                if batch_number % 500 == 0:
                    print(f"{ctime()}: {name}> average_loss:{local_loss/500.}, local_accuracy:{local_pred/500.}")
                    local_pred = 0.
                    local_loss = 0.
            except tf.errors.OutOfRangeError:
                break
        print(f"{ctime()}: {name}> total_loss:{total_loss/batch_number}, total_accuracy:{total_pred/batch_number}")


for i in tqdm(range(1000)):
    print(f"train epoch: {i}")
    run(mnli.train, train=True, name="train")
    print(f"evaluate on dev_matched")
    run(mnli.dev_matched, name="matched")
    print(f"evaluate on dev_mismatched")
    run(mnli.dev_mismatched, name="mismatched")
    # for epoch in range(1):
    #     train_loss = 0.
    #     batch_number = 0
    #     total_pred = 0. #total_pred for one epoch
    #     local_pred = 0.
    #     local_loss = 0.
    #     # bc = 0
    #    #training set

    #         #init_trainset
    #     mnli.train(sess)
    #         #batch_all
    #         #while not sess.should_stop():
    #     while True:
    #         try:
    #             _, loss_value,pred = sess.run((train_op ,loss, correntPred))
    #             train_loss += loss_value
    #             local_loss += loss_value
    #             total_pred += pred
    #             local_pred += pred
    #             batch_number += 1
    #             # bc+=8
    #             if batch_number % 500 == 0:
    #                 print(f"train> average_loss:{local_loss/500.}, local_accuracy:{local_pred/500.}")
    #                 local_pred = 0.
    #                 local_loss = 0.
    #             # if bc > 8000:
    #             #     break
    #         except tf.errors.OutOfRangeError:
    #             break
    #         # except:
    #         #     pass
    #     print(f"train_loss:{train_loss/batch_number}, total_accuracy:{total_pred/batch_number}")



    # dev set

    # dev_loss = 0.
    # batch_number = 0
    # total_pred = 0. #total_pred for one epoch
    # local_pred = 0.
    # local_loss = 0.
    # #init_devset
    # mnli.dev(sess)
    # #batch_all
    # #while not sess.should_stop():
    # while True:
    #     try:
    #         loss_value,pred = sess.run((loss, correntPred))
    #         dev_loss += loss_value
    #         local_loss += loss_value
    #         batch_number += 1
    #         total_pred += pred
    #         local_pred += pred
    #         # bc+=8
    #         if batch_number % 500 == 0:
    #             print(f"dev> average_loss:{local_loss/500.}, local_accuracy:{local_pred/500.}")
    #             local_pred = 0.
    #             local_loss = 0.
    #             # break
    #     except tf.errors.OutOfRangeError:
    #         break
    #     # except:
    #     #     pass
    # print(f"dev_loss:{dev_loss/batch_number}, total_accuracy:{total_pred/batch_number}")

print("done!")





# sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
# Training




