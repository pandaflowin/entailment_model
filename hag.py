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

from nn import embedded, mask, highway_network, attention, normalize


######parameters

keep_prob = 1
learning_rate = 0.00005 #peter: 0.001 -> 0.5 -> 0.00005 (20180628)
batch_num = 16
max_len = 100
num_heads = 5 #for transformer
hidden_dim = 300 #a dim reduction after highway network

##############################




mnli = MultiNli("glove.txt.gz", "./DIIN/data/multinli_0.9",
                max_len = max_len,
                batch=batch_num,
                train_epoch=1,
                dev_epoch=1,
                char_emb_dim=8,
                all_printable_char=True,
                #trainfile="multinli_0.9_train_5000.jsonl",
)

weights =mnli.embedding

sentence1 = mnli.sentence1
sentence2 = mnli.sentence2

sent1_mask = tf.cast(tf.sign(sentence1), dtype=tf.float32)
sent2_mask = tf.cast(tf.sign(sentence2), dtype=tf.float32)
sent1_len = tf.reduce_sum(sent1_mask, -1)
sent2_len = tf.reduce_sum(sent2_mask, -1)

antonym1  = tf.expand_dims(mnli.antonym1, -1)
antonym2  = tf.expand_dims(mnli.antonym2, -1)
exact1to2 = tf.expand_dims(mnli.exact1to2, -1)
exact2to1 = tf.expand_dims(mnli.exact2to1, -1)
synonym1  = tf.expand_dims(mnli.synonym1, -1)
synonym2  = tf.expand_dims(mnli.synonym2, -1)
sent1char = mnli.sent1char
sent2char = mnli.sent2char

with tf.variable_scope("word_embedding"):
    glove_embedding = embedded(mnli.embedding)
    embedding_pre = glove_embedding(sentence1)
    embedding_hyp = glove_embedding(sentence2)

# with tf.variable_scope("char_embedding"):
#     char_embedding = embedded(mnli.char_embedding, name="char")
#     char_embedding_pre = char_embedding(sent1char)
#     char_embedding_hyp = char_embedding(sent2char)

#     with tf.variable_scope("conv") as scope:
#         conv_pre = char_conv(char_embedding_pre)
#         scope.reuse_variables()
#         conv_hyp = char_conv(char_embedding_hyp)

# embed_pre = tf.concat((embedding_pre, antonym1, exact1to2, synonym1, conv_pre), -1)
# embed_hyp = tf.concat((embedding_hyp, antonym2, exact2to1, synonym2, conv_hyp), -1)

embed_pre = tf.concat((embedding_pre, antonym1, exact1to2, synonym1), -1)
embed_hyp = tf.concat((embedding_hyp, antonym2, exact2to1, synonym2), -1)

hout_pre = highway_network(embed_pre, 2, [tf.nn.sigmoid] * 2, "premise")
hout_hyp = highway_network(embed_hyp, 2, [tf.nn.sigmoid] * 2, "hypothesis")

#peter: dim reduction
hout_pre = normalize(tf.layers.dense(hout_pre, hidden_dim, activation=tf.nn.sigmoid))
hout_hyp = normalize(tf.layers.dense(hout_hyp, hidden_dim, activation=tf.nn.sigmoid))

hout_pre = mask(hout_pre, sent1_mask)
hout_hyp = mask(hout_hyp, sent2_mask)

pre_atten = attention(hout_pre,
                      hout_pre)

hyp_atten = attention(hout_hyp,
                      hout_hyp)

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
P_ = tf.concat([concatP, mulP, subP], axis=2)
H_ = tf.concat([concatH, mulH, subH], axis=2)

P_ = mask(P_)
H_ = mask(H_)

#[B, L, 1200]
ph = tf.concat([P_,H_], axis=1)


###baseline:dynamic_rnn
rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=128)
_outputs, state = tf.nn.dynamic_rnn(rnn_cell, ph,
                                    dtype=tf.float32)
outputs = state

# 'state' is a tensor of shape [batch_size, cell_state_size]
#dynmic_rnn output: [B, L, 128] 

y = tf.layers.dense(outputs, 3)

###labels
labels = mnli.label

# # training
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                     logits=y))
l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
loss += l2_loss * 9e-5

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

##evaluate

# current accuracy
predictlabel = tf.argmax(y, axis=1)
correctlabel = tf.cast(tf.equal(predictlabel, labels), dtype=tf.float32)
correctnumber = tf.reduce_sum(correctlabel)
correntPred = tf.reduce_mean(correctlabel)


init = tf.global_variables_initializer()
saver = tf.train.Saver()

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
sess.run(init)

# saver.save(sess, "model/basemodel_v1")
#saver.restore(sess, "model/htg")


para_num = sum([np.prod(sess.run(tf.shape(v))) for v in tf.trainable_variables()])
print(f"parameters num : {para_num}")

def run(init, e=1, train=False, name="", printnum=500):
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
                if batch_number % printnum == 0:
                    print(f"{ctime()}: {name}> average_loss:{local_loss/printnum}, local_accuracy:{local_pred/printnum}")
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

print("done!")



