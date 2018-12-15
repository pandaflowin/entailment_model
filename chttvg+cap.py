#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 18:57:59 2018

@author: huihsuan
"""
import os
from time import time

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from dataset import MultiNli
from nn import embedded, mask, highway_network, multihead_attention, normalize, char_conv
from util import tprint
from rnn_capsule import RNN_Capsule


######parameters

keep_prob = 1
learning_rate = 0.000001
batch_num = 128
max_len = None
filter_size = 3
num_heads = 8 #for transformer
hidden_dim = 300 #a dim reduction after highway network
char_emb_dim=8

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
##############################



tprint("start loading dataset")
mnli = MultiNli("glove.txt.gz", "./DIIN/data/multinli_0.9",
                max_len = max_len,
                batch=batch_num,
                train_epoch=10,
                dev_epoch=1,
                char_emb_dim=char_emb_dim,
                pad2=False,
                #all_printable_char=True,
                #trainfile="multinli_0.9_train_5000.jsonl",
)

tprint("building graph")
BST = time()
############# dataset inputs
weights =mnli.embedding

sentence1 = mnli.sentence1
sentence2 = mnli.sentence2

###labels
labels = mnli.label


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
###############################



##### model
tprint("building embedding")
with tf.variable_scope("word_embedding"):
    glove_embedding = embedded(mnli.embedding)
    embedding_pre = glove_embedding(sentence1)
    embedding_hyp = glove_embedding(sentence2)

with tf.variable_scope("char_embedding"):
    char_embedding = embedded(mnli.char_embedding, name="char")
    char_embedding_pre = char_embedding(sent1char)
    char_embedding_hyp = char_embedding(sent2char)

    with tf.variable_scope("conv") as scope:
        conv_pre = char_conv(char_embedding_pre, filter_size=filter_size)
        scope.reuse_variables()
        conv_hyp = char_conv(char_embedding_hyp, filter_size=filter_size)

embed_pre = tf.concat((embedding_pre, antonym1, exact1to2, synonym1, conv_pre), -1)
embed_hyp = tf.concat((embedding_hyp, antonym2, exact2to1, synonym2, conv_hyp), -1)


tprint("building highway encoder")
hout_pre = highway_network(embed_pre, 2, [tf.nn.sigmoid] * 2, "premise")
hout_hyp = highway_network(embed_hyp, 2, [tf.nn.sigmoid] * 2, "hypothesis")

#peter: dim reduction
hout_pre = normalize(tf.layers.dense(hout_pre, hidden_dim, activation=tf.nn.sigmoid))
hout_hyp = normalize(tf.layers.dense(hout_hyp, hidden_dim, activation=tf.nn.sigmoid))

hout_pre = mask(hout_pre, sent1_mask)
hout_hyp = mask(hout_hyp, sent2_mask)


tprint("build attention")
pre_atten = multihead_attention(hout_pre,
                                hout_pre,
                                hout_pre,
                                h = num_heads,
                                scope="pre_atten"
)

hyp_atten = multihead_attention(hout_hyp,
                                hout_hyp,
                                hout_hyp,
                                h = num_heads,
                                scope="hyp_atten"
)

# p2h_atten = multihead_attention(pre_atten,
#                                 hyp_atten,
#                                 hyp_atten,
#                                 h = num_heads,
#                                 scope="p2h_atten"
# )

# h2p_atten = multihead_attention(hyp_atten,
#                                 pre_atten,
#                                 pre_atten,
#                                 h = num_heads,
#                                 scope="h2p_atten"
# )


# ##concat the output of hw &attention

tprint("build attention integration")
concatP =tf.concat(values = [hout_pre, pre_atten],axis = 2, name='concatP')
concatH =tf.concat(values = [hout_hyp, hyp_atten],axis = 2, name='concatH')

#[B, L, 300]
mulP =tf.multiply(hout_pre, pre_atten)
mulH =tf.multiply(hout_hyp, hyp_atten)

#[B, L, 300]
subP =tf.abs(tf.subtract(hout_pre, pre_atten))
subH =tf.abs(tf.subtract(hout_hyp, hyp_atten))

#[B, L, 600+300+300]
P_ = normalize(tf.concat([concatP, mulP, subP], axis=2))
H_ = normalize(tf.concat([concatH, mulH, subH], axis=2))

P_ = mask(P_, sent1_mask)
H_ = mask(H_, sent2_mask)

# concatP2H =tf.concat(values = [hout_pre, p2h_atten],axis = 2, name='concatP2H')
# concatH2P =tf.concat(values = [hout_hyp, h2p_atten],axis = 2, name='concatH2P')

# #[B, L, 300]
# mulP2H =tf.multiply(hout_pre, p2h_atten)
# mulH2P =tf.multiply(hout_hyp, h2p_atten)

# #[B, L, 300]
# subP2H =tf.abs(tf.subtract(hout_pre, p2h_atten))
# subH2P =tf.abs(tf.subtract(hout_hyp, h2p_atten))

# #[B, L, 600+300+300]
# PH_ = tf.concat([concatP2H, mulP2H, subP2H], axis=2)
# HP_ = tf.concat([concatH2P, mulH2P, subH2P], axis=2)

# PH_ = mask(PH_, sent1_mask)
# HP_ = mask(HP_, sent2_mask)

# P_ = tf.concat([P_, PH_], 2)
# H_ = tf.concat([H_, HP_], 2)


# #[B, L, 1200]
# #ph = tf.concat([P_,H_], axis=1)


# ###baseline:dynamic_rnn
# tprint("build rnn")
# rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=128)
# p_outputs, p_state = tf.nn.dynamic_rnn(rnn_cell, P_,
#                                      sequence_length=sent1_len,
#                                      dtype=tf.float32)

# h_outputs, h_state = tf.nn.dynamic_rnn(rnn_cell, H_,
#                                      sequence_length=sent2_len,
#                                      initial_state=p_state,
#                                      dtype=tf.float32)

p_outputs = P_
h_outputs = H_


tprint("build rnn-capsule")
outputs = tf.concat([p_outputs, h_outputs], 1)
rnn_capsule = RNN_Capsule(3, labels)

ps, rs = rnn_capsule(outputs)
y = ps

# # training
# loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=y))

# with tf.variable_scope("final") as scope:
#     final = highway_network(rs, 2, [tf.nn.sigmoid] * 2, "reconstruct")
#     scope.reuse_variables()
#     ph_final = highway_network(h_state, 2, [tf.nn.sigmoid] * 2, "reconstruct", reuse=True)

# _y = tf.layers.dense(final, 1) #[B, 3, 1]
# _y = tf.squeeze(_y, -1) #[B, 3] 
# _ph = tf.layers.dense(ph_final, 3) #[B, 3]


tprint("build loss")
loss = rnn_capsule.loss(outputs)
# loss += tf.reduce_mean(tf.reduce_sum(tf.square(_y  - _ph), -1))
l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
loss += l2_loss * 9e-5



tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 1.0)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#train_op = optimizer.minimize(loss)
train_op = optimizer.apply_gradients(zip(grads, tvars))


##evaluate

# current accuracy
predictlabel = tf.argmax(y, axis=1)
correctlabel = tf.cast(tf.equal(predictlabel, labels), dtype=tf.float32)
correctnumber = tf.reduce_sum(correctlabel)
correntPred = tf.reduce_mean(correctlabel)


tprint(f"finish build graph. take {time()-BST} seconds.")


init = tf.global_variables_initializer()
saver = tf.train.Saver()

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
sess.run(init)

# saver.save(sess, "model/basemodel_v1")
saver.restore(sess, "model/cap+hinge")


para_num = sum([np.prod(sess.run(tf.shape(v))) for v in tf.trainable_variables()])
tprint(f"parameters num: {para_num}")

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
                    tprint(f"{name}> average_loss:{local_loss/printnum}, local_accuracy:{local_pred/printnum}")
                    local_pred = 0.
                    local_loss = 0.
            except tf.errors.OutOfRangeError:
                break
        tprint(f"{name}> total_loss:{total_loss/batch_number}, total_accuracy:{total_pred/batch_number}"
)



for i in tqdm(range(1000)):
    tprint(f"train epoch: {i}")
    run(mnli.train, train=True, name="train")
    tprint(f"evaluate on dev_matched")
    run(mnli.dev_matched, name="matched")
    tprint(f"evaluate on dev_mismatched")
    run(mnli.dev_mismatched, name="mismatched")

tprint("done!")



