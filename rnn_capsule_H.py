import tensorflow as tf
import numpy as np

def simple_attention(x):
    e = tf.layers.dense(x, 1, use_bias=False)
    a = tf.nn.softmax(e, 1)
    v = a * x
    return tf.reduce_sum(v, 1)

def capsule(x, attention=simple_attention):
    vc = attention(x)
    p = tf.layers.dense(vc, 1, activation=tf.nn.sigmoid)
    r = p * vc
    return p, r

def capsules(x, n, attention=simple_attention):
    caps = []
    for i in range(n):
        with tf.variable_scope(f"capsule_{i}"):
            cap = capsule(x, attention=attention)
            caps.append(cap)

    ps = tf.concat([p for p,r in caps], -1) #[B, 3]
    rs = tf.concat([tf.expand_dims(r, 1) for p,r in caps], 1) #[B, 3, d]
    return ps, rs

def true_mask(label, n):
    return tf.one_hot(label, n)

def false_mask(label, n):
    tm = true_mask(label, n)
    return tf.ones_like(tm) - tm

def false_inf_mask(label, n, y):
    tm = true_mask(label, n)
    return tf.where(tf.cast(tm, dtype=bool), tf.ones_like(tm) * -tf.constant(np.inf), y)

def instance_representation(H):
    return tf.reduce_mean(H, 1)

def hinge(label, y, n, k=1):
    maxc = tf.reduce_sum( true_mask(label, n) * y, -1)
    minf = tf.reduce_max( false_inf_mask(label, n, y), -1)
    l = k - maxc + minf
    return tf.reduce_sum(tf.nn.relu(l))

def J(p, y, n, k=1):
    return hinge(y, p, n, k)

def U(H, r, y, n, k=1):
    #r: [B, 3, d]
    vs = instance_representation(H) #ir: [B,d]
    u = tf.squeeze(tf.matmul(r, tf.expand_dims(vs, -1)) , -1) #[B, 3]
    return hinge(y, u, n, k)

def hinge_loss(label, H, ps, rs, n, jk=1, uk=1):
    return J(ps, label, n, jk) + U(H, rs, label, n, uk)

class RNN_Capsule:
    def __init__(self, n, label, attention=simple_attention):
        self.n = n
        self.label = label
        self.attention = attention

    def __call__(self, H):
        self.ps, self.rs = capsules(H, self.n, attention=self.attention)
        return self.ps, self.rs

    def loss(self, H, jk=1, uk=1):
        self.loss = hinge_loss(self.label, H, self.ps, self.rs, self.n, jk=jk, uk=uk)
        return self.loss
