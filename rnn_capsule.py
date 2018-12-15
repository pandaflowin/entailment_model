import tensorflow as tf

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

def label2activate(label, n):
    onehot = tf.one_hot(label, n)
    yi = tf.ones_like(onehot) - 2 * onehot
    return yi

def instance_representation(H):
    return tf.reduce_mean(H, 1)

def J(p, y, n, yact=label2activate):
    yi = yact(y, n)
    j = 1 + tf.reduce_sum(yi * p, 1)
    return tf.reduce_sum(tf.nn.relu(j))

def U(H, r, y, n, yact=label2activate):
    yi = yact(y, n)
    #r: [B, 3, d], ir: [B, d, 1]
    vs = instance_representation(H) #ir: [B,d]
    #normalize vector
    # vs_len = tf.squeeze(tf.sqrt(tf.matmul(tf.expand_dims(vs, 1), tf.expand_dims(vs, -1))), -1) #[B, 1]
    # r_len = tf.sqrt(tf.reduce_sum(r * r, -1, keepdims=True)) #[B, 3, 1]
    # _u = tf.squeeze(tf.matmul(r / r_len, tf.expand_dims(vs / vs_len, -1)), -1)
    #
    # _u = tf.nn.relu(_u + tf.one_hot(y, n))
    # u = tf.reduce_sum(_u, -1)
    #
    #u = n + tf.reduce_sum( yi * _u, -1)
    #
    # _u = tf.squeeze(tf.matmul(r, tf.expand_dims(vs, -1)) , -1) #[B, 3]
    # u = 1 + tf.reduce_sum( yi * _u, -1)
    _u = tf.squeeze(tf.matmul(r, tf.expand_dims(vs, -1)) , -1) #[B, 3]
    maxc = tf.reduce_sum(tf.one_hot(y, n) * _u, -1) #[B]
    minf = tf.reduce_max((tf.ones_like(_u) - tf.one_hot(y, n)) * _u, -1) #[B]
    u = 1 - maxc + minf
    return tf.reduce_sum(tf.nn.relu(u))


def hinge_loss(label, H, ps, rs, n, jyact=label2activate, uyact=label2activate):
    return J(ps, label, n, yact=jyact) + U(H, rs, label, n, yact=uyact)


class RNN_Capsule:
    def __init__(self, n, label, attention=simple_attention):
        self.n = n
        self.label = label
        self.attention = attention

    def __call__(self, H):
        self.ps, self.rs = capsules(H, self.n, attention=self.attention)
        return self.ps, self.rs

    def loss(self, H, jyact=label2activate, uyact=label2activate):
        self.loss = hinge_loss(self.label, H, self.ps, self.rs, self.n, jyact=jyact, uyact=uyact)
        return self.loss
