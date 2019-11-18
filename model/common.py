import tensorflow as tf

def resolve(model, batch):
    lr_batch = tf.cast(batch, tf.dtypes.float32)
    sr_batch = model(lr_batch)
    sr_batch = denormalize(sr_batch)
    return sr_batch

def evaluate(model, dataset):
    snr_values = []
    for lr, hr in dataset:
        lr, hr = tf.reshape(lr, [-1, lr.shape[1], 1]), tf.reshape(hr, [-1, hr.shape[1], 1])
        sr = model(lr)
        snr_value = snr(hr, sr)
        snr_values.append(snr_value)
    return tf.reduce_mean(snr_values, axis=0)[0]


def normalize(x):
    x = x / 2**15
    return tf.cast(x, tf.dtypes.float32)

def denormalize(x):
    x = x * (2**15)
    return tf.cast(x, tf.dtypes.int16)

# ---------------------------------------
#  Metric
# ---------------------------------------
def snr(hr, sr):
    signal = tf.reduce_sum(hr ** 2)
    noise = tf.reduce_sum((hr - sr) ** 2)
    return 20*tf.math.log(signal/noise, name='SNR')

# ---------------------------------------
#  See https://arxiv.org/abs/1609.05158
# ---------------------------------------

def pixel_shuffle(x, scale):
    x = tf.transpose(x, [2,1,0])
    x = tf.batch_to_space(x, [scale], [[0,0]])
    x = tf.transpose(x, [2,1,0])
    return x