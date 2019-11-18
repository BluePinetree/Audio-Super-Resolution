import tensorflow as tf

def resolve(model, batch):
    lr_batch = tf.cast(batch, tf.dtypes.float32)
    sr_batch = model(lr_batch)
    return sr_batch

def evaluate(model, dataset):
    snr_values = []
    for lr, hr in dataset:
        # 전처리
        lr, hr = tf.reshape(lr, (-1, lr.shape[0], 1)), tf.reshape(hr, (-1, hr.shape[0], 1))
        #평가
        sr = model(lr)
        snr_value = snr(hr, sr)
        snr_values.append(snr_value)
    return tf.reduce_mean(snr_values)

def normalize(x):
    x = x / 2**15
    return x

def normalize_1(x):
    x = x + 2**15
    x = x / 2**16
    return x

def denormalize(x):
    x = x * (2**15)
    return x

def denormalize_1(x):
    x = x * (2**16)
    x = x - 2**15
    return x

# ---------------------------------------
#  Metric
# ---------------------------------------
def snr(hr, sr):
    signal = tf.reduce_sum(hr ** 2)
    noise = tf.reduce_sum((hr - sr) ** 2)
    numerator = tf.math.log(signal/noise)
    denorminator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return 10*(numerator/denorminator)

# ---------------------------------------
#  See https://arxiv.org/abs/1609.05158
# ---------------------------------------

def pixel_shuffle(x, scale):
    x = tf.transpose(x, [2,1,0])
    x = tf.batch_to_space(x, [scale], [[0,0]])
    x = tf.transpose(x, [2,1,0])
    return x