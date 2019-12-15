import tensorflow as tf
from functools import partial
from tensorflow.keras.layers import Add, Conv1D, Lambda, PReLU, Conv2DTranspose
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from model.common import pixel_shuffle, denormalize, normalize, normalize_1, denormalize_1

def edsr(scale, num_filters=32, res_blocks=[9,9,9,5,5,5,3,3], res_block_scaling=None, summary=True):
    assert type(res_blocks) == list, 'res_blocks should be list type'

    print(f'Number of Residual blocks {len(res_blocks)}')

    x_in = Input(shape=(None, 1))
    # x = Lambda(normalize, name='Normalize')(x_in)
    x = b = Conv1D(num_filters, 9, padding='same')(x_in)

    # residual blocks
    for f in res_blocks:
        x = residual_block2(x, num_filters, f, res_block_scaling)

    b = Conv1D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    x = Conv1D(1, 3, padding='same')(x)
    # x = Lambda(denormalize, name='Denormalize')(x)

    model = Model(x_in, x)

    if summary:
        model.summary()

    return model

def edsr2(scale, num_filters=32, res_blocks=[9,9,9,5,5,5,3,3], res_block_scaling=None, summary=True):
    assert type(res_blocks) == list, 'res_blocks should be list type'

    print(f'Number of Residual blocks {len(res_blocks)}')

    x_in = Input(shape=(None, 1))
    # x = Lambda(normalize, name='Normalize')(x_in)
    x = b = Conv1D(num_filters, 9, padding='same')(x_in)

    # residual blocks
    for f in res_blocks:
        x = residual_block(x, num_filters, f, res_block_scaling)

    b = Conv1D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    x = Conv1D(num_filters//2, 3, padding='same', activation='tanh', kernel_initializer='glorot_uniform')(x)
    # x = PReLU(alpha_initializer='zeros', shared_axes=[1])(x)
    x = Conv1D(8, 3, padding='same')(x)
    x = Conv1D(1, 3, padding='same')(x)
    # x = Lambda(denormalize, name='Denormalize')(x)

    model = Model(x_in, x)

    if summary:
        model.summary()

    return model

def residual_block(x_in, num_filters, f, scaling):
    x = Conv1D(num_filters, f, padding='same', activation='tanh', kernel_initializer='glorot_uniform')(x_in)
    # x = PReLU(alpha_initializer='zeros', shared_axes=[1])(x)
    x = Conv1D(num_filters, f, padding='same')(x)
    x = Conv1D(num_filters, f, padding='same')(x)
    if scaling:
        x_in = Lambda(lambda x : x * (1-scaling))(x_in)
        x = Lambda(lambda x : x * scaling)(x)
    x = Add()([x, x_in])
    return x

def residual_block2(x_in, num_filters, f, scaling):
    x1 = x2 = x_in
    # Main branch(auto encoder)
    x1 = Conv1D(num_filters, f, padding='same', activation='tanh')(x1)
    x1 = Conv1D(num_filters, f, padding='same', strides=2)(x1)
    x1 = Conv1D(num_filters, f, padding='same')(x1)
    x1 = Conv1D(num_filters, f, padding='same', strides=2)(x1)
    x1 = Conv1D(num_filters, f, padding='same')(x1)
    x1 = Conv1DTranspose(x1, num_filters, f, 2, 'SAME', [1,0])
    x1 = Conv1DTranspose(x1, num_filters, f, 2, 'SAME', [1,0])

    # Branch_1
    x2 = Conv1D(num_filters, f, padding='same', activation='tanh')(x2)
    # x2 = PReLU(alpha_initializer='zeros', shared_axes=[1])(x2)
    x2 = Conv1D(num_filters, f, padding='same')(x2)
    if scaling:
        x1 = Lambda(lambda x : x * scaling)(x1)
        x2 = Lambda(lambda x : x * scaling)(x2)
        x_in = Lambda(lambda x : x * scaling)(x_in)

    x = Add()([x1, x2, x_in])
    return x

def Conv1DTranspose(x, filters, kernel_size, strides=2, padding='same', output_padding=[2,1]):
    x = Lambda(lambda x : tf.expand_dims(x, axis=2))(x)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides,1), padding=padding, output_padding=output_padding)(x)
    x = Lambda(lambda x : tf.squeeze(x, axis=2))(x)
    return x


def upsample(x, scale, num_filters):
    def upsample_1(x, scale, **kwargs):
        x = Conv1D(num_filters * scale, 3, padding='same', **kwargs)(x)
        return Lambda(partial(pixel_shuffle, scale=scale))(x)

    if scale == 2:
        x = upsample_1(x, scale, name='conv1d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, scale, name='conv1d_1_scale_3')
    else:
        x = upsample_1(x, 2, name='conv1d_1_scale_2')
        x = upsample_1(x, 2, name='conv1d_2_scale_2')

    return x
