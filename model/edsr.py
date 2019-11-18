import tensorflow as tf
from functools import partial
from tensorflow.keras.layers import Add, Conv1D, Lambda, PReLU
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from model.common import pixel_shuffle, denormalize, normalize, normalize_1, denormalize_1

def edsr(scale, num_filters=32, res_blocks=[9,9,9,5,5,5,3,3], res_block_scaling=None, normalize_input=False, summary=True):
    assert type(res_blocks) == list, 'res_blocks should be list type'

    print(f'Number of Residual blocks {len(res_blocks)}')

    x_in = Input(shape=(None, 1))
    x = Lambda(normalize, name='Normalize')(x_in)
    x = b = Conv1D(num_filters, 9, padding='same')(x)

    # residual blocks
    for f in res_blocks:
        x = residual_block(x, num_filters, f, res_block_scaling)

    b = Conv1D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    x = Conv1D(1, 3, padding='same')(x)
    x = Lambda(denormalize, name='Denormalize')(x)

    model = Model(x_in, x)

    if summary:
        model.summary()

    return model

def residual_block(x_in, num_filters, f, scaling):
    x = Conv1D(num_filters, f, padding='same')(x_in)
    x = PReLU(alpha_initializer='zeros', shared_axes=[1])(x)
    x = Conv1D(num_filters, f, padding='same')(x)
    if scaling:
        x = Lambda(lambda x : x * scaling, name='res_scaling')(x)
    x = Add()([x, x_in])
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
