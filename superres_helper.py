import tensorflow as tf
import cv2
from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.python.keras.models import Model
import tqdm
import numpy as np
import os

import argparse

parser = argparse.ArgumentParser(description="DeeplabV3+ And Evaluation")

# model parameters

parser.add_argument('--partition', type=int)
parser.add_argument('--oak-fp', type=str)
parser.add_argument('--year', type=int)

args = parser.parse_args()

DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255


def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]


def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch


def evaluate(model, dataset):
    psnr_values = []
    for lr, hr in dataset:
        sr = resolve(model, lr)
        psnr_value = psnr(hr, sr)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)


# ---------------------------------------
#  Normalization
# ---------------------------------------


def normalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return (x - rgb_mean) / 127.5


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return x * 127.5 + rgb_mean


def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0


def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5


# ---------------------------------------
#  Metrics
# ---------------------------------------


def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)


# ---------------------------------------
#  See https://arxiv.org/abs/1609.05158
# ---------------------------------------


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)

    x = b = Conv2D(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)

    x = Lambda(denormalize)(x)
    return Model(x_in, x, name="edsr")


def res_block(x_in, filters, scaling):
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = Conv2D(filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')

    return x

def load_image(path):
    return np.array(Image.open(path))[:, :, :3]



model = edsr(scale=2, num_res_blocks=16)
model.load_weights('./naip_process/weights/edsr-16-x2/weights.h5')

# year = '2016'
# oak_fp = '/oak/stanford/groups/deho/building_compliance/san_jose_naip_512/'
oak_fp = args.oak_fp
year = args.year
partition = args.partition
total_images = len(os.listdir(os.path.join(oak_fp, year, 'images')))
space = total_images/4

start_im = partition*space
end_im = (partition+1)*space
if end_im > total_images:
    end_im = total_images

if not os.path.exists(os.path.join(oak_fp, year, 'imagesx2')):
    os.mkdir(os.path.join(oak_fp, year, 'imagesx2'))

for i in tqdm.tqdm(os.listdir(os.path.join(oak_fp, year, 'images'))[start_im:end_im]):
    if not os.path.exists(os.path.join(oak_fp, year, 'imagesx2', i)):
        image = np.load(os.path.join(oak_fp, year, 'images', i))

        image_superres = resolve_single(model, image[:, :, :3]).numpy()

        np.save(os.path.join(oak_fp, year, 'imagesx2', i), image_superres)