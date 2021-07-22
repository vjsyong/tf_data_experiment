import tensorflow as tf
import pathlib
import pandas as pd
from .im_augmentations import image_augmentations

image_size = 224

def load_chalearn(image_path, label):
    label = tf.math.round(label)
    label = tf.clip_by_value(label, 0, 100)
    label = tf.cast(label, dtype=tf.int32)
    label = tf.one_hot(label, 101, dtype=tf.uint8)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, dtype=tf.uint8)
    return image, label

def load_augment_batch_dataset(batch_size, im_size=224, split_ratio=0.7):
    global image_size
    image_size = im_size
    
    # Set Parameters
    CL_dir = pathlib.Path("../ChaLearn_aligned")
    CL_df = pd.read_csv(CL_dir / 'train_gt_aligned.csv')
    CL_df['file_name'] = CL_df['file_name'].apply(lambda x: f"{CL_dir}/{x}")
    CL_len = CL_df.shape[0]

    # Split training and validation according to split_ratio
    CL_path_labels = tf.data.Dataset.from_tensor_slices((CL_df.file_name, CL_df.apparent_age_avg)).shuffle(CL_len)
    train_size = int(split_ratio * CL_len)
    train_ds = CL_path_labels.take(train_size)
    test_ds = CL_path_labels.skip(train_size)
    
    # Load files, augment, batch, prefetch into dataset
    train_ds = CL_path_labels.map(load_chalearn, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(24).map(image_augmentations, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(load_chalearn, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).cache()
    
    return train_ds, test_ds
