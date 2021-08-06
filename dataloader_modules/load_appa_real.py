import tensorflow as tf
import pathlib
import pandas as pd
from .im_tools import image_augmentations, load_image_and_labels, image_cutout

image_size = 224
AR_dir = pathlib.Path("../training_data/appa-real")

def read_and_load_csv(name):
    df = pd.read_csv(AR_dir/f'gt_avg_{name}.csv')
    df = df.drop(['num_ratings', 'apparent_age_std', 'real_age'], axis=1)
    df['file_name'] = df['file_name'].apply(lambda x: f"{AR_dir}/{name}/{x.split('.')[0]}.jpg_face.jpg")
    df_len = df.shape[0]
    return df, df_len

def load_augment_batch_dataset(batch_size, im_size=224):
    global image_size
    image_size = im_size

    df_train, train_len = read_and_load_csv("train")
    df_val, val_len = read_and_load_csv("valid")
    
    train_path_labels = tf.data.Dataset.from_tensor_slices((df_train.file_name, df_train.apparent_age_avg))
    val_path_labels = tf.data.Dataset.from_tensor_slices((df_val.file_name, df_val.apparent_age_avg)).shuffle(val_len, reshuffle_each_iteration=True)

    train_ds = train_path_labels.map(load_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(train_len, reshuffle_each_iteration=True).map(image_augmentations, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).map(image_cutout, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_path_labels.map(load_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).cache()
    return train_ds, val_ds, train_len//batch_size, val_len//batch_size

def load_test_dataset(batch_size, im_size=224):
    df_test, test_len = read_and_load_csv("test")
    test_path_labels = tf.data.Dataset.from_tensor_slices((df_test.file_name, df_test.apparent_age_avg))
    test_ds = test_path_labels.map(load_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return test_ds
