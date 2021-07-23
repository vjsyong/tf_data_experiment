import tensorflow as tf
import pathlib
import pandas as pd
from .im_tools import image_augmentations

image_size = 224

def load_appa_real(image_path, label):
  label = tf.math.round(label)
  label = tf.clip_by_value(label, 0, 100)
  label = tf.cast(label, dtype=tf.int32)
  label = tf.one_hot(label, 101, dtype=tf.uint8)
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [image_size, image_size])
  image = tf.cast(image, dtype=tf.uint8)
  return image, label

def load_augment_batch_dataset(batch_size, im_size=224):
    global image_size
    image_size = im_size

    AR_dir = pathlib.Path("../appa-real")

    AR_df_train = pd.read_csv(AR_dir/'gt_avg_train_aligned.csv')
    AR_df_train = AR_df_train.drop(['num_ratings', 'apparent_age_std', 'real_age'], axis=1)
    AR_df_train['file_name'] = AR_df_train['file_name'].apply(lambda x: f"{AR_dir}/train_aligned/{x}")
    AR_len_train = AR_df_train.shape[0]

    AR_df_val = pd.read_csv(AR_dir/'gt_avg_valid_aligned.csv')
    AR_df_val = AR_df_val.drop(['num_ratings', 'apparent_age_std', 'real_age'], axis=1)
    AR_df_val['file_name'] = AR_df_val['file_name'].apply(lambda x: f"{AR_dir}/valid_aligned/{x}")
    AR_len_val = AR_df_val.shape[0]

    AR_path_labels_train = tf.data.Dataset.from_tensor_slices((AR_df_train.file_name, AR_df_train.apparent_age_avg)).shuffle(AR_len_train)
    AR_path_labels_val = tf.data.Dataset.from_tensor_slices((AR_df_val.file_name, AR_df_val.apparent_age_avg)).shuffle(AR_len_val)
    
    train_ds = AR_path_labels_train.map(load_appa_real, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(24).map(image_augmentations, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    test_ds = AR_path_labels_val.map(load_appa_real, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).cache()
    return train_ds, test_ds




