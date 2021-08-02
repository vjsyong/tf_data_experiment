import tensorflow as tf
import pathlib
import pandas as pd
from .im_tools import image_augmentations, load_image_and_labels, image_cutout

image_size = 224

def load_augment_batch_dataset(batch_size, im_size=224):
    global image_size
    image_size = im_size

    AR_dir = pathlib.Path("../training_data/appa-real")

    AR_df_train = pd.read_csv(AR_dir/'gt_avg_train.csv')
    AR_df_train = AR_df_train.drop(['num_ratings', 'apparent_age_std', 'real_age'], axis=1)
    AR_df_train['file_name'] = AR_df_train['file_name'].apply(lambda x: f"{AR_dir}/train/{x.split('.')[0]}.jpg_face.jpg")
    AR_len_train = AR_df_train.shape[0]

    AR_df_val = pd.read_csv(AR_dir/'gt_avg_valid.csv')
    AR_df_val = AR_df_val.drop(['num_ratings', 'apparent_age_std', 'real_age'], axis=1)
    AR_df_val['file_name'] = AR_df_val['file_name'].apply(lambda x: f"{AR_dir}/valid/{x.split('.')[0]}.jpg_face.jpg")
    AR_len_val = AR_df_val.shape[0]

    AR_path_labels_train = tf.data.Dataset.from_tensor_slices((AR_df_train.file_name, AR_df_train.apparent_age_avg))
    AR_path_labels_val = tf.data.Dataset.from_tensor_slices((AR_df_val.file_name, AR_df_val.apparent_age_avg)).shuffle(AR_len_val, reshuffle_each_iteration=True)
    
    # train_ds = AR_path_labels_train.interleave(
    #     lambda self, _: AR_path_labels_train.map(load_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE).shuffle(128).map(image_augmentations, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).map(image_cutout, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE),
    #     num_parallel_calls=tf.data.AUTOTUNE
    # )

    train_ds = AR_path_labels_train.map(load_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE).shuffle(100, reshuffle_each_iteration=True).map(image_augmentations, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).map(image_cutout, num_parallel_calls=tf.data.AUTOTUNE).repeat()
    test_ds = AR_path_labels_val.map(load_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).cache()
    return train_ds, test_ds, AR_len_train//batch_size, AR_len_val//batch_size





