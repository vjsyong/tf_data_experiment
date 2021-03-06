import tensorflow as tf
import pathlib
import pandas as pd
from .im_tools import load_image_and_labels, image_augmentations, image_cutout

image_size = 224

def load_augment_batch_dataset(batch_size, im_size=224, split_ratio=0.7, dataset="wiki"):
    # Arguments: dataset = {"imdb" or "wiki"}

    global image_size
    image_size = im_size

    ds_dir = pathlib.Path(f"../training_data/imdb_wiki/{dataset}_crop")
    df = pd.read_csv(ds_dir / f'{dataset}.csv')

    paths = df['full_path'] = df['full_path'].str.replace(r'[\[\]\']', '').apply(lambda x: f"{ds_dir}/{x}") # Strip [, ], and ' characters
    
    ds_len = df.shape[0]

    ages  = []
    for path in paths:
        tokens = path.split("_")
        dob = tokens[-2].split("-")[0]
        picture_date = tokens[-1].split(".")[0]
        age = int(picture_date) - int(dob)
        ages.append(age)

    # Disable auto sharding
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    ds_path_labels = tf.data.Dataset.from_tensor_slices((paths, ages)).with_options(options)
    train_size = int(split_ratio * ds_len)
    train_ds = ds_path_labels.take(train_size).shuffle(24).cache()
    test_ds = ds_path_labels.skip(train_size).shuffle(24).cache()

    train_steps_per_epoch = train_size // batch_size
    test_steps_per_epoch = (ds_len-train_size) // batch_size

    # train_ds = train_ds.interleave(
    #     lambda self, _: train_ds.map(load_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE).map(image_augmentations, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE),
    #     num_parallel_calls=tf.data.AUTOTUNE
    # )

    # test_ds = test_ds.interleave(
    #     lambda self, _: test_ds.map(load_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE),
    #     num_parallel_calls=tf.data.AUTOTUNE
    # )

    # train_ds = train_ds.map(load_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE).cache().map(image_augmentations, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    # train_ds = train_ds.map(load_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE).cache().map(image_augmentations, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    train_ds = train_ds.map(load_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE).map(image_augmentations, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).map(image_cutout, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(load_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds, train_steps_per_epoch, test_steps_per_epoch





