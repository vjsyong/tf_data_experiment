import tensorflow as tf
import pathlib
import pandas as pd
from .im_tools import load_image_and_labels, image_augmentations

image_size = 224

def load_augment_batch_dataset(batch_size, im_size=224, split_ratio=0.7, dataset="wiki"):
    # Arguments: dataset = {"imdb" or "wiki"}

    global image_size
    image_size = im_size

    wiki_dir = pathlib.Path(f"../imdb_wiki/{dataset}_crop")
    df = pd.read_csv(wiki_dir / f'{dataset}.csv')

    paths = df['full_path'] = df['full_path'].str.replace(r'[\[\]\']', '').apply(lambda x: f"{wiki_dir}/{x}") # Strip [, ], and ' characters
    
    wiki_len = df.shape[0]

    ages  = []
    for path in paths:
        tokens = path.split("_")
        dob = tokens[-2].split("-")[0]
        picture_date = tokens[-1].split(".")[0]
        age = int(picture_date) - int(dob)
        ages.append(age)

    wiki_path_labels = tf.data.Dataset.from_tensor_slices((paths, ages))
    

    train_size = int(split_ratio * wiki_len)
    train_ds = wiki_path_labels.take(train_size)
    test_ds = wiki_path_labels.skip(train_size)

    train_ds = train_ds.interleave(lambda self, x: train_ds.map(load_image_and_labels).cache().map(image_augmentations, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    test_ds = test_ds.interleave(lambda self, x: test_ds.map(load_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).cache())


    # train_ds = train_ds.map(load_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE).cache().map(image_augmentations, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    # train_ds = train_ds.map(load_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE).cache().map(image_augmentations, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    # test_ds = test_ds.map(load_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).cache()

    return train_ds, test_ds





