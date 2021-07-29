import tensorflow as tf
import pathlib, os
from .im_tools import image_augmentations, image_cutout

image_size = 224

def load_utkface(file_path):
  label = tf.strings.split(tf.strings.split(file_path, "/")[-1], "_")[0]
  label = tf.strings.to_number(label, out_type=tf.dtypes.int32)
  label = tf.clip_by_value(label, 0, 100)
  label = tf.one_hot(label, 101)
  label = tf.cast(label, dtype=tf.uint8)
  image = tf.io.read_file(file_path)
  image = tf.image.decode_jpeg(image)
  image = tf.image.resize(image, [image_size, image_size])
  image = tf.cast(image, dtype=tf.uint8)
  return image, label

def load_augment_batch_dataset(batch_size, im_size=224, split_ratio=0.7):
    global image_size
    image_size = im_size
    
    # Set parameters
    UTKFace_dir = pathlib.Path("../UTKFace")
    UTKFace_len = len([name for name in os.listdir(UTKFace_dir) if os.path.isfile(os.path.join(UTKFace_dir, name))])

    # Retrieve list of files from UTKFace directory
    UTK_list_ds = tf.data.Dataset.list_files(str(UTKFace_dir/'*'))

    # Disable auto sharding
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    # Split training and validation according to split_ratio
    UTK_list_ds = UTK_list_ds.shuffle(UTKFace_len).with_options(options)
    train_size = int(split_ratio * UTKFace_len)
    train_ds = UTK_list_ds.take(train_size).cache()
    test_ds = UTK_list_ds.skip(train_size).cache()

    train_steps_per_epoch = train_size // batch_size
    test_steps_per_epoch = (UTKFace_len-train_size) // batch_size

    # train_ds = train_ds.interleave(
    #     lambda _: train_ds.map(load_utkface, num_parallel_calls=tf.data.AUTOTUNE).cache().map(image_augmentations, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).map(image_cutout, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE),
    #     num_parallel_calls=tf.data.AUTOTUNE
    # )

    # test_ds = test_ds.interleave(
    #     lambda _: test_ds.map(load_utkface, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE),
    #     num_parallel_calls=tf.data.AUTOTUNE
    # )

    train_ds = train_ds.map(load_utkface, num_parallel_calls=tf.data.AUTOTUNE).cache().map(image_augmentations, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).map(image_cutout, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(load_utkface, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).cache()
    
    return train_ds, test_ds, train_steps_per_epoch, test_steps_per_epoch

