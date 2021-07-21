import tensorflow as tf
import tensorflow_addons as tfa
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

np.set_printoptions(precision=4)

UTKFace_dir = pathlib.Path("../UTKFace")
UTKFace_len = 0

AR_dir = pathlib.Path("../appa-real")

CL_dir = pathlib.Path("../ChaLearn_aligned")

for _ in UTKFace_dir.glob("*"):
    UTKFace_len += 1

image_size = 224

UTK_list_ds = tf.data.Dataset.list_files(str(UTKFace_dir/'*'))
appa_real_ds = tf.data.Dataset.list_files(str(AR_dir/'train/*.jpg'))

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

CL_df = pd.read_csv(CL_dir / 'train_gt_aligned.csv')
CL_df['file_name'] = CL_df['file_name'].apply(lambda x: f"{CL_dir}/{x}")
CL_len = CL_df.shape[0]

CL_path_labels = tf.data.Dataset.from_tensor_slices((CL_df.file_name, CL_df.apparent_age_avg)).shuffle(CL_len)

def load_appa_real(image_path, label):
  label = tf.math.round(label)
  label = tf.clip_by_value(label, 0, 100)
  label = tf.cast(label, dtype=tf.int32)
  label = tf.one_hot(label, 101)
  label = tf.cast(label, dtype=tf.uint8)
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [224, 224])
  image = tf.cast(image, dtype=tf.uint8)
  return image, label

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

def image_augmentations(image, label):
  # Image property augments
  image = tf.image.random_contrast(image, 0.7, 1.3)
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_saturation(image, 0.7, 1.3)
  image = tf.image.random_hue(image, 0.05)
  image = tf.image.random_brightness(image, 0.3)
  
  # Image spacial augments
  image = tfa.image.rotate(image, tf.random.normal((), 0, 0.04, tf.float32, seed=0), fill_mode='nearest')
  image = tfa.image.shear_x(image, tf.random.normal((), 0, 0.04, tf.float32, seed=0), 0)

  # Pad empty batch to work with random cutout
  image = tf.expand_dims(image, axis=0)
  image = tfa.image.random_cutout(image, (48, 48), constant_values = 0)  
  image = tf.squeeze(image)

  return image, label

def get_datasets_utkface(batch_size=32, split_ratio=0.7):
    # images_ds = list_ds.map(process_dataset, num_parallel_calls=tf.data.AUTOTUNE).shuffle(UTKFace_len)
    train_size = int(split_ratio * UTKFace_len)
    print("train_size is ", train_size)
    lds = UTK_list_ds.shuffle(train_size)

    train_ds = lds.take(train_size)
    test_ds = lds.skip(train_size)

    train_ds = train_ds.map(load_utkface, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(24).map(image_augmentations, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(load_utkface, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).cache()
    return train_ds, test_ds

def get_datasets_appa_real(batch_size):
    # train_ds = AR_path_labels_train.map(load_appa_real, num_parallel_calls=tf.data.AUTOTUNE).map(image_augmentations, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    train_ds = AR_path_labels_train.map(load_appa_real, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(24).map(image_augmentations, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    test_ds = AR_path_labels_val.map(load_appa_real, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).cache()
    return train_ds, test_ds

def get_datasets_chalearn(batch_size):
    train_ds = CL_path_labels.map(load_appa_real, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(24).map(image_augmentations, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return train_ds

def steps_per_epoch_utkface(batch_size=32):
    return UTKFace_len // batch_size

def steps_per_epoch_appa_real(batch_size=32):
    return AR_len_train // batch_size





def show(image, label):
#   image = 255 * image
  plt.figure()
  plt.imshow(image)
  index = tf.argmax(label, axis=0)
  plt.title(index.numpy())
  plt.axis('off')
  plt.show()

def construct_mosaic(batch, batch_num):

  fig = plt.figure(figsize=(4., 4.))
  grid = ImageGrid(fig, 111,  # similar to subplot(111)
                  nrows_ncols=(16, 16),  # creates 2x2 grid of axes
                  # axes_pad=0.1,  # pad between axes in inch.
                  )
  
  for ax, (image, label) in zip(grid, batch):
    ax.axis('off')
    ax.imshow(image)
  plt.savefig(f'./dataset_mosaic/train/16x16_chalearn_fam_{batch_num}.png', bbox_inches='tight', dpi=600)
  # plt.show()
      
def main():
    # train, test = get_datasets_appa_real(32)
    train = get_datasets_chalearn(32)

    for i in range(1):
      construct_mosaic(train.take(8).unbatch(), i)

    # for image, label in train.take(2).unbatch():
    #     # print(batch)
    #     # for image, label in batch.unbatch():
    #       show(image, label)

if __name__ == "__main__":
    main()


