import tensorflow as tf
import tensorflow_addons as tfa

image_size = 224

def load_image_and_labels(image_path, label):
  label = tf.math.round(label)
  label = tf.clip_by_value(label, 0, 100)
  label = tf.cast(label, dtype=tf.int32)
  label = tf.one_hot(label, 101, dtype=tf.uint8)
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [image_size, image_size])
  image = tf.cast(image, dtype=tf.uint8)
  return image, label

def image_cutout(batch, labels):
  cutout_batch = tfa.image.random_cutout(batch, (48, 48), constant_values = 0)  
  # translate_batch = tfa.image.translate(cutout_batch, tf.random.normal((2,), 0, 8, tf.float32, seed=0),fill_mode='nearest')
  return cutout_batch, labels

def image_augmentations(image, label):
  # Image property augments
  image = tf.image.random_contrast(image, 0.7, 1.3)
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_saturation(image, 0.7, 1.3)
  image = tf.image.random_hue(image, 0.05)
  image = tf.image.random_brightness(image, 0.3)
  
  # Image spacial augments
  # image = tfa.image.rotate(image, tf.random.normal((), 0, 0.1, tf.float32, seed=0), fill_mode='nearest')
  # image = tfa.image.shear_x(image, tf.random.normal((), 0, 0.04, tf.float32, seed=0), 0)
  # image = tfa.image.translate(image, tf.random.normal((2,), 0, 8, tf.float32, seed=0),fill_mode='nearest')

  # Pad empty batch to work with random cutout
  # image = tf.expand_dims(image, axis=0)
  # image = tfa.image.random_cutout(image, (48, 48), constant_values = 0)  
  # image = tf.squeeze(image)

  return image, label

