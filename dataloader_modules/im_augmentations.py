import tensorflow as tf
import tensorflow_addons as tfa

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