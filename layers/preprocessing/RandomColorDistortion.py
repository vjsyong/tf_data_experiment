import tensorflow as tf
import numpy as np

class RandomColorDistortion(tf.keras.layers.Layer):
    def __init__(self, contrast_range=[0.7, 1.3], 
                 brightness_delta=0.3, 
                 saturation_delta=[-0.7, 1.3], 
                 hue_delta=0.05, **kwargs):
        super(RandomColorDistortion, self).__init__(**kwargs)
        self.contrast_range = contrast_range
        self.brightness_delta = brightness_delta
        self.saturation_delta = saturation_delta
        self.hue_delta = hue_delta
    
    def call(self, images, training=None):
        if not training:
            return images

        images = tf.image.random_contrast(images, self.contrast_range[0], self.contrast_range[1])
        images = tf.image.random_brightness(images, self.brightness_delta)
        images = tf.image.random_saturation(images, self.saturation_delta)
        images = tf.image.random_hue(images, self.hue_delta)
        
        
        # contrast = np.random.uniform(
        #     self.contrast_range[0], self.contrast_range[1])
        # brightness = np.random.uniform(
        #     self.brightness_delta[0], self.brightness_delta[1])
        
        # images = tf.image.adjust_contrast(images, contrast)
        # images = tf.image.adjust_brightness(images, brightness)
        images = tf.clip_by_value(images, 0, 1)
        return images

