import dataloader_modules.load_appa_real as appa_real_loader
import dataloader_modules.load_utkface as utk_loader
import dataloader_modules.load_chalearn as chalearn_loader
import tensorflow as tf
import matplotlib.pyplot as plt

# This class defines a Dataloader object that consolidates different
# data pipeline modules into a single source

class Dataloader():

    def __init__(self, batch_size=32):
        self.batch_size = batch_size

    def get_datasets_appa_real(self): 
        #appa_dataset already pre-splits datset, hence no option to split
        train_ds, val_ds = appa_real_loader.load_augment_batch_dataset(self.batch_size)
        return train_ds, val_ds
    
    def get_datasets_utkface(self):
        train_ds, val_ds = utk_loader.load_augment_batch_dataset(self.batch_size, split_ratio=0.7)
        return train_ds, val_ds

    def get_datasets_chalearn(self):
        train_ds, val_ds = chalearn_loader.load_augment_batch_dataset(self.batch_size, split_ratio=0.7)
        return train_ds, val_ds

    def show_sample(self, image, label):
        plt.figure()
        plt.imshow(image)
        index = tf.argmax(label, axis=0)
        plt.title(index.numpy())
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    dl = Dataloader()
    train_ds, val_ds = dl.get_datasets_chalearn()
    print(train_ds)
    for image, label in val_ds.take(1).unbatch():
        dl.show_sample(image, label)