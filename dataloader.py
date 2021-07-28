import dataloader_modules.load_appa_real as appa_real_loader
import dataloader_modules.load_utkface as utk_loader
import dataloader_modules.load_chalearn as chalearn_loader
import dataloader_modules.load_imdb_wiki as imdb_wiki_loader
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
    
    def get_datasets_utkface(self, split_ratio=0.7):
        train_ds, val_ds, train_steps, val_steps = utk_loader.load_augment_batch_dataset(self.batch_size, split_ratio=split_ratio)
        return train_ds, val_ds, train_steps, val_steps

    def get_datasets_chalearn(self, split_ratio=0.7):
        train_ds, val_ds = chalearn_loader.load_augment_batch_dataset(self.batch_size, split_ratio=split_ratio)
        return train_ds, val_ds

    def get_datasets_wiki(self, split_ratio=0.7):
        train_ds, val_ds, train_steps, val_steps = imdb_wiki_loader.load_augment_batch_dataset(self.batch_size, split_ratio=split_ratio, dataset="wiki")
        return train_ds, val_ds, train_steps, val_steps

    def get_datasets_imdb(self, split_ratio=0.7):
        train_ds, val_ds, train_steps, val_steps = imdb_wiki_loader.load_augment_batch_dataset(self.batch_size, split_ratio=split_ratio, dataset="imdb")
        return train_ds, val_ds, train_steps, val_steps

    # Create IMDB-WIKI pretraining dataset
    def get_datasets_pretraining(self):
        # Use all files in both datasets to create a pretraining dataset
        train_ds_imdb, val_ds_imdb, tsi, vsi = self.get_datasets_imdb(split_ratio=0.8)
        train_ds_wiki, val_ds_wiki, tsw, vsw = self.get_datasets_wiki(split_ratio=0.8)
        train_steps = tsi + tsw
        val_steps = vsi + vsw
        train_ds = train_ds_imdb.concatenate(train_ds_wiki)
        val_ds = val_ds_imdb.concatenate(val_ds_wiki)
        return train_ds, val_ds, train_steps, val_steps
    
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
    for image, label in train_ds.take(1).unbatch():
        dl.show_sample(image, label)