import dataloader_modules.load_appa_real as appa_real_loader
import dataloader_modules.load_utkface as utk_loader
import dataloader_modules.load_chalearn as chalearn_loader
import dataloader_modules.load_imdb_wiki as imdb_wiki_loader
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# This class defines a Dataloader object that consolidates different
# data pipeline modules into a single source

class Dataloader():

    def __init__(self, batch_size=32):
        self.batch_size = batch_size

    def get_datasets_appa_real(self): 
        #appa_dataset already pre-splits datset, hence no option to split
        train_ds, val_ds, train_steps, val_steps = appa_real_loader.load_augment_batch_dataset(self.batch_size)
        ds_name = "appa_real"
        return train_ds, val_ds, train_steps, val_steps, ds_name
    
    def get_datasets_utkface(self, split_ratio=0.7):
        train_ds, val_ds, train_steps, val_steps = utk_loader.load_augment_batch_dataset(self.batch_size, split_ratio=split_ratio)
        ds_name = "utkface"
        return train_ds, val_ds, train_steps, val_steps, ds_name

    def get_datasets_chalearn(self, split_ratio=0.7):
        train_ds, val_ds = chalearn_loader.load_augment_batch_dataset(self.batch_size, split_ratio=split_ratio)
        ds_name = "chalearn"
        return train_ds, val_ds, ds_name

    def get_datasets_wiki(self, split_ratio=0.7):
        train_ds, val_ds, train_steps, val_steps = imdb_wiki_loader.load_augment_batch_dataset(self.batch_size, split_ratio=split_ratio, dataset="wiki")
        ds_name = "wiki"
        return train_ds, val_ds, train_steps, val_steps, ds_name

    def get_datasets_imdb(self, split_ratio=0.7):
        train_ds, val_ds, train_steps, val_steps = imdb_wiki_loader.load_augment_batch_dataset(self.batch_size, split_ratio=split_ratio, dataset="imdb")
        ds_name = "imdb"
        return train_ds, val_ds, train_steps, val_steps, ds_name

    # Create IMDB-WIKI pretraining dataset
    def get_datasets_pretraining(self):
        # Use all files in both datasets to create a pretraining dataset
        train_ds_imdb, val_ds_imdb, tsi, vsi, dsni = self.get_datasets_imdb(split_ratio=0.9)
        train_ds_wiki, val_ds_wiki, tsw, vsw, dsnw = self.get_datasets_wiki(split_ratio=0.9)
        train_steps = tsi + tsw
        val_steps = vsi + vsw
        train_ds = train_ds_imdb.concatenate(train_ds_wiki)
        val_ds = val_ds_imdb.concatenate(val_ds_wiki)
        ds_name = f"{dsni}_{dsnw}"

        return train_ds, val_ds, train_steps, val_steps, ds_name
    
    def construct_mosaic(self, batch, batch_num):
        fig = plt.figure(figsize=(4., 4.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                        # axes_pad=0.1,  # pad between axes in inch.
                        )
        
        for ax, (image, label) in zip(grid, batch):
            ax.axis('off')
            ax.imshow(image)
        plt.savefig(f'./dataset_mosaic/augment_test/{batch_num}.png', bbox_inches='tight', dpi=600)
        # plt.show()
    
    def show_sample(self, image, label):
        plt.figure()
        plt.imshow(image)
        index = tf.argmax(label, axis=0)
        plt.title(index.numpy())
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    dl = Dataloader()
    train_ds, val_ds, train_steps, val_steps, ds_name = dl.get_datasets_appa_real()
    for i in range(1):
      dl.construct_mosaic(train_ds.take(2).unbatch(), i)
    print(train_ds)
    # for batch_num, batch in enumerate(train_ds.take(2)):
    #     dl.construct_mosaic(batch, batch_num)
    #     # dl.show_sample(image, label)