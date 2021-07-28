import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, Adam
import dataloader
import os

from model import get_model, age_mae, unfreeze_model


def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--utk_dir", type=str, default=None,
    #                     help="path to the UTK face dataset")
    # parser.add_argument("--afad_dir", type=str, default=None,
    #                     help="path to the AFAD face dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="checkpoint dir")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=30,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument("--opt", type=str, default="sgd",
                        help="optimizer name; 'sgd' or 'adam'")
    parser.add_argument("--model_name", type=str, default="EfficientNetB0",
                        help="model name:'EfficientNetB[0/3]'")
    args = parser.parse_args()
    return args


class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.2
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.04
        return self.initial_lr * 0.008


def get_optimizer(opt_name, lr):
    if opt_name == "sgd":
        return SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    elif opt_name == "adam":
        return Adam(learning_rate=lr)
    else:
        raise ValueError("optimizer name should be 'sgd' or 'adam'")


def main():
    args = get_args()
    # appa_dir = args.appa_dir
    # utk_dir = args.utk_dir
    # afad_dir = args.afad_dir
    model_name = args.model_name
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    opt_name = args.opt

    if model_name == 'EfficientNetB0':
        image_size = 224
    elif model_name == 'EfficientNetB3':
        image_size = 300

    # Data Pipeline
    dl = dataloader.Dataloader(batch_size)
    # train_ds, val_ds = dl.get_datasets_wiki()
    train_ds, val_ds, train_steps, val_steps = dl.get_datasets_pretraining()
    
    os.system("rm -rf ./logs/")

    # Training Pipeline
    model = get_model(model_name=model_name, feature_extractor_trainable=False)
    unfreeze_model(model, layer_count=100)
    opt = get_optimizer(opt_name, lr)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=[age_mae])
    # model.summary()
    output_dir = Path(__file__).resolve().parent.joinpath(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
                TensorBoard(log_dir="./logs", histogram_freq=1),
                # LearningRateScheduler(schedule=Schedule(nb_epochs, initial_lr=lr)),
                ReduceLROnPlateau(monitor='val_age_mae', factor=0.2,
                              patience=6, min_lr=0.0001, verbose=1),
                ModelCheckpoint(str(output_dir) + f"/weights/dense256-pretrain2-{model_name}/{batch_size}-{lr}-{opt_name}/" + "{epoch:03d}-{val_loss:.3f}-{val_age_mae:.3f}.hdf5",
                                 monitor="val_age_mae",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="min")
                 ]

    hist = model.fit(x=train_ds,
                               epochs=nb_epochs,
                               validation_data=val_ds,
                               steps_per_epoch=train_steps,
                               validation_steps=val_steps,
                               verbose=1,
                               callbacks=callbacks)

    np.savez(str(output_dir.joinpath("history.npz")), history=hist.history)


if __name__ == '__main__':
    main()
