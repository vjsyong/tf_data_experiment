import numpy as np
import pandas as pd
import dataloader
from model import get_model, age_mae

def main(model_name, weight_file):
    # Load dataset
    dl = dataloader.Dataloader(32)
    test_ds = dl.get_datasets_appa_real_test()
    # _, test_ds, _, _, _ = dl.get_datasets_utkface(split_ratio=0)


    # Load model
    model = get_model(model_name=model_name)
    model.load_weights(weight_file)
    model.compile(optimizer="SGD", loss="categorical_crossentropy", metrics=[age_mae])

    # Evaluate model
    model.evaluate(test_ds)

    # Print results

if __name__ == '__main__':
    model_name = "EfficientNetB0"
    weight_file = "/nfs/home/seanyong/Documents/Age_Estimation/tf_data_experiment/checkpoints/weights/EfficientNetB0-dense256/128-0.001-adam/appa_real025-3.380-4.383.hdf5"
    main(model_name, weight_file)