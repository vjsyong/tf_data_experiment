# import better_exceptions
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision

import tensorflow as tf

mirrored_strategy = tf.distribute.MirroredStrategy()
# mixed_precision.set_global_policy('mixed_float16')


def age_mae(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, 101, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, 101, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_age - pred_age))
    return mae


def get_model(model_name="EfficientNetB0", feature_extractor_trainable=True):
    
    
    with mirrored_strategy.scope():
        base_model = None

        if model_name == 'EfficientNetB0':
            base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling="avg")
        elif model_name == 'EfficientNetB3':
            base_model = EfficientNetB3(include_top=False, weights='imagenet', input_shape=(300, 300, 3), pooling="avg")
        
        if not feature_extractor_trainable:
            print("freezing feature extractor")
            base_model.trainable = False
            
        dense1 = Dense(units=256, activation="relu", name="dense_1", kernel_regularizer='l2', kernel_initializer="he_normal")(base_model.output)
        dropout1 = Dropout(0.2)(dense1)

        dense2 = Dense(units=256, activation="relu", name="dense_2", kernel_regularizer='l2', kernel_initializer="he_normal")(dropout1)
        dropout2 = Dropout(0.2)(dense2)

        prediction = Dense(units=101, activation="softmax",
                        name="pred_age")(dropout2)

        # prediction = Dense(units=101, kernel_initializer="he_normal", use_bias=False, activation="softmax",
        #                 name="pred_age")(base_model.output)

        model = Model(inputs=base_model.input, outputs=prediction)

        return model

def unfreeze_model(model, layer_count=20, freeze_batchnorm=True):
    # We unfreeze the top n layers while leaving BatchNorm layers frozen
    print(type(model))
    for layer in model.layers[-layer_count:]:
    # for i, layer in enumerate(model.layers):
        
        if not isinstance(layer, BatchNormalization) or freeze_batchnorm == False:
            print(layer.name)
            layer.trainable = True


def main():
    model = get_model("EfficientNetB0")

if __name__ == '__main__':
    main()
