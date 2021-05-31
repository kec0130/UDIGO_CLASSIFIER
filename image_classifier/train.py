import os
import argparse
import tensorflow as tf
from dataloader import DataLoader
from model import ModelSelect, BuildMLP

# from utils import HyperParams

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tfr_path", type=str, default="./")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--n_class", type=int, default=10)
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--model_name", type=str, default="b0")
    args = parser.parse_args()

    params = {"layer_1": 512, "layer_1_af": "relu", "patience": 5}

    # data 불러오기
    data = DataLoader(args.tfr_path, args.image_size)
    train_set, validation_set, steps = data.build_dataset()

    # model 불러오기
    base_model = ModelSelect(args.model_name)
    model_instance = BuildMLP(base_model)
    model = model_instance.build_mlp(args.image_size, params)

    # model compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # callbacks
    # models directory가 없으면 생성
    save_dir = "./models/"
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=params["patience"]
    )
    mc = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_dir + "{epoch}-{val_loss:.2f}-{val_accuracy:.2f}.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    # train
    history = model.fit(
        train_set,
        epochs=9999,
        validation_data=validation_set,
        steps_per_epoch=steps,
        callbacks=[es, mc],
    )
