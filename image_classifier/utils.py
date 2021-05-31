# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class HyperParams:
    """
    하이퍼파라미터 정의!
    상속하여 사용!
    """

    batch_size = 32
    image_size = 224
    learning_rate = 0.001
    n_class = 10
    train_size = 0.8
