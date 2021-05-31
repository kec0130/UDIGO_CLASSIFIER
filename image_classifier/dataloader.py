import math
from functools import partial
import tensorflow as tf
from utils import HyperParams


class DataLoader(HyperParams):
    """
    TFRecord를 Parsing하여 Dataset으로 만드는 class
    """

    autotune = tf.data.experimental.AUTOTUNE

    def __init__(self, tfr_path, img_size):
        self.tfr_path = tfr_path
        self.img_size = img_size

    def _parse_function(self, tfrecord_serialized, size):
        """
        Tensorflow 공식 홈페이지 참고
        """
        features = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }
        parsed_features = tf.io.parse_single_example(tfrecord_serialized, features)

        image = tf.io.decode_raw(parsed_features["image"], tf.uint8)
        image = tf.reshape(image, [256, 256, 3])
        image = tf.image.resize(image, [size, size])

        label = tf.cast(parsed_features["label"], tf.int64)
        label = tf.one_hot(label, 10)

        return image, label

    def build_dataset(self):
        """
        dataset build 함수
        """
        # dataset 불러오기
        dataset = tf.data.TFRecordDataset(self.tfr_path)
        # dataset size 정의
        dataset_size = len(list(dataset))
        train_size = int(self.train_size * dataset_size)
        val_size = int((1 - self.train_size) * dataset_size)
        # data parsing and shuffle
        dataset = dataset.map(
            partial(self._parse_function, size=self.img_size),
            num_parallel_calls=self.autotune,
        )
        dataset = dataset.shuffle(dataset_size)
        # trainset build
        train_ds = dataset.take(train_size)
        train_ds = train_ds.batch(self.batch_size)
        train_ds = train_ds.repeat().prefetch(self.autotune)
        # validationset build
        val_ds = dataset.skip(train_size)
        val_ds = val_ds.take(val_size)
        val_ds = val_ds.batch(self.batch_size)

        steps = math.floor(dataset_size / self.batch_size)

        return train_ds, val_ds, steps
