import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from utils import HyperParams


class InvalidModelError(Exception):
    pass


class ModelSelect:
    """
    Model을 선택하는 class
    """

    def __init__(self, model_name):
        self.model_name = model_name

    def _model_select(self):
        """
        사용가능 모델 불러오기
        EfficientNetB0 or EfficientNetB1 or EfficientNetB2
        """
        if self.model_name == "B0" or "b0":
            model = EfficientNetB0(include_top=False, pooling="avg")
        elif self.model_name == "B1" or "b1":
            model = EfficientNetB1(include_top=False, pooling="avg")
        elif self.model_name == "B2" or "b2":
            model = EfficientNetB2(include_top=False, pooling="avg")
        else:
            # 없는 모델 입력시 error 발생
            raise InvalidModelError("You input invalid model name.")
        return model

    def __call__(self):
        return self._model_select()


class BuildMLP(HyperParams):
    """
    BaseModel 끝단에 MLP를 Build하는 class
    """

    def __init__(self, base_model):
        self.base_model = base_model

    def build_mlp(self, image_size, layer_1, layer_1_af):
        """
        MLP를 빌드하는 함수
        """
        _input = tf.keras.layers.Input((image_size, image_size, 3))
        model = self.base_model()(_input)
        model = tf.keras.layers.Dense(layer_1, activation=layer_1_af)(model)
        model = tf.keras.layers.BatchNormalization()(model)
        model = tf.keras.layers.Dense(self.n_class)(model)
        model = tf.keras.Model(_input, model)
        return model
