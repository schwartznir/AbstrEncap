from keras.layers import Layer
import keras.backend as kb


class Attention_Layer(Layer):
    def __init__(self, **kwargs):
        super(Attention_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name="weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention_Layer, self).build(input_shape)

    def get_config(self):
        return super(Attention_Layer, self).get_config()

    def output_shape(self, input_shape):
        return input_shape[0], input_shape[1]

    def call(self, x):
        e_t = kb.squeeze(kb.tanh(kb.dot(x, self.w) + self.b), axis=-1)
        a_t = kb.softmax(e_t)
        a_t = kb.expand_dims(a_t, axis=-1)
        return kb.sum(x * a_t, axis=1)
