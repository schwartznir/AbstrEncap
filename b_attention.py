import tensorflow as tf
from keras.layers import Layer
import keras.backend as kb


class BahdanauAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(BahdanauAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name='W',
                                 shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                 initializer='uniform',
                                 trainable=True)
        self.u = self.add_weight(name='U',
                                 shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                 initializer='uniform',
                                 trainable=True)
        self.v = self.add_weight(name='V',
                                 shape=tf.TensorShape((input_shape[0][2], 1)),
                                 initializer='uniform',
                                 trainable=True)

        super(BahdanauAttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        def context_step(inputs):
            c_i = kb.sum(enc_out * kb.expand_dims(inputs, -1), axis=1)
            return c_i, [c_i]

        def energy_step(inputs, states):
            batch_size = enc_out.shape[0]
            enc_seq_len, enc_hidden = enc_out.shape[1], enc_out.shape[2]
            reshaped_enc_out = kb.reshape(enc_out, (batch_size * enc_seq_len, enc_hidden))
            inp_w_s = kb.reshape(kb.dot(reshaped_enc_out, self.w), (batch_size, enc_seq_len, enc_hidden))
            inp_u_h = kb.expand_dims(kb.dot(inputs, self.u), 1)
            reshaped_sum_inps = kb.tanh(kb.reshape(inp_w_s + inp_u_h, (batch_size * enc_seq_len, enc_hidden)))
            e_j = kb.reshape(kb.dot(reshaped_sum_inps, self.v), (batch_size, enc_seq_len))
            e_j = kb.softmax(e_j)

            return e_j, [e_j]

        enc_out, dec_out = inputs
        # Initialize a RNN with temp context and e states
        tmp_state_c = kb.zeros(shape=(enc_out.shape[0], enc_out.shape[-1]))
        tmp_state_e = kb.zeros(shape=(enc_out.shape[0], enc_out.shape[1]))

        last_out, e_outputs, _ = kb.rnn(energy_step, dec_out, [tmp_state_e],)
        last_out, c_outputs, _ = kb.rnn(context_step, e_outputs, [tmp_state_c],)

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        # The layer should return its output and states
        return [tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
                tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))]
