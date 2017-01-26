"""
Common attention layers reimplemented as Keras layers

Reference:
A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task (Chen et al., 2016)
https://cs.stanford.edu/people/danqi/papers/acl2016.pdf
https://github.com/danqi/rc-cnn-dailymail/blob/master/code/nn_layers.py#L102


"""

import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras.initializations import uniform


class BilinearAttentionLayer(Layer):

    def __init__(self, init=None, **kwargs):
        """
        :param init: weight initialization function. Can be the name of an existing function (str),
                or a Theano function (see: https://keras.io/initializations/).
        :param kwargs:
        """

        # loading weight initializations if not given
        # uniform (-0.01,0.01) are the values given in Chen et al. 2016 paper
        if init is None:
            def my_init(shape, name=None):
                value = uniform(shape, scale=0.01)
                return K.variable(value, name=name)

            self.init = my_init
        else:
            self.init = init

        super(BilinearAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        :param input_shape:
                input_shape:[0] : batch * len * h
                input_shape:[1] : batch * h
        :return:
        """

        # Create a trainable weight variable for this layer.
        # W = h x h
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer=self.init,
                                 trainable=True)
        super(BilinearAttentionLayer, self).build()  # Be sure to call this somewhere!

    def call(self, inputs,  mask=None):
        """
        :param inputs:
                shape inputs[0] : batch * l * h   referred to in the paper as p
                shape inputs[1] : batch  * h         referred to in the paper as q
        :param mask:
        :return:
        """
        if len(inputs) != 2:
            raise TypeError('Attention layers must be called on with a list of 2 tensors ' + str(inputs))

        M = K.dot(inputs[1], self.W).dimshuffle(0, 'x', 1)     # M shape = batch x 1 x h
        alpha = K.softmax(K.sum(inputs[0] * M, axis=2))
        output = K.sum(inputs[0] * alpha.dimshuffle(0, 1, 'x'), axis=1)

        return output   # size Batch * h

    def get_output_shape_for(self, input_shape):

        return input_shape[0], input_shape[-1]

class DotproductAttentionLayer(Layer):
    def __init__(self, **kwargs):

        super(DotproductAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        :param input_shape:
                input_shape:[0] : batch * len * h
                input_shape:[1] : batch * h
        :return:
        """
        super(DotproductAttentionLayer, self).build()  # Be sure to call this somewhere!

    def call(self, inputs, mask=None):
        """
        :param inputs:
                shape inputs[0] : batch * l * h   referred to in the paper as p
                shape inputs[1] : batch  * h         referred to in the paper as q
        :param mask:
        :return:
        """
        if len(inputs) != 2:
            raise TypeError('Attention layers must be called on with a list of 2 tensors ' + str(inputs))

        alpha = K.softmax(K.sum(input[0] * inputs[1].dimshuffle(0, 'x', 1), axis=2))
        output = K.sum(inputs[0] * alpha.dimshuffle(0, 1, 'x'), axis=1)

        return output  # size Batch * h

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]

