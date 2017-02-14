"""
Common attention layers reimplemented as Keras layers

Reference:
A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task (Chen et al., 2016)
https://cs.stanford.edu/people/danqi/papers/acl2016.pdf
https://github.com/danqi/rc-cnn-dailymail/blob/master/code/nn_layers.py#L102


"""

from keras import backend as K
from keras.engine.topology import Layer
from keras.initializations import uniform
from keras.layers import Merge

from IPython.core.debugger import Pdb; debug_here = Pdb().set_trace


class BilinearAttentionLayer(Layer):

    def __init__(self, layers=None, init='uniform', node_indices=None,
                 tensor_indices=None, name=None, **kwargs):
        """
        :param init: weight initialization function. Can be the name of an existing function (str),
                or a Theano function (see: https://keras.io/initializations/).
        :param kwargs:
        """

        # layer variables
        self.init = init
        self.layers = layers

        # Layer parameters.
        self.inbound_nodes = []
        self.outbound_nodes = []
        self.constraints = {}
        self.trainable_weights = []
        self.non_trainable_weights = []
        self.supports_masking = False
        self.uses_learning_phase = False
        self.input_spec = None  # Compatible with anything.
        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(K.get_uid(prefix))
        self.name = name

        self.built = False
        self.add_inbound_node(layers)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # W = h x h
        # assume it has only one output i.e. node indices = 0
        h = input_shape[0][-1]
        self.W = self.add_weight(shape=(h, h), initializer=self.init, trainable=True)
        self.built = True

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

        return input_shape[0][0], input_shape[0][-1]

class DotproductAttentionLayer(Merge):

    def call(self, inputs, mask=None):
        """
        :param inputs:
                shape inputs[0] : batch * l * h   referred to in the paper as p
                shape inputs[1] : batch  * h         referred to in the paper as q
        :param mask:
        :return:
        """

        a = K.sum(inputs[0] * inputs[1].dimshuffle(0, 'x', 1), axis=2)
        alpha = K.softmax(a)
        output = K.sum(inputs[0] * alpha.dimshuffle(0, 1, 'x'), axis=1)

        return output  # size Batch * h

    def _arguments_validation(self, layers, mode, concat_axis, dot_axes,
                              node_indices, tensor_indices):
        return True

    def get_output_shape_for(self, input_shape):

        return input_shape[0][0], input_shape[0][-1]

