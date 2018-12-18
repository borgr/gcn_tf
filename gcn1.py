import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np

from tensorflow import expand_dims
from tensorflow import tile
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
# from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.util import tf_decorator
from tf_export import tf_export
from tensorflow.python.layers import base
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import nn
# from tensorflow.python import math_ops
# from tensorflow.contrib.eager import context


@tf_export('layers.GCN')
class GCN(base.Layer):
    """Densely-connected layer class.
    This layer implements the operation:
    `outputs = activation(inputs * kernel + bias)`
    Where `activation` is the activation function passed as the `activation`
    argument (if not `None`), `kernel` is a weights matrix created by the layer,
    and `bias` is a bias vector created by the layer
    (only if `use_bias` is `True`).
    Arguments:
      units: Integer or Long, dimensionality of the output space.
      activation: Activation function (callable). Set it to None to maintain a
        linear activation.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: Initializer function for the weight matrix.
        If `None` (default), weights are initialized using the default
        initializer used by `tf.get_variable`.
      bias_initializer: Initializer function for the bias.
      kernel_regularizer: Regularizer function for the weight matrix.
      bias_regularizer: Regularizer function for the bias.
      activity_regularizer: Regularizer function for the output.
      kernel_constraint: An optional projection function to be applied to the
          kernel after being updated by an `Optimizer` (e.g. used to implement
          norm constraints or value constraints for layer weights). The function
          must take as input the unprojected variable and must return the
          projected variable (which must have the same shape). Constraints are
          not safe to use when doing asynchronous distributed training.
      bias_constraint: An optional projection function to be applied to the
          bias after being updated by an `Optimizer`.
      trainable: Boolean, if `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such cases.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Properties:
      units: Python integer, dimensionality of the output space.
      edges_label_num: Python integer, dimensionality of the edge label space.
      bias_label_num: Python integer, dimensionality of the bias label space.
      activation: Activation function (callable).
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: Initializer instance (or name) for the kernel matrix.
      bias_initializer: Initializer instance (or name) for the bias.
      kernel_regularizer: Regularizer instance for the kernel matrix (callable)
      bias_regularizer: Regularizer instance for the bias (callable).
      activity_regularizer: Regularizer instance for the output (callable)
      kernel_constraint: Constraint function for the kernel matrix.
      bias_constraint: Constraint function for the bias.
      kernel: Weight matrix (TensorFlow variable or tensor).
      bias: Bias vector, if applicable (TensorFlow variable or tensor).
    """

    def __init__(self, units,
                 activation=None,
                 gate=True,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 gate_kernel_initializer=None,
                 gate_bias_initializer=init_ops.zeros_initializer(),
                 gate_kernel_regularizer=None,
                 gate_bias_regularizer=None,
                 gate_kernel_constraint=None,
                 gate_bias_constraint=None,
                 # activity_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(GCN, self).__init__(trainable=trainable, name=name,
                                  # activity_regularizer=activity_regularizer,
                                  **kwargs)
        self.units = units
        self.gate = gate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        # self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.gate_kernel_initializer = gate_kernel_initializer
        self.gate_bias_initializer = gate_bias_initializer
        self.gate_kernel_regularizer = gate_kernel_regularizer
        self.gate_bias_regularizer = gate_bias_regularizer
        # self.gate_kernel_constraint = gate_kernel_constraint
        self.gate_bias_constraint = gate_bias_constraint
        self.input_spec = [base.InputSpec(min_ndim=2), base.InputSpec(
            min_ndim=2), base.InputSpec(min_ndim=2)]

    def build(self, input_shape):
        print("building")
        base_input_shape = tensor_shape.TensorShape(input_shape[0])
        print("base_input_shape", base_input_shape)
        if base_input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `GCN` '
                             'should be defined. Found `None`.')
        edge_shape = tensor_shape.TensorShape(input_shape[1])
        if edge_shape[-1].value is None:
            raise ValueError('The last dimension of the edges inputs to `GCN` '
                             'should be defined. Found `None`.')
        self.main_input_spec = base.InputSpec(min_ndim=2,
                                              axes={-1: base_input_shape[-1].value})
        self.kernel = self.add_variable('kernel',
                                        shape=[
                                            base_input_shape[-1].value, self.units, edge_shape[-1].value],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        # constraint=self.kernel_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        self.gate_kernel = self.add_variable('gate_kernel',
                                             shape=[
                                                 base_input_shape[-1].value, edge_shape[-1].value],
                                             initializer=self.gate_kernel_initializer,
                                             regularizer=self.gate_kernel_regularizer,
                                             # constraint=self.gate_kernel_constraint,
                                             dtype=self.dtype,
                                             trainable=True)
        self.edges_spec = base.InputSpec(
            min_ndim=4, axes={-3: base_input_shape[-1].value, -2: base_input_shape[-1].value, -1: edge_shape[-1].value})
        if self.use_bias:
            bias_shape = tensor_shape.TensorShape(input_shape[2])
            if bias_shape[-1].value is None:
                raise ValueError('The last dimension of the biases inputs to `GCN` '
                                 'should be defined. Found `None`.')
            self.bias_labels_spec = base.InputSpec(
                min_ndim=4, axes={-3: base_input_shape[-1].value, -2: base_input_shape[-1].value, -1: bias_shape[-1].value})
            self.bias = self.add_variable('bias',
                                          shape=[self.units,
                                                 bias_shape[-1].value],
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          # constraint=self.bias_constraint,
                                          dtype=self.dtype,
                                          trainable=True)
            self.gate_bias = self.add_variable('gate_bias',
                                               shape=[bias_shape[-1].value],
                                               initializer=self.gate_bias_initializer,
                                               regularizer=self.gate_bias_regularizer,
                                               # constraint=self.gate_bias_constraint,
                                               dtype=self.dtype,
                                               trainable=True)
            self.input_spec = [self.main_input_spec,
                               self.edges_spec, self.bias_labels_spec]
        else:
            self.input_spec = [self.main_input_spec, self.edges_spec]
            self.bias = None
            self.gate_bias = None
        self.built = True
        print("specs", [(spec.ndim, spec.shape) for spec in self.input_spec])

    def calculate_gates(self, inputs):
        if self.use_bias:
            bias_shape = self.bias.get_shape().as_list()
            bias_labels = ops.convert_to_tensor(inputs[2], dtype=self.dtype)
            biases = math_ops.reduce_sum(
                math_ops.multiply(self.gate_bias, bias_labels), [-1])
            print("gate bias shape", biases.get_shape().as_list())

        labeled_edges = ops.convert_to_tensor(inputs[1], dtype=self.dtype)
        edges = math_ops.reduce_sum(
            math_ops.multiply(self.kernel, labeled_edges), [-1])
        print("gate edges shape", edges.get_shape().as_list())

        gates = math_ops.sigmoid(edges + biases)
        print("gate shape", gates.get_shape().as_list())
        return gates

    def calculate_kernel(self, inputs, gates):
        x = ops.convert_to_tensor(inputs[0], dtype=self.dtype)
        shape = x.get_shape().as_list()

        # broadcast labels to allow for multiple unit outputs and repeat for
        # edges labels
        labeled_edges = ops.convert_to_tensor(inputs[1], dtype=self.dtype)
        edges_shape = labeled_edges.get_shape().as_list()
        kernel_labeled_edges = expand_dims(labeled_edges, -2)
        kerne_labeled_edges = tile(
            kernel_labeled_edges, [1] * (len(edges_shape) - 1) + [self.units, 1])

        # TODO use sparse matrices
        # kernel = expand_dims(self.kernel, 1)
        # self.kernel = logging_ops.Print(self.kernel, [array_ops.shape(self.kernel), ",",  array_ops.shape(
        #     labeled_edges)[:3], array_ops.shape(labeled_edges)[3:]], message="kernel, labeled_edges[:3],[3:] shapes:")
        # self.kernel
        gates = expand_dims(gates, -1)
        gates = expand_dims(gates, -1)
        gates = tile(
            kernel_labeled_edges, [1] * (len(edges_shape) - 1) + [self.units, 1])
        print("broadcasteing gate shape", gates.get_shape().as_list())

        graph_kernel = math_ops.multiply(math_ops.multiply(self.kernel, kernel_labeled_edges), gates)

        print("kernel shape", self.kernel.get_shape().as_list())
        print("edges label shape", edges_shape)
        print("graph kernel shape", graph_kernel.get_shape().as_list())
        print("inputs shape", shape)

        if len(shape) > 2:
            # Broadcasting is required for the inputs.
            # TODO use sparse matrices if possible
            outputs = special_math_ops.einsum(
                "bi,bnilo->bnlo", x, graph_kernel)
            outputs = math_ops.reduce_sum(outputs, [-1])
            # Reshape the output back to the original ndim of the input.
            # if not context.executing_eagerly():
            output_shape = shape[:-1] + [self.units]
            outputs.set_shape(output_shape)
        else:
            # TODO use sparse matrices
            # x = logging_ops.Print(
            #     x, [array_ops.shape(x), ",", array_ops.shape(graph_kernel)[
            #         :3], array_ops.shape(graph_kernel)[3:]],
            #     message="input, graph_kernel[:3],[3:] shapes")

            outputs = special_math_ops.einsum(
                "bi,bnilo->bnlo", x, graph_kernel)

            print("before reduction kernel*inputs shape",
                  outputs.get_shape().as_list())
            outputs = math_ops.reduce_sum(outputs, [-1])
        print("kernel*inputs shape", outputs.get_shape().as_list())
        return outputs

    def calculate_bias(self, inputs, gates):
        shape = self.bias.get_shape().as_list()
        labels = ops.convert_to_tensor(inputs[2], dtype=self.dtype)
        labels_shape = labels.get_shape().as_list()
        print("biases shapes", shape, labels_shape)
        labeled_bias = standard_ops.tensordot(labels, self.bias,
                                              [[len(labels_shape) - 1], [len(shape) - 1]])
        gates = expand_dims(gates, [-1])
        print("gating biases is heavy for some reason, perhaps using sparse matrices will solve that")
        labeled_bias = math_ops.multiply(labeled_bias, gates)
        labeled_bias = math_ops.reduce_sum(labeled_bias, [-2])
        print("labeled_bias shape", labeled_bias.get_shape().as_list())
        return labeled_bias

    def call(self, inputs,  *args, **kwargs):
        print("inputs", inputs)

        if self.gate:
            gates = self.calculate_gates(inputs)
        else:
            gates = 1
        outputs = self.calculate_kernel(inputs, gates)

        if self.use_bias:
            outputs = outputs + self.calculate_bias(inputs, gates)

        print("output shape", outputs.get_shape().as_list())

        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        # TODO make sure the edges are summed right (we sum all neighbors of v
        # and not everything v is a neighbor of)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[0][:-1].concatenate(self.units)


@tf_export('layers.gcn')
def gcn(
        inputs, units,
        activation=None,
        gate=True,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=init_ops.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        gate_kernel_initializer=None,
        gate_bias_initializer=init_ops.zeros_initializer(),
        gate_kernel_regularizer=None,
        gate_bias_regularizer=None,
        gate_kernel_constraint=None,
        gate_bias_constraint=None,
        # activity_regularizer=None,
        trainable=True,
        name=None,
        reuse=None):
    """Functional interface for the graph convolutional network.
    This layer implements the operation:
    `outputs = activation(inputs.labeled_graph_kernel + labeled_bias)`
    Where `activation` is the activation function passed as the `activation`
    argument (if not `None`), `kernel` is a weights matrix created by the layer,
    a different matrix per label,
    and `bias` is a bias vector created by the layer, a different bias per label
    (only if `use_bias` is `True`).
    Arguments:
      inputs: List of Tensor inputs.
      The inputs, the edges labels and the bias labels.
      Labels are expected in the form of neighbors X vertices X labels tensors
      with 0 or one representing the existence of a labeled edge between a vertice to its neighbor.
      units: Integer or Long, dimensionality of the output space.
      activation: Activation function (callable). Set it to None to maintain a
        linear activation.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: Initializer function for the weight matrix.
        If `None` (default), weights are initialized using the default
        initializer used by `tf.get_variable`.
      bias_initializer: Initializer function for the bias.
      kernel_regularizer: Regularizer function for the weight matrix.
      bias_regularizer: Regularizer function for the bias.
      activity_regularizer: Regularizer function for the output.
      kernel_constraint: An optional projection function to be applied to the
          kernel after being updated by an `Optimizer` (e.g. used to implement
          norm constraints or value constraints for layer weights). The function
          must take as input the unprojected variable and must return the
          projected variable (which must have the same shape). Constraints are
          not safe to use when doing asynchronous distributed training.
      bias_constraint: An optional projection function to be applied to the
          bias after being updated by an `Optimizer`.
      trainable: Boolean, if `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      name: String, the name of the layer.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      Output tensor the same shape as `inputs` except the last dimension is of
      size `units`.
    Raises:
      ValueError: if eager execution is enabled.
    """
    layer = GCN(units,
                activation=activation,
                gate=True,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                gate_kernel_initializer=None,
                gate_bias_initializer=init_ops.zeros_initializer(),
                gate_kernel_regularizer=None,
                gate_bias_regularizer=None,
                gate_kernel_constraint=None,
                gate_bias_constraint=None,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                # activity_regularizer=activity_regularizer,
                trainable=trainable,
                name=name,
                # dtype=inputs[0].dtype.base_dtype,
                _scope=name,
                _reuse=reuse
                )

    return layer.apply(inputs)
