# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Utils for operations unit tests
"""

import numpy as np
import pytest

from cntk.tests.test_utils import *

from ...context import get_new_context
from ...reader import *
from .. import constant, input_numpy, sparse_input_numpy


# Keeping things short
I = input_numpy
SI = sparse_input_numpy

def batch_dense_to_sparse(batch, dynamic_axis=''):
    '''
    Helper test function that converts a batch of dense tensors into sparse
    representation that can be consumed by :func:`cntk.ops.sparse_input_numpy`.

    Args:
        batch (list): list of samples. If `dynamic_axis` is given, samples are sequences of tensors. Otherwise, they are simple tensors.
        dynamic_axis (str or :func:`cntk.ops.dynamic_axis` instance): the dynamic axis

    Returns:
        (indices, values, shape)
    '''

    batch_indices = []
    batch_values = []
    tensor_shape = []

    shapes_in_tensor = set()

    for tensor in batch:
        if isinstance(tensor, list):
            tensor = np.asarray(tensor)

        if dynamic_axis:
            # collecting the shapes ignoring the dynamic axis
            shapes_in_tensor.add(tensor.shape[1:])
        else:
            shapes_in_tensor.add(tensor.shape)

        if len(shapes_in_tensor) != 1:
            raise ValueError('except for the sequence dimensions all shapes ' +
                             'should be the same - instead we %s' %
                             (", ".join(str(s) for s in shapes_in_tensor)))

        t_indices = range(tensor.size)
        t_values = tensor.ravel(order='F')

        batch_indices.append(list(t_indices))
        batch_values.append(list(t_values))

    return batch_indices, batch_values, shapes_in_tensor.pop()

def test_batch_dense_to_sparse():
    i, v, s = batch_dense_to_sparse(
            [
                [[1,2,3], [4,5,6]],
                [[10,20,30], [40,50,60]],
            ])
    assert i == [
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5],
            ]
    assert v == [
            [1,4,2,5,3,6],
            [10,40,20,50,30,60]
            ]
    assert s == (2,3)
    
    i, v, s = batch_dense_to_sparse([[1]])
    assert i == [[0]]
    assert v == [[1]]
    assert s == (1,)
