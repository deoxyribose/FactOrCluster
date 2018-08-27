from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from contextlib import contextmanager

import functools
import threading

import tensorflow_probability as tfp
ed = tfp.edward2
interception = ed.interception

@contextmanager
def tape():
  """Context manager for recording interceptable executions onto a tape.
  Similar to `tf.GradientTape`, operations are recorded if they are executed
  within this context manager. In addition, the operation must be registered
  (wrapped) as `ed.interceptable`.
  Yields:
    tape: OrderedDict where operations are recorded in sequence. Keys are
      the `name` keyword argument to the operation (typically, a random
      variable's `name`) and values are the corresponding output of the
      operation. If the operation has no name, it is not recorded.
  #### Examples
  ```python
  from tensorflow_probability import edward2 as ed
  def probabilistic_matrix_factorization():
    users = ed.Normal(0., 1., sample_shape=[5000, 128], name="users")
    items = ed.Normal(0., 1., sample_shape=[7500, 128], name="items")
    ratings = ed.Normal(loc=tf.matmul(users, items, transpose_b=True),
                        scale=0.1,
                        name="ratings")
    return ratings
  with ed.tape() as model_tape:
    ratings = probabilistic_matrix_factorization()
  assert model_tape["users"].shape == (5000, 128)
  assert model_tape["items"].shape == (7500, 128)
  assert model_tape["ratings"] == ratings
  ```
  """
  tape_data = collections.OrderedDict({})

  def record(f, *args, **kwargs):
    """Records execution to a tape."""
    name = kwargs.get("name")
    output = f(*args, **kwargs) #modified from output = interceptable(f)(*args, **kwargs) 
    if name:
      tape_data[name] = output
    return output

  with interception(record):
    yield tape_data