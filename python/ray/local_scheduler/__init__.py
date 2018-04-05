from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ObjectID(object):
  def __init__(self, x):
    self.value = x

  def __call__(self, binary_object_id):
    return binary_object_id
