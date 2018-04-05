#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

from .local_scheduler import ObjectID
from .services import get_node_ip_address
from .utils import is_cython
from .worker import cleanup


def init(*args, **kwargs):
  pass


def put(x):
  obj = ObjectID(x)
  return obj


def get(obj):
  if isinstance(obj, ObjectID):
    return obj.value
  else:
    return obj


def wait(object_ids, num_returns=1, timeout=None, worker=None):
  return object_ids[:num_returns], object_ids[num_returns:]


def remote(*args, **kwargs):
  def make_remote_decorator(func_or_class):
    if inspect.isfunction(func_or_class) or is_cython(func_or_class):
      func_name = "{}.{}".format(func_or_class.__module__, func_or_class.__name__)

      def func_invoker(*args, **kwargs):
        """This is used to invoke the function."""
        raise Exception("Remote functions cannot be called directly. "
                        "Instead of running '{}()', try '{}.remote()'."
                        .format(func_name, func_name))

      func_invoker.remote = func_or_class
      func_invoker.executor = func_or_class
      func_invoker.is_remote = True
      func_invoker.func_name = func_name
      return func_invoker
    elif inspect.isclass(func_or_class):
      cls = func_or_class

      class Class(func_or_class):
        def __ray_terminate__(self, actor_id):
          pass

      Class.__module__ = cls.__module__
      Class.__name__ = cls.__name__

      # 目前只能在ray-tune中移除所有.remote方法
      # TODO: 借鉴ray.actor，make_actor_handle_class, 完成对类方法的.remote封装

      class ActorHandle(object):
        @classmethod
        def remote(cls, *args, **kwargs):
          return Class(*args, **kwargs)

      return ActorHandle
    else:
      raise Exception("The @ray.remote decorator must be applied to "
                      "either a function or to a class.")

  if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
    func_or_class = args[0]
    return make_remote_decorator(func_or_class)
  else:
    return make_remote_decorator


class GlobalState(object):
  def client_table(self):
    entry = {
        'ClientType': 'local_scheduler',
        'Deleted': False,
        # TODO: 解除限死资源，目前只能单线程运行
        'CPU': 1,
        'GPU': 1,
    }
    client = [entry]
    clients = {
      '127.0.0.1': client
    }
    return clients


global_state = GlobalState()
