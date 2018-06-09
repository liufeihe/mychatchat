# -*- coding: utf-8 -*-

import json
import types
import datetime
import decimal
import numpy as np


def request(method):
    def wrapper(self, *args, **kwargs):
        try:
            resp = method(self, *args, **kwargs)
            self.write(resp)
        except Exception, e:
            raise e
    return wrapper


def response(status, data='', user_id=None):
    try:
        resp = {'status': status,
                'data': data}
        resp_str = json.dumps(resp, ensure_ascii=True, default=to_json)
        return resp_str
    except Exception, e:
        raise e


def to_json(python_object):
    """
    convert python object to a dictionary, since json.dump can't deal with python object directly
    """
    if isinstance(python_object, types.InstanceType):
        return python_object.__dict__
    elif isinstance(python_object, (bool, np.bool_, datetime.date, datetime.datetime)):
        return python_object.__str__()
    elif isinstance(python_object, np.int32):
        return int(python_object)
    elif isinstance(python_object, decimal.Decimal):
        return float(python_object)
    raise TypeError(repr(python_object) + 'is not JSON serializable')