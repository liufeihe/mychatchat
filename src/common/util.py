# -*- coding: utf-8 -*-

import json


def _decode_list(lst):
    newlist = []
    for i in lst:
        if isinstance(i, unicode):
            i = i.encode('utf-8')
        elif isinstance(i, list):
            i = _decode_list(i)
        newlist.append(i)
    return newlist


def _decode_dict(dct):
    newdict = {}
    for k, v in dct.iteritems():
        if isinstance(k, unicode):
            k = k.encode('utf-8')
        if isinstance(v, unicode):
            v = v.encode('utf-8')
        elif isinstance(v, list):
            v = _decode_list(v)
        newdict[k] = v
    return newdict


def __json_decode(json_obj, decoding):
    if isinstance(json_obj, dict):
        new_json_obj = {}
        for k,v in json_obj.iteritems():
            new_json_obj[__json_decode(k, decoding)] = __json_decode(v, decoding)
    elif isinstance(json_obj, (tuple, list)):
        new_json_obj = []
        for v in json_obj:
            new_json_obj.append(__json_decode(v, decoding))
    elif isinstance(json_obj, str):
        new_json_obj = json_obj.decode(decoding)
    else:
        new_json_obj = json_obj
    return new_json_obj


def json_load_to_str(s, decoding = None):
    json_obj = json.loads(s, object_hook=_decode_dict) if s else {}
    if decoding:
        json_obj = __json_decode(json_obj, decoding)
    return json_obj
