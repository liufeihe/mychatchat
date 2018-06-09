# -*- coding:utf-8 -*-

import os

try:
    from ConfigParser import SafeConfigParser
except:
    # In Python 3, ConfigParser has been renamed to configparser for PEP 8 compliance.
    from configparser import ConfigParser as SafeConfigParser


def get_config(config_file='seq2seq.ini'):
    parser = SafeConfigParser()
    parser.read(os.path.join(os.path.dirname(__file__), config_file))
    # get the ints, floats and strings
    _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
    _conf_floats = [(key, float(value)) for key, value in parser.items('floats')]
    _conf_strings = [(key, str(value)) for key, value in parser.items('strings')]
    return dict(_conf_ints + _conf_floats + _conf_strings)