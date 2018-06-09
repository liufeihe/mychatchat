# -*- coding: utf-8 -*-

# from ConfigParser import SafeConfigParser
# from configparser import SafeConfigParser
import getConfig
from datamgr import chat_mgr
#
#
# def cfg_parse_example():
#     parser = SafeConfigParser()
#     parser.read('./seq2seq.ini')
#     _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
#     print _conf_ints


def main():
    # cfg_parse_example()
    getConfig.get_config('../chatchat/seq2seq.ini')
    chat_mgr.ask_answer('我爱你')

if __name__ == '__main__':
    main()
