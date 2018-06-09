# -*- coding: utf-8 -*-

import tensorflow as tf
from chatchat import seq2seq_execute

_model_info = None


def init_model():
    global _model_info
    if not _model_info:
        print 'loading model going....'
        sess = tf.Session()
        sess, model, enc_vocab, rev_dec_vocab = seq2seq_execute.init_session(sess)
        _model_info = {'sess': sess, 'model': model, 'enc_vocab': enc_vocab, 'rev_dec_vocab': rev_dec_vocab}
        print 'loading model ok.'


def get_model():
    global _model_info
    if not _model_info:
        print 'loading model going....'
        sess = tf.Session()
        sess, model, enc_vocab, rev_dec_vocab = seq2seq_execute.init_session(sess)
        _model_info = {'sess': sess, 'model': model, 'enc_vocab': enc_vocab, 'rev_dec_vocab': rev_dec_vocab}
        print 'loading model ok.'

    return _model_info


def ask_answer(msg):
    if msg == '':
        return '请与我聊聊天吧'

    model = get_model()

    print('init...')
    msg = ''.join([f + ' ' for fh in msg for f in fh])
    print(msg)
    print('answering...')
    resp_msg = seq2seq_execute.decode_line(model['sess'], model['model'], model['enc_vocab'], model['rev_dec_vocab'], msg)
    resp_msg = resp_msg.replace('_UNK', '^_^')
    resp_msg = resp_msg.strip()
    print resp_msg
    return resp_msg
