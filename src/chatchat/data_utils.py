# -*- coding:utf-8 -*-

import math
import os
import random
import getConfig
from tensorflow.python.platform import gfile
import re
import jieba

# 特殊标记，用来填充标记对话
PAD = "__PAD__"
GO = "__GO__"
EOS = "__EOS__"  # 对话结束
UNK = "__UNK__"  # 标记未出现在词汇表中的字符
START_VOCABULART = [PAD, GO, EOS, UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# 用于语句切割的正则表达
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


# 定义字典生成函数
def create_vocabulary(input_file, vocabulary_size, ouput_file):
    vocabulary = {}
    k = int(vocabulary_size)
    with open(input_file, 'r') as f:
        counter = 0
        for line in f:
            counter += 1
            tokens = [word for word in line.split()]
            for word in tokens:
                if word in vocabulary:
                    vocabulary[word] += 1
                else:
                    vocabulary[word] = 1
        vocabulary_list = START_VOCABULART + sorted(vocabulary, key=vocabulary.get, reverse=True)
        # 取前20000个常用汉字
        if len(vocabulary_list) > k:
            vocabulary_list = vocabulary_list[:k]
        print input_file+" 词汇表大小:"+str(len(vocabulary_list))
        with open(ouput_file, 'w') as ff:
            for word in vocabulary_list:
                ff.write(word+'\n')


# 把对话字符串转为向量形式
def convert_to_vector(input_file, vocabulary_file, output_file):
    print '对话转向量...'
    tmp_vocab = []
    # 读取字典文件的数据，生成一个dict，也就是键值对的字典
    with open(vocabulary_file, 'r') as f:
        tmp_vocab.extend(f.readlines())
    tmp_vocab = [line.strip() for line in tmp_vocab]
    # 将vocabulary_file中的键值对互换，因为在字典文件里是按照{123：好}这种格式存储的，我们需要换成{好：123}格式
    vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])

    output_f = open(output_file, 'w')
    with open(input_file, 'r') as f:
        for line in f:
            line_vec = []
            for words in line.split():
                line_vec.append(vocab.get(words, UNK_ID))
            # 将input_file里的中文字符通过查字典的方式，替换成对应的key，并保存在output_file
            output_f.write(" ".join([str(num) for num in line_vec])+'\n')
    output_f.close()


def prepare_custom_data(working_directory, train_enc, train_dec, test_enc, test_dec, enc_vocabulary_size, dec_vocabulary_size, tokenizer=None):
    # Create vocabularies of the appropriate sizes.
    enc_vocab_path = os.path.join(working_directory, 'vocab%d.enc'%enc_vocabulary_size)
    dec_vocab_path = os.path.join(working_directory, 'vocab%d.dec'%dec_vocabulary_size)
    create_vocabulary(train_enc, enc_vocabulary_size, enc_vocab_path)
    create_vocabulary(train_dec, dec_vocabulary_size, dec_vocab_path)

    # Create token ids for the training data.
    enc_train_ids_path = train_enc + ('.ids%d'%enc_vocabulary_size)
    dec_train_ids_path = train_dec + ('.ids%d'%dec_vocabulary_size)
    convert_to_vector(train_enc, enc_vocab_path, enc_train_ids_path)
    convert_to_vector(train_dec, dec_vocab_path, dec_train_ids_path)

    # Create token ids for the development data.
    enc_dev_ids_path = test_enc + ('.ids%d'%enc_vocabulary_size)
    dec_dev_ids_path = test_dec + ('.ids%d'%dec_vocabulary_size)
    convert_to_vector(test_enc, enc_vocab_path, enc_dev_ids_path)
    convert_to_vector(test_dec, dec_vocab_path, dec_dev_ids_path)

    return enc_train_ids_path, dec_train_ids_path, enc_dev_ids_path, dec_dev_ids_path, enc_vocab_path, dec_vocab_path


def basic_tokenizer(sentence):
    # 将一个语句中的字符切割成一个list，这样是为了下一步进行向量化训练
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def sentence_to_token_ids(sentence, vocabulary, normalize_digits=True):
    # 将输入语句从中文字符转换成数字符号
    words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]


def initialize_vocabulary(vocabulary_path):
    vocabulary_path = os.path.join(os.path.dirname(__file__), vocabulary_path)
    # 初始化字典，这里的操作与上面的48行的的作用是一样的，是对调字典中的key-value
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with open(vocabulary_path, 'r') as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError('Vocabulary file %s is not found'%vocabulary_path)


def convert_seq2seq_files(questions, answers, test_set_size):
    # 创建文件
    train_enc = open(gConfig['train_enc'], 'w')  # ask
    train_dec = open(gConfig['train_dec'], 'w')  # answer
    test_enc = open(gConfig['test_enc'], 'w')  # ask
    test_dec = open(gConfig['test_dec'], 'w')  # answer

    test_index = random.sample([i for i in range(len(questions))], test_set_size)
    for i in range(len(questions)):
        if i in test_index:
            test_enc.write(questions[i]+'\n')
            test_dec.write(answers[i]+'\n')
        else:
            train_enc.write(questions[i] + '\n')
            train_dec.write(answers[i] + '\n')
        if i % 1000 == 0:
            print(len(range(len(questions))), '处理进度：', i)

    train_enc.close()
    train_dec.close()
    test_enc.close()
    test_dec.close()


def prepare_data():
    conv_path = gConfig['resource_data']
    if not os.path.exists(conv_path):
        exit()

    convs = []  # 用于存储对话集合
    with open(conv_path) as f:
        one_conv = []  # 存储一次完整对话
        for line in f:
            line = line.strip('\n').replace('/', '')  # 去除换行符，并在字符间添加空格符，原因是用于区分 123 与1 2 3.
            if line == '':
                continue
            if line[0] == gConfig['e']:
                if one_conv:
                    convs.append(one_conv)
                one_conv = []
            elif line[0] == gConfig['m']:
                one_conv.append(line.split(' ')[1])  # 将一次完整的对话存储下来

    # 把对话分成问与答两个部分
    ask = []
    response = []
    for conv in convs:
        if len(conv) == 1:
            continue
        if len(conv)%2 != 0:  # 因为默认是一问一答的，所以需要进行数据的粗裁剪，对话行数要是偶数的
            conv = conv[:-1]
        for i in range(len(conv)):
            conv[i] = " ".join(jieba.cut(conv[i]))  # 使用jieba分词器进行分词
            if i%2 == 0:
                ask.append(conv[i])  # 因为i是从0开始的，因此偶数行为发问的语句，奇数行为回答的语句
            else:
                response.append(conv[i])

    # 生成的*.enc文件保存了问题
    # 生成的*.dec文件保存了回答
    convert_seq2seq_files(ask, response, 10000)


if __name__ == '__main__':
    gConfig = getConfig.get_config()
    prepare_data()