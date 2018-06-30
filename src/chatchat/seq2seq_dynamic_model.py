# -*- coding:utf-8 -*_
# python basic2_seq2seq.py --mode train
# python basic2_seq2seq.py --mode predict

import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.platform import gfile
import numpy as np
import os
import sys
import data_utils


class Config(object):
    epochs = 10
    batch_size = 64
    rnn_size = 256
    rnn_layers = 3
    encoding_embedding_size = 15
    decoding_embedding_size = 15
    learning_rate = 0.01

    checkpoint = './dynamic_seq2seq_dir/train_dynamic_model2.ckpt'
    display_step = 10
    max_train_data_size = 5000
    source_vocab_path = './working_dir/vocab20000.enc'
    target_vocab_path = './working_dir/vocab20000.dec'
    train_source_data_path = './working_dir/train.enc.ids20000'
    train_target_data_path = './working_dir/train.dec.ids20000'
    test_source_data_path = './working_dir/test.enc.ids20000'
    test_target_data_path = './working_dir/test.dec.ids20000'


class DynamicSeq2Seq(object):
    def __init__(self, mode, config):
        self.mode = mode
        self.config = config
        # load data
        source_int_to_letter, source_letter_to_int, \
        target_int_to_letter, target_letter_to_int = self.load_data_vocab()
        self.source_int_to_letter = source_int_to_letter
        self.source_letter_to_int = source_letter_to_int
        self.target_int_to_letter = target_int_to_letter
        self.target_letter_to_int = target_letter_to_int
        # build graph
        self.build_graph()

    def train(self, sess, saver):
        config = self.config
        batch_size = config.batch_size
        train_source, train_target = \
            self.read_data(config.train_source_data_path, config.train_target_data_path, config.max_train_data_size)
        valid_source, valid_target = \
            self.read_data(config.test_source_data_path, config.test_target_data_path, config.max_train_data_size)

        (valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = \
            next(self.get_batches(valid_target, valid_source, batch_size,
                                  self.source_letter_to_int[data_utils.PAD],
                                  self.target_letter_to_int[data_utils.PAD]))
        display_step = config.display_step
        checkpoint = os.path.join(os.path.dirname(__file__), config.checkpoint)

        sess.run(tf.global_variables_initializer())
        for epoch_i in range(1, config.epochs + 1):
            for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                    self.get_batches(train_target, train_source, batch_size,
                                     self.source_letter_to_int[data_utils.PAD],
                                     self.target_letter_to_int[data_utils.PAD])
            ):
                _, loss = sess.run([self.train_op, self.loss], {
                    self.inputs: sources_batch,
                    self.targets: targets_batch,
                    self.learning_rate: config.learning_rate,
                    self.target_sequence_length: targets_lengths,
                    self.source_sequence_length: sources_lengths
                })
                if batch_i % display_step == 0:
                    validation_loss = sess.run([self.loss], {
                        self.inputs: valid_sources_batch,
                        self.targets: valid_targets_batch,
                        self.learning_rate: config.learning_rate,
                        self.target_sequence_length: valid_targets_lengths,
                        self.source_sequence_length: valid_sources_lengths
                    })
                    print 'Epoch {:>3}/{} Batch {:>4}/{} - Trainging loss: {:>6.3f} - Validation loss: {:>6.3f}' \
                        .format(epoch_i, self.config.epochs, batch_i, len(train_source) // batch_size, loss, validation_loss[0])
            saver.save(sess, checkpoint)
            print 'model trained and saved'

    def predict(self, sess, saver):
        config = self.config
        input_word = '你好'
        text = self.source_to_seq(input_word)
        checkpoint = config.checkpoint
        batch_size = config.batch_size
        loader = tf.train.import_meta_graph(os.path.join(os.path.dirname(__file__), config.checkpoint + '.meta'))
        loader.restore(sess, os.path.join(os.path.dirname(__file__), config.checkpoint))
        graph = tf.get_default_graph()
        input_data = graph.get_tensor_by_name('inputs:0')
        logits = graph.get_tensor_by_name('predictions:0')
        source_sequence_length = graph.get_tensor_by_name('source_sequence_length:0')
        target_sequence_length = graph.get_tensor_by_name('target_sequence_length:0')
        answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                          target_sequence_length: [len(input_word)] * batch_size,
                                          source_sequence_length: [len(input_word)] * batch_size
                                          })[0]
        pad = self.source_letter_to_int[data_utils.PAD]
        print 'input:' + input_word
        print '\n Source'
        print '   Word 编号： {}'.format([i for i in text])
        print '   Input Words: {}'.format(' '.join([self.source_int_to_letter[i] for i in text]))
        print '\n Target'
        print '   Word 编号： {}'.format([i for i in answer_logits if i != pad])
        print '   Response Words: {}'.format(' '.join([self.target_int_to_letter[i] for i in answer_logits if i != pad]))

    def source_to_seq(self, text):
        sequence_length = 7
        source_letter_to_int = self.source_letter_to_int
        return [source_letter_to_int.get(word, source_letter_to_int[data_utils.UNK]) for word in text] + \
               [source_letter_to_int[data_utils.PAD]] * (sequence_length - len(text))

    def get_batches(self, targets, sources, batch_size, source_pad_int, target_pad_int):
        for batch_i in range(0, len(sources) // batch_size):
            start_i = batch_i * batch_size
            sources_batch = sources[start_i: start_i + batch_size]
            targets_batch = targets[start_i: start_i + batch_size]
            pad_sources_batch = np.array(self.pad_sentence_batch(sources_batch, source_pad_int))
            pad_targets_batch = np.array(self.pad_sentence_batch(targets_batch, target_pad_int))
            targets_lengths = []
            for target in targets_batch:
                targets_lengths.append(len(target))
            source_lengths = []
            for source in targets_batch:
                source_lengths.append(len(source))

            yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths

    def pad_sentence_batch(self, sentence_batch, pad_int):
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

    def build_graph(self):
        with tf.variable_scope('input'):
            self.inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
            self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            self.target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
            self.max_target_sequence_length = tf.reduce_max(self.target_sequence_length, name='max_target_length')
            self.source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

        with tf.variable_scope('encoder'):
            encoder_embed_input = tf.contrib.layers.embed_sequence(self.inputs,
                                                                   len(self.source_letter_to_int),
                                                                   self.config.encoding_embedding_size)
            encoder_cell = tf.contrib.rnn.MultiRNNCell([self.get_lstm_cell(self.config.rnn_size) for _ in range(self.config.rnn_layers)])
            encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                              encoder_embed_input,
                                                              sequence_length=self.source_sequence_length,
                                                              dtype=tf.float32)

        with tf.variable_scope('decoder'):
            # 1. embedding
            decoder_input = self.process_decoder_input(self.targets,
                                                       self.target_letter_to_int,
                                                       self.config.batch_size)
            target_vocab_size = len(self.target_letter_to_int)
            decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size,
                                                                self.config.decoding_embedding_size]))
            decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)
            # decoder_embed_input = tf.contrib.layers.embed_sequence(decoder_input, target_vocab_size, self.config.decoding_embedding_size)
            # 2. construct the rnn
            decoder_cell = tf.contrib.rnn.MultiRNNCell([self.get_lstm_cell(self.config.rnn_size) for _ in range(self.config.rnn_layers)])
            # 3. output fully connected
            output_layer = Dense(target_vocab_size,
                                 kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            if self.mode == 'train':
                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                                    sequence_length=self.target_sequence_length,
                                                                    time_major=False)
                training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, training_helper, encoder_state, output_layer)
                decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                                  impute_finished=True,
                                                                                  maximum_iterations=self.max_target_sequence_length)
            else:
                start_tokens = tf.tile(tf.constant([self.target_letter_to_int[data_utils.GO]], dtype=tf.int32),
                                       [self.config.batch_size],
                                       name='start_tokens')
                predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                                             start_tokens,
                                                                             self.target_letter_to_int[data_utils.EOS])
                predicting_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, predicting_helper,
                                                                     encoder_state, output_layer)
                decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                                    impute_finished=True,
                                                                                    maximum_iterations=self.max_target_sequence_length)

        with tf.variable_scope('loss'):
            training_logits = tf.identity(decoder_output.rnn_output, 'logits')
            if self.mode == 'predict':
                predicting_logits = tf.identity(decoder_output.sample_id, name='predictions')
            masks = tf.sequence_mask(self.target_sequence_length, self.max_target_sequence_length, dtype=tf.float32, name='masks')
            self.loss = tf.contrib.seq2seq.sequence_loss(training_logits, self.targets, masks)

        with tf.name_scope('optimize'):
            # optimizer = tf.train.AdamOptimizer(lr)
            # gradients = optimizer.compute_gradients(cost)
            # capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            # train_op = optimizer.apply_gradients(capped_gradients)
            training_variables = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, training_variables), 5)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, training_variables), name='train_op')

    def get_lstm_cell(self, rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell

    def process_decoder_input(self, data, vocab_to_int, batch_size):
        print vocab_to_int[data_utils.GO]
        ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int[data_utils.GO]), ending], 1)
        return decoder_input

    def initialize_vocabulary(self, vocabulary_path):
        vocabulary_path = os.path.join(os.path.dirname(__file__), vocabulary_path)
        # 初始化字典，这里的操作与上面的48行的的作用是一样的，是对调字典中的key-value
        if gfile.Exists(vocabulary_path):
            rev_vocab = []
            with open(vocabulary_path, 'r') as f:
                rev_vocab.extend(f.readlines())
            rev_vocab = [line.strip() for line in rev_vocab]
            vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
            return rev_vocab, vocab
        else:
            raise ValueError('Vocabulary file %s is not found' % vocabulary_path)

    def load_data_vocab(self):
        config = self.config
        # 构造映射表
        source_int_to_letter, source_letter_to_int = self.initialize_vocabulary(config.source_vocab_path)
        target_int_to_letter, target_letter_to_int = self.initialize_vocabulary(config.target_vocab_path)
        return source_int_to_letter, source_letter_to_int, target_int_to_letter, target_letter_to_int

    def read_data(self, source_path, target_path, max_size=None):
        source_ids = []
        target_ids = []
        with tf.gfile.GFile(source_path, mode="r") as source_file:
            with tf.gfile.GFile(target_path, mode="r") as target_file:
                source, target = source_file.readline(), target_file.readline()
                counter = 0
                while source and target and (not max_size or counter < max_size):
                    counter += 1
                    if counter % 100000 == 0:
                        print("  reading data line %d" % counter)
                        sys.stdout.flush()
                    source_id = [int(x) for x in source.split()]
                    target_id = [int(x) for x in target.split()]
                    target_id.append(data_utils.EOS_ID)
                    source_ids.append(source_id)
                    target_ids.append(target_id)
                    source, target = source_file.readline(), target_file.readline()
        return source_ids, target_ids


def main():
    config = Config()
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('mode', '', 'mode')
    mode = FLAGS.mode
    model = DynamicSeq2Seq(mode, config)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        if mode == 'train':
            model.train(sess, saver)
        else:
            model.predict(sess, saver)


if __name__ == '__main__':
    main()
