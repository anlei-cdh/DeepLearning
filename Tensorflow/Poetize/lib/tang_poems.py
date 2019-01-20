# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from lib.model import rnn_model
from lib.poems import process_poems, generate_batch
import Tensorflow.Util.DBUtil as dbUtil

tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate.')
tf.app.flags.DEFINE_string('checkpoints_dir', os.path.abspath('./checkpoints/poems/'), 'checkpoints save path.')
tf.app.flags.DEFINE_string('file_path', os.path.abspath('./data/poems.txt'), 'file name of poems.')
tf.app.flags.DEFINE_string('model_prefix', 'poems', 'model save prefix.')
tf.app.flags.DEFINE_integer('epochs', 50, 'train how many epochs.')

FLAGS = tf.app.flags.FLAGS

start_token = 'G' #'G'
end_token = '。' #'E'


def run_training():
    if not os.path.exists(os.path.dirname(FLAGS.checkpoints_dir)):
        os.mkdir(os.path.dirname(FLAGS.checkpoints_dir))
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.mkdir(FLAGS.checkpoints_dir)

    poems_vector, word_to_int, vocabularies = process_poems(FLAGS.file_path)
    batches_inputs, batches_outputs = generate_batch(FLAGS.batch_size, poems_vector, word_to_int)

    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(init_op)

        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("[INFO] restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('[INFO] start training...')
        try:
            for epoch in range(start_epoch, FLAGS.epochs):
                n = 0
                n_chunk = len(poems_vector) // FLAGS.batch_size
                for batch in range(n_chunk):
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']
                    ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                    n += 1
                    print('[INFO] Epoch: %d , batch: %d , training loss: %.6f' % (epoch, batch, loss))
                    
                if epoch % 6 == 0:
                    saver.save(sess, './model/', global_step=epoch)
                    #saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_prefix), global_step=epoch)
        except KeyboardInterrupt:
            print('[INFO] Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_prefix), global_step=epoch)
            print('[INFO] Last epoch were saved, next time will start from epoch {}.'.format(epoch))


def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]

def get_poem_cotent(init_op,saver,word_int_map,end_points,input_data,begin_word,vocabularies):
    with tf.Session() as sess:
        sess.run(init_op)

        #checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
        checkpoint = tf.train.latest_checkpoint('./model/')
        #saver.restore(sess, checkpoint)
        saver.restore(sess, './model/-24')

        x = np.array([list(map(word_int_map.get, start_token))])

        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})
        if begin_word:
            word = begin_word
        else:
            word = to_word(predict, vocabularies)
        poem = ''
        num = 0
        while num < 4 and len(poem) < 200: # while word != end_token and len(poem) < 100:
            # print ('runing')
            if(word == end_token):
                num += 1
            poem += word
            x = np.zeros((1, 1))
            x[0, 0] = word_int_map[word]
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x, end_points['initial_state']: last_state})
            word = to_word(predict, vocabularies)
        # word = words[np.argmax(probs_)]
        poem_content = poem.replace(" ", "").replace("G","")
        if(check_poem(poem_content)):
            return poem_content
        else:
            return get_poem_cotent(init_op,saver,word_int_map,end_points,input_data,begin_word,vocabularies)

def check_poem(poem_content):
    poem_len = len(poem_content)
    print(poem_content)
    if(poem_len >= 48): # 诗的长度必须是大于等于48的
        poem_sentences = poem_content.split('。')
        sentences_len = len(poem_sentences)
        if(sentences_len == 5): # 必须是四句诗
            word_len = 0
            index = 0
            for s in poem_sentences:
                index += 1
                if(word_len == 0): # 第一句的长度赋值
                    word_len = len(s)
                print(index, " - ", word_len, " - ", len(s))
                if(word_len > 20 or word_len < 11): # 每句诗的长度在11-19之间
                    return False
                if(word_len != len(s) and index < 5): # 每句诗的长度必须相等，最后一句是空字符串不需要比对
                    return False
            return True
        else:
            return False

    return False

def gen_poem(begin_word):
    batch_size = 1

    poems_vector, word_int_map, vocabularies = process_poems(FLAGS.file_path)

    input_data = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    poem_content = get_poem_cotent(init_op,saver,word_int_map,end_points,input_data,begin_word,vocabularies)
    return poem_content

dbHelper = dbUtil.DBUtil()

def pretty_print_poem(poem):
    poem_sentences = poem.split('。')
    poem_result = ""
    for s in poem_sentences:
        if s != '' and len(s.replace(" ","")) > 5:
            line = s + '。'
            poem_result += line

    sql = "UPDATE dl_poem_data SET content = '%s' WHERE id = 3" % (poem_result)
    dbHelper.runSql(sql)
    print("result: ",poem_result)

def getPoem(begin_word):
    poem2 = gen_poem(begin_word)
    pretty_print_poem(poem2)

def main():
    getPoem("飞")
    # run_training()

if __name__ == '__main__':
    tf.app.run()