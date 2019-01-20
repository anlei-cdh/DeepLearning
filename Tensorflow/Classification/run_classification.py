import os.path
import re
import numpy as np
import tensorflow as tf
import Tensorflow.Util.DBUtil as dbUtil
import Tensorflow.Util.HashUtil as hashUtil
import Tensorflow.Util.DictUtil as dictUtil

model_path = "model"
classify_top = 1

class ImageClassify(object):

  def __init__(self, label_lookup_path=None, uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(model_path, 'label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(model_path, 'label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]

def create_graph():
  with tf.gfile.FastGFile(os.path.join(
          model_path, 'classify_image.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

hashHelper = hashUtil.HashUtil()
dbHelper = dbUtil.DBUtil()

def run_image_classify(image_name):
  image_path = "image/" + image_name
  if not tf.gfile.Exists(image_path):
    tf.logging.fatal('File does not exist %s', image_path)
  image_data = tf.gfile.FastGFile(image_path, 'rb').read()

  create_graph()

  with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)
    node_lookup = ImageClassify()
    top_k = predictions.argsort()[-classify_top:][::-1]
    list = []
    for node_id in top_k:
      info_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      info = info_string.split(',')[0]
      classify_string = info.split(' ')
      len = classify_string.__len__()
      classify = classify_string[len - 1]
      type = dictUtil.get_classify_dict(classify)
      hash_id = hashHelper.getHash(image_name)
      sql = "INSERT INTO dl_classify_data(`hash_id`,`name`,`type`,`score`) VALUES(%d,'%s','%s',%f) " \
            "ON DUPLICATE KEY UPDATE `name` = VALUES(`name`),`type` = VALUES(`type`),`score` = VALUES(`score`)" \
            % (hash_id, image_name, type, score)
      list.append(sql)
      print(hash_id,' - ',image_name,' - ',type,' - ',score)
    dbHelper.runSql(list)

if __name__ == '__main__':
  images = ['1.jpg', '2.jpg', '3.jpg', '4.jpg']
  for image_name in images:
    run_image_classify(image_name)