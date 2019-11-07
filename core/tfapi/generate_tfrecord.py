"""
Usage:

# Create train data:
python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/train_labels.csv  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/train.record

# Create test data:
python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/test_labels.csv  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/test.record

python3 generate_tfrecord.py --label0=baret --label1=glass --label2=c3 --label3=headphone --csv_input=/home/erkan/Desktop/tensorflow/workspace/training_demo/annotations/test_labels.csv --output_path=/home/erkan/Desktop/tensorflow/workspace/training_demo/annotations/test.record --img_path=/home/erkan/Desktop/tensorflow/workspace/training_demo/images/test
python3 generate_tfrecord.py --label0=baret --label1=glass --label2=c3 --label3=headphone --csv_input=/home/erkan/Desktop/tensorflow/workspace/training_demo/annotations/train_labels.csv --output_path=/home/erkan/Desktop/tensorflow/workspace/training_demo/annotations/train.record --img_path=/home/erkan/Desktop/tensorflow/workspace/training_demo/images/train
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
import sys
from pathlib import Path
import json
import fileinput


from PIL import Image
from object_detection.utils import dataset_util
from object_detection.protos import string_int_label_map_pb2
from google.protobuf import text_format
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map', '', 'Label map file')
flags.DEFINE_string('img_path', '', 'Path to images')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
# for multiple labels add more else if statements
def class_text_to_int(row_label):

    #if row_label == FLAGS.label:  # 'ship':
    #    return 1
    # comment upper if statement and uncomment these statements for multiple labelling

    label_map_path  = FLAGS.label_map

    label_map = load_labelmap(label_map_path)
    label_map_dict = {}
    for item in label_map.item:
        print(item.name)
        if row_label == item.name:
          return int(item.id)
        else:
            None



def _validate_label_map(label_map):
  """Checks if a label map is valid.
  Args:
    label_map: StringIntLabelMap to validate.
  Raises:
    ValueError: if label map is invalid.
  """
  for item in label_map.item:
    if item.id < 0:
      raise ValueError('Label map ids should be >= 0.')
    if (item.id == 0 and item.name != 'background' and
        item.display_name != 'background'):
      raise ValueError('Label map id 0 is reserved for the background label')

def load_labelmap(path):
  """Loads label map proto.
  Args:
    path: path to StringIntLabelMap proto text file.
  Returns:
    a StringIntLabelMapProto
  """
  with tf.gfile.GFile(path, 'r') as fid:
    label_map_string = fid.read()
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    try:
      text_format.Merge(label_map_string, label_map)
    except text_format.ParseError:
      label_map.ParseFromString(label_map_string)
  _validate_label_map(label_map)
  return label_map

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):

    nameoffile = '{}'.format(group.filename)

    if ".jpg" in nameoffile:
        nameoffile = '{}'.format(group.filename)
    else:
        print("found missing file extention added in your label csv file added .jpg")
        nameoffile = '{}.jpg'.format(group.filename)


    with tf.gfile.GFile(os.path.join(path, nameoffile), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')

    image_format = b'jpg'
    # check if the image format is matching with your images.
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        if row['xmin'] <=1:
            real_xmin = row['xmin'] * width
            print("found xmin lower than 1 recalculated: ",real_xmin)
        else:
            real_xmin = row['xmin']
            print("found xmin normal")

        if row['xmax'] <=1:
            real_xmax = row['xmax'] * width
            print("found xmax lower than 1 recalculated: ",real_xmax)
        else:
            real_xmax = row['xmax']
            print("found xmax normal")

        if row['ymin'] <=1:
            real_ymin = row['ymin'] * height
            print("found ymin lower than 1 recalculated: ",real_ymin)
        else:
            real_ymin = row['ymin']
            print("found ymin normal")

        if row['ymax'] <=1:
            real_ymax = row['ymax'] * height
            print("found ymax lower than 1 recalculated: ",real_ymax)
        else:
            real_ymax = row['ymax']
            print("found ymax normal")


        #0.46	471.04	0.50125	513.28	0.883677	602.667714	0.913696	623.140672



        xmins.append(real_xmin / width)
        xmaxs.append(real_xmax / width)
        ymins.append(real_ymin / height)
        ymaxs.append(real_ymax / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))
        filenameinrow = row['filename']

        print("ThinkerFarm : Genereting TensorFlow record for {}".format(filenameinrow))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):

    label_map_path  = FLAGS.label_map
    label_map = load_labelmap(label_map_path)

    filepath = "trainer/pre_trained_mobilenet/model.config"
    found_line = ""
    with open(filepath) as fp:
        line = fp.readline()
        cnt = 1
        while line:

            if cnt == 9:
                found_line = "{}".format(line.strip())
            line = fp.readline()
            cnt += 1




    with fileinput.FileInput("trainer/pre_trained_mobilenet/model.config", inplace=True, backup='.bak') as file:
        for line in file:
            print(line.replace("{}".format(found_line), "num_classes: {}".format(len(label_map.item))), end='')



    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), FLAGS.img_path)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')






    for group in grouped:
        nameoffile = '{}'.format(group.filename)

        if ".jpg" in nameoffile:
            nameoffile = '{}'.format(group.filename)
        else:
            print("found missing file extention added in your label csv file added .jpg")
            nameoffile = '{}.jpg'.format(group.filename)

        configpath = '%s/%s'%(path,nameoffile)


        config = Path(configpath)
        print("config : ",config)

        if config.is_file():
            tf_example = create_tf_example(group, path)
            writer.write(tf_example.SerializeToString())
        else:
            print("file does not exist")


    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
