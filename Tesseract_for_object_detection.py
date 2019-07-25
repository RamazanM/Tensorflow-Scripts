#!/usr/bin/python3
## AUTHOR: Ramazan Mutlu
## Repository: https://github.com/RamazanM/Tensorflow-Scripts
##TODO:Add comment lines

import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from PIL import Image,ImageDraw
import pytesseract

sys.path.append("..")
sys.path.append("REPLACE_WITH_YOUR_TENSORFLOW_LIBRARIES_FOLDER")    ## Added tensorflow libraries to path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  ## Used for disable unnecessary logs

from utils import label_map_util
from utils import visualization_utils as vis_util

from absl import flags,app


flags.DEFINE_string("graph_dir","inference_graph","Path of inference graph folder")
flags.DEFINE_string("image_path",None,"Path of the test image")
flags.DEFINE_string("labelmap_path",None,"Path of the labelmap.pbtxt")
flags.DEFINE_integer("num_classes",None,"Number of trained classes")
flags.DEFINE_string("search_class",None,"Select a class for ocr operation (if exist, show only selected class' ocr result)")
flags.mark_flag_as_required("image_path")
flags.mark_flag_as_required("labelmap_path")
flags.mark_flag_as_required("num_classes")
FLAGS=flags.FLAGS

def main(argv):
    MODEL_NAME = FLAGS.graph_dir
    IMAGE_NAME = FLAGS.image_path
    CWD_PATH = os.getcwd()
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join(CWD_PATH,FLAGS.labelmap_path)
    PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)
    NUM_CLASSES = FLAGS.num_classes


    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.compat.v1.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    image = cv2.imread(PATH_TO_IMAGE)
    image_expanded = np.expand_dims(image, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})




    img=Image.open(PATH_TO_IMAGE)
    drw=img.copy()
    h=img.size[1]
    w=img.size[0]
    d=ImageDraw.Draw(drw)
    pad=10
    for i,box in enumerate(boxes[0]):
        if scores[0][i]>0.6:
            if(FLAGS.search_class==None or FLAGS.search_class==category_index.get(classes[0][i]).get('name')):
                img_crop=img.crop((box[1]*w-pad, box[0]*h-pad,box[3]*w+pad, box[2]*h+pad))
                text = pytesseract.image_to_string(img_crop,lang="tur",config='--psm 1 --oem 1')
                print("########-"+str(i)+"-##########\n"+text)
                d.rectangle([(box[1]*w-pad, box[0]*h-pad),(box[3]*w+pad, box[2]*h+pad)],outline="#f00")
                d.text([box[1]*w,box[0]*h],str(i),fill="#00a")
    drw.show()

if __name__ == '__main__':
    app.run(main)