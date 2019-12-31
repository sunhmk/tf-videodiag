import numpy as np
import cv2
import tensorflow as tf
import os
from xml.etree import ElementTree as ET

class  train_model:
    def __init__(self):
        pb_path = './pb_model_user'
        with tf.gfile.FastGFile(pb_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.input_image_tensor, self.input_is_training_tensor,self.output_tensor_name\
                = tf.import_graph_def(
                graph_def, return_elements=["input_img:0","is_training:0",'prediction:0']
            )
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

    def write_xml(self,pic_name, prediction):
        pic = '{}'.format(pic_name)
        pre = '{}'.format(prediction)
        dirs = './sample.xml'
        if not os.path.exists(dirs):
            root = ET.Element('videoNodes')
            tree = ET.ElementTree(root)
            ET.dump(root)
            tree.write('sample.xml', encoding="utf-8", xml_declaration=True)
        tree = ET.parse('./sample.xml')
        root = tree.getroot()
        video = ET.Element('videoNode')
        name = ET.SubElement(video, 'videoId')
        name.text = pic
        normal = ET.SubElement(video, 'normal')
        normal.text = pre
        root.append(video)
        tree.write('sample.xml', encoding="utf-8", xml_declaration=True)
