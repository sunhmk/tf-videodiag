import numpy as np
import cv2
import tensorflow as tf
import os
from xml.etree import ElementTree as ET

import argparse
model_path='./model/pb_model_user'
class train_model:
    __init__():
        pass
def parse_args():
    parser = argparse.ArgumentParser(description='plate locate', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--videocapture', dest='videocapture', default=False, type=bool, help='是否为视频抓拍')
    parser.add_argument('--debug', dest='debug', default=False, type=bool, help='是否打印调试信息')
    parser.add_argument('--picture', dest='picture', default=False, type=bool, help='是否显示图片')
    args = parser.parse_args()
    return args

def main(img_roots):
    pb_path = model_path
    files = os.listdir(img_roots)
    for file in files:
        if file[-4] == '_' and file[:-4]!='.jpg':
            continue
        img = read_img(os.path.join(img_roots, file))
        img_pre = img[np.newaxis, :]
        prediction = freeze_graph_test(pb_path,img_pre)
        if prediction==0:
            cv2.imshow('img',img)
            cv2.waitKey(0)
        write_xml(file,prediction)

def read_img(img_path):
    img = cv2.imread(img_path)
    print(img_path)
    img = cv2.resize(img, (480, 300))
    return img

def freeze_graph_test(pb_path, img):
  '''
  :param pb_path:pb文件的路径
  :param image_path:测试图片的路径
  '''
  with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    with open(pb_path, "rb") as f:
      output_graph_def.ParseFromString(f.read())
      tf.import_graph_def(output_graph_def, name="")
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      # 定义输
      入的张量名称,对应网络结构的输入张量
      input_image_tensor = sess.graph.get_tensor_by_name("input_img:0")
      input_is_training_tensor = sess.graph.get_tensor_by_name("is_training:0")
      output_tensor_name = sess.graph.get_tensor_by_name('prediction:0')

      out=sess.run(output_tensor_name,feed_dict={input_image_tensor:img,input_is_training_tensor:False})
  return out


def write_xml(pic_name, prediction):
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

if __name__ == '__main__':
    args = parse_args()
    
    if args.videocapture:
        pass
    # 保存xml路径
    # 每天时间执行
    main('./data/test')
