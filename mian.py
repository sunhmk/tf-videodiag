import cv2
import os
from abormalPicDet import train_model
import numpy as np
import tensorflow as tf

class imageProcess(train_model):

    def __init__(self, root, files, training = False):
        train_model.__init__(self)
        self.root = root
        self.name = os.path.join(root, files)
        self.image = cv2.imread(self.name)
        self.img = cv2.resize(self.image, (480, 300))
        if training ==False:
            self.label = None
        else:
            self.label = self.readLabel()
        self.prediction = self.predict_image()
        self.threshold = 0.90

    def predict_image(self):
        img_pre = self.img[np.newaxis, :]
        with tf.Session() as sess:
            outValue = sess.run([self.output_tensor_name, self.output_tensor_value],
                                  feed_dict={self.input_image_tensor: img_pre,
                                             self.input_is_training_tensor: False})
            if outValue[0][0] > self.threshold:
                prediction = 0
            else:
                prediction = 1

        return prediction


def main():
    root = './test'
    files = os.listdir(root)
    for file in files:
        if file[-4] == '_' and file[:-4] != '.jpg':
            continue
        img = imageProcess(root, file)
        print(img.name)
        img.write_xml(file, img.prediction)

if __name__ == '__main__':
    main()
