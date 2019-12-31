import cv2
import os
from abormalPicDet import train_model
import numpy as np
import tensorflow as tf
from shutil import copyfile


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
        self.fm = self.variance_of_laplacian()

    def predict_image(self):
        # files = os.listdir(self.root)
        # for file in files:
        #     if file[-4] == '_' and file[:-4] != '.jpg':
        #         continue
        img_pre = self.img[np.newaxis, :]
        with tf.Session() as sess:
            prediction = sess.run(self.output_tensor_name,
                                  feed_dict={self.input_image_tensor: img_pre,
                                             self.input_is_training_tensor: False})
        return prediction

    def variance_of_laplacian(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        return fm

    def readLabel(self):
        if self.root == '':
            return 0
        elif self.root =='v':
            return 1

def main():
    root = 'D:\GoogleDownload/trafficDataset/OriginalNoise'
    new_path = './pre'
    files = os.listdir(root)
    for file in files:
        if file[-4] == '_' and file[:-4] != '.jpg':
            continue
        img = imageProcess(root, file)
        if img.prediction == 0:
            print(img.fm)
            img1 = cv2.putText(img.image, "{:.2f}".format(img.fm),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            cv2.imwrite('{}'.format(os.path.join(new_path, file)), img1)


if __name__ == '__main__':
    main()