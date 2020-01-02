import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

class LeNet():
    def __init__(self, input_data, is_training):
        self.image_height = 300
        self.image_width = 480
        tf.reset_default_graph()
        self.input_data = input_data
        self.training = is_training

        self.x_input = tf.placeholder(tf.float32, [None, 300, 480, 3], name='input_img')
        self.y_target = tf.placeholder(tf.int64, [None])
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.conv1_weights = tf.Variable(tf.truncated_normal([5, 5, 3, 6],
                                                             stddev=0.1, dtype=tf.float32))
        self.conv1_biases = tf.Variable(tf.zeros([6], dtype=tf.float32))
        self.conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 6, 16],
                                                             stddev=0.1, dtype=tf.float32))
        self.conv2_biases = tf.Variable(tf.zeros([16], dtype=tf.float32))
        self.resulting_width = self.image_width // (2 * 2)
        self.resulting_height = self.image_height // (2 * 2)
        self.full1_input_size = self.resulting_width * self.resulting_height * 16
        self.full1_weight = tf.Variable(tf.truncated_normal([self.full1_input_size, 84],
                                                            stddev=0.1, dtype=tf.float32))
        self.full1_bias = tf.Variable(tf.truncated_normal([84], stddev=0.1, dtype=tf.float32))
        self.full2_weight = tf.Variable(tf.truncated_normal([84, 2], stddev=0.1, dtype=tf.float32))
        self.full2_bias = tf.Variable(tf.truncated_normal([2], stddev=0.1, dtype=tf.float32))


    def my_conv_net(self,input_data,is_training):
        #     lays 1
        conv1 = tf.nn.conv2d(input_data, self.conv1_weights,
                             strides=[1, 1, 1, 1], padding='SAME')
        nor = tf.layers.batch_normalization(tf.nn.bias_add(conv1, self.conv1_biases),
                                            training=is_training)
        relul = tf.nn.relu(nor)
        max_pooll = tf.nn.max_pool(relul, ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1], padding='SAME')
        #     lays 2
        conv2 = tf.nn.conv2d(max_pooll, self.conv2_weights,
                             strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, self.conv2_biases))
        max_pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1], padding='SAME')
        flat_output = tf.layers.flatten(max_pool2)
        #     fc lays1
        if self.training == True:
            fully_connectedl = tf.nn.dropout(tf.nn.relu(
                                            tf.add(tf.matmul(flat_output, self.full1_weight),
                                             self.full1_bias)), 0.5)
        else:
            fully_connectedl = tf.nn.relu(tf.add(
                                            tf.matmul(flat_output, self.full1_weight),
                                            self.full1_bias))

        #     fc lays2
        fully_connected2 = tf.add(tf.matmul(fully_connectedl, self.full2_weight), self.full2_bias)
        return fully_connected2



def main():
    train_data =''
    y = ''
    X_train, X_test, y_train, y_test = train_test_split(train_data, y, test_size=0.1)
    Net = LeNet()
    model_output = Net.my_conv_net(X_train, is_training=True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=model_output, labels=Net.y_target))

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)
    # prediction result
    prediction = tf.argmax(model_output, axis=1, name='prediction')
    correct_prediction = tf.equal(prediction, Net.y_target)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    for i in range(2000):
        rand_index = np.random.choice(7912, 256)
        rand_x = X_train[rand_index]
        rand_y = y_train[rand_index]
        _, err = sess.run([train_op, loss], feed_dict={Net.x_input: rand_x,
                                                       Net.y_target: rand_y, Net.is_training: True})
        if i % 20 == 0:
            acc = sess.run(accuracy, feed_dict={Net.x_input: X_test[:300],
                                                Net.y_target: y_test[:300], Net.is_training: False})
            print('epoch:', i)
            print('loss:', err)
            print('acc:', acc)

if __name__ == '__main__':
    main()
