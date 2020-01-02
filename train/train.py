import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

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
