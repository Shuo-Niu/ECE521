import tensorflow as tf
import os
import math

class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.init_global_step()
        self.init_cur_epoch()

    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, os.path.join(self.config.checkpoint_dir, self.config.exp_name), self.global_step_tensor)

    def load(self, sess, file_name=0 ):
        if file_name == 0:
            latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
            if latest_checkpoint:
                print("Loading model checkpoint {} ... \n".format(latest_checkpoint))
                self.saver.restore(sess, latest_checkpoint)
                print("Model loaded")
            else:
                print("Model loaded fail")
        else:
            # selected_checkpoint = os.path.join(self.config.checkpoint_dir,file_name)
            selected_checkpoint = os.path.join(file_name)
            print("Loading model checkpoint {} ... \n".format(selected_checkpoint))
            self.saver.restore(sess, selected_checkpoint)
            print("Model loaded")


    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0., trainable=False, name = 'cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    def init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0., trainable=False, name = 'global_step')

    def init_saver(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError


class NN(BaseModel):
    def __init__(self, config):
        super(NN, self).__init__(config)

        self.num_hidden_layer = config.num_hidden_layer
        self.hidden1_size = config.hidden1_size
        self.hidden2_size = config.hidden2_size

        self.BATCH_SIZE = self.config.train_batch_size
        self.IMAGE_SIZE = self.config.image_size
        self.output_size = self.config.output_size
        self.weight_decay = float(self.config.weight_decay)

        self.dropout_rate = config.dropout_rate


        self.build_model()
        self.loss()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=(None, self.IMAGE_SIZE, self.IMAGE_SIZE))
        self.y = tf.placeholder(tf.float32, shape=(None, self.output_size))

        self.input = tf.reshape(self.x, [-1, self.IMAGE_SIZE * self.IMAGE_SIZE])

        if self.num_hidden_layer == 1:
            self.hidden1 = tf.layers.dense(inputs= self.input,\
                                      units= self.hidden1_size, \
                                      kernel_initializer= tf.contrib.layers.xavier_initializer(), \
                                      bias_initializer=tf.zeros_initializer(), \
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay), \
                                      activation=tf.nn.relu,\
                                      name= 'hidden1')
            self.dropout1 = tf.layers.dropout(inputs=self.hidden1, \
                                         rate=self.dropout_rate, \
                                         training=self.is_training)
            self.output = tf.layers.dense(self.dropout1, \
                                      units=self.config.output_size, \
                                      name='output')
        else:
            self.hidden1 = tf.layers.dense(inputs=self.input, \
                                      units=self.hidden1_size, \
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(), \
                                      bias_initializer=tf.zeros_initializer(), \
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay), \
                                      activation=tf.nn.relu, \
                                      name='hidden1')
            self.dropout1 = tf.layers.dropout(inputs=self.hidden1, \
                                         rate=self.dropout_rate, \
                                         training=self.is_training)
            self.hidden2 = tf.layers.dense(inputs=self.dropout1, \
                                      units=self.hidden2_size, \
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(), \
                                      bias_initializer=tf.zeros_initializer(), \
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay), \
                                      activation=tf.nn.relu, \
                                      name='hidden2')
            self.dropout2 = tf.layers.dropout(inputs=self.hidden2, \
                                         rate=self. dropout_rate, \
                                         training=self.is_training)
            self.output = tf.layers.dense(self.dropout2, \
                                          units=self.config.output_size, \
                                          name='output')


    def loss(self):
        # loss
        self.cross_e = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.output))

        self.weight_decay_loss = tf.losses.get_regularization_loss()

        self.total_loss = self.cross_e + self.weight_decay_loss

        #self.total_loss = self.cross_e
        # accuracy
        self.prediction = tf.argmax(self.output,1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, tf.argmax(self.y,1)), tf.float32))

        # update parameters
        if self.config.adam:
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.total_loss, global_step=self.global_step_tensor)
        else:
            self.train_step = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.total_loss, global_step=self.global_step_tensor)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
