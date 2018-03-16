import tensorflow as tf

#from data_loader.data_generator import DataGenerator
from model import*
from train import*
from utils import*
from logger import*
from data_generator import *
def main():
    # capture the config
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir, config.plot_dir])

    # select device
    if config.gpu:
        device_name = "/gpu:0"
    else:
        device_name = "/cpu:0"

    # sess = tf.Session()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    training_data = DataGenerator(sess, config)
    print(training_data)
    if config.logistic:
        model = LogisticClassification(config)
    else:
        model = LinearRegression(config)
    logger = Logger(sess, config)

    with tf.device(device_name):
        train(sess, training_data, model, config, logger)


if __name__ == '__main__':
    main()
