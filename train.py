import argparse
import tensorflow as tf
from tf_unet import unet, util, image_util
import os
from PIL import Image
import numpy as np
import cv2


def write_graph(sess):
    for variable in tf.global_variables():
        print(variable.name)
        tensor = tf.constant(variable.eval())
        tf.assign(variable, tensor, name="nWeights")
    tf.train.write_graph(sess.graph_def, ".", 'graph2.pb', as_text=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--data_dir', type=str, default="./quangnam_data/*.tif",
                        help='data directory')
    parser.add_argument('--output_path', type=str, default="./output",
                        help='output path')
    parser.add_argument('--layers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--features_root', type=int, default=64,
                        help='number of features')
    parser.add_argument('--channels', type=int, default=3,
                        help='number of channels')
    parser.add_argument('--classes', type=int, default=3,
                        help='number of classes')
    parser.add_argument('--training_iters', type=int, default=32,
                        help='number of training iterations per epoch')
    parser.add_argument('--write_graph', type=bool, default=False,
                        help='write graph def')
    parser.add_argument('--restore', type=bool, default=False,
                        help='restore from a checkpoint')
    args = parser.parse_args()
    train(args)


def train(args):
    # preparing data loading
    data_provider = image_util.ImageDataProvider(args.data_dir, n_class=args.classes, class_colors=[0, 255, 127])

    # setup & training
    net = unet.Unet(layers=args.layers, features_root=args.features_root,
                    channels=args.channels, n_class=args.classes)
    trainer = unet.Trainer(net)
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Total number of parameters:{0}".format(total_parameters))
    trainer.train(data_provider, args.output_path, training_iters=args.training_iters,
                  epochs=args.num_epochs, write_graph=args.write_graph, restore=args.restore)


if __name__ == "__main__":
    main()
