import argparse
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os
from tqdm import tqdm

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255)

    :param img: the array to convert [nx, ny, channels]

    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)

    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img


if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./frozen_models/model4.pb", type=str,
                        help="Frozen model file to import")
    parser.add_argument("--input", default="./quangnam_cut", type=str)
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.model)

    input_data = graph.get_tensor_by_name('prefix/Placeholder:0')
    predictor = graph.get_tensor_by_name('prefix/div_1:0')
    keep_prob = graph.get_tensor_by_name('prefix/Placeholder_2:0')
    paths = os.listdir(args.input)
    results = []
    with tf.Session(graph=graph) as sess:
        for idx, path in enumerate(paths):
            if idx % 100 == 0:
                print("Completed {}%".format(idx/len(paths)*100))
            img = Image.open(os.path.join(args.input, path))
            img = ImageOps.expand(img, border=20, fill='black')
            x_test = np.array(img)
            max = np.max(x_test)
            if max == 0:
                x_test = 255 - x_test
            x_test = (x_test - np.min(x_test)) / np.max(x_test)
            x_test = np.asarray([x_test])
            pred = sess.run(predictor, feed_dict={input_data: x_test,
                                                  keep_prob: 1.})[0]
            x_test = to_rgb(pred[..., 1])
            # print(x_test.round().astype(np.uint8))
            x_test = Image.fromarray(x_test.round().astype(np.uint8))
            x_test.save("./quangnam_results/{}".format(path))
