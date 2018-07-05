"""
Visualization of a given model
"""
import argparse
import math

import dataset
import util

import cv2
import imageio
import keras
import matplotlib.pyplot as plt
import numpy as np
import vis.utils.utils as vutils
import vis.visualization as vvis


def import_model(model_path):
    """
    Utility method for importing a model given its path.
    :param model_path: path to model file.
    :return: Keras model
    """
    return keras.models.load_model(
        model_path,
        compile=False
    )


def prediction_layer_linear_activation(model):
    """
    Utility method for changing prediction layer's activation to use linear.
    This method will reload the model.
    Args:
        model: model to apply change to.

    Returns: reloaded model.

    """
    # Utility to search for layer index by name.
    # Alternatively we can specify this as -1 since it corresponds to the last layer.
    layer_idx = vutils.find_layer_idx(model, 'predictions')

    # Swap softmax with linear
    model.layers[layer_idx].activation = keras.activations.linear

    return util.reload_model(model)


def get_seed_image(bpart, img_size, img_path):
    """
    Utility method for getting a seeding image to visualize attention.
    Pick a random image from validation set, unless img_path is specified.
    :param bpart: Body part to pick.
    :param img_size: Image size to reshape to.
    :param img_path: path to image file.
    :return: Keras model
    """
    _, valid_labeled, _, valid_path = dataset.load_dataframe()
    if not img_path:
        df_valid = dataset.build_dataframe(valid_labeled, valid_path)
        df_valid = df_valid[df_valid["body_part"] == bpart]
        rdm_row = df_valid.sample(1).iloc[0]
        img_path = rdm_row["path"]
        label = rdm_row["label"]
    else:
        label = valid_labeled[valid_labeled["path"] == img_path].iloc[0]["label"]
    img = imageio.imread(img_path)
    img = dataset.grayscale(img)
    img = dataset.zero_pad(img)
    img = cv2.resize(img, (img_size, img_size)).reshape((img_size, img_size, 1))
    return img, img_path, label


def plt_saliency(model, img, ax, idx):
    """
    Plot saliency graph, which generates an image that represents
    the highest activation based on a seeding image;

    Reference: https://arxiv.org/pdf/1312.6034v2.pdf

    Args:
        model: Model to plot.
        img: Seed image.
        ax: Matplotlib axis.
        idx: Index of the plot to be shown on the axis.

    Returns: None

    """
    pred_layer_idx = vutils.find_layer_idx(model, "predictions")

    sal = vvis.visualize_saliency(
        model, pred_layer_idx, filter_indices=None, seed_input=img
    )

    ax[idx].imshow(sal, cmap='jet')
    ax[idx].set_title("Saliency")


def plt_cam(model, img, ax, idx, layer_idx=None):
    """
    Plot Class Activation Map(CAM), which represents the activation
    at the end of all convolutional layer;

    Reference: https://arxiv.org/pdf/1610.02391v1.pdf

    Args:
        model: Model to plot.
        img: Seed image.
        ax: Matplotlib axis.
        idx: Index of the plot to be shown on the axis.
        layer_idx: Index of the layer to plot Grad-CAM. Optional.

    Returns: None

    """
    pred_layer_idx = vutils.find_layer_idx(model, "predictions")
    hmap = vvis.visualize_cam(
        model, pred_layer_idx, filter_indices=None, seed_input=img
    )
    ax[idx].imshow(vvis.overlay(hmap, np.stack((img.reshape(img.shape[0:2]),)*3, -1)))
    ax[idx].set_title("Heatmap")


def plt_attention(model_path, img_path=None, bpart="all", img_size=512, **kwargs):
    """
    Plot attention graph, including saliency and CAM.

    :param model_path: Path to the model
    :param img_path: Path to a validation image. Optional
    :param bpart: Body part to pick if img_path not given
    :param img_size: Size of the image to reshape to
    :param kwargs: Unused arguments
    :return:
    """
    model = import_model(model_path)
    model = prediction_layer_linear_activation(model)

    img, path, label = get_seed_image(bpart, img_size, img_path)

    prediction = model.predict(np.asarray([img]))

    f, ax = plt.subplots(1, 3)

    # plot input image
    ax[0].imshow(img.reshape(img.shape[0:2]), cmap='gray')
    ax[0].set_title("Input")

    # plot saliency
    plt_saliency(model, img, ax, 1)

    # plot heatmap
    plt_cam(model, img, ax, 2)

    plt.tight_layout()
    plt.suptitle("Prediction: {}, Label: {}".format(prediction, label))
    plt.figtext(.5, 0, "Image: {}".format(path))
    plt.show()


def plt_activation(model_path, layer_idx=-1, max_iter=None, **kwargs):
    """
    Plot activation of a given layer in a model by generating an image that
    maximizes the output of all `filter_indices` in the given `layer_idx`.
    Args:
        model_path: Path to the model file.
        layer_idx: Index of the layer to plot.
        max_iter: Maximum number of iterations to generate the input image.
        kwargs: Unused arguments.

    Returns:

    """
    model = import_model(model_path)
    model = prediction_layer_linear_activation(model)
    if type(model.layers[layer_idx]) == keras.layers.Dense:
        img = vvis.visualize_activation(
            model, layer_idx, max_iter=max_iter, filter_indices=None
        )
    else:
        filters = np.arange(vvis.get_num_filters(model.layers[layer_idx]))

        # Generate input image for each filter.
        vis_images = []
        for idx in filters:
            act_img = vvis.visualize_activation(
                model, layer_idx, max_iter=max_iter, filter_indices=idx
            )

            vis_images.append(act_img)

        # Generate stitched image palette with 8 cols.
        img = vutils.stitch_images(vis_images, cols=math.floor(math.sqrt(len(vis_images)*2)))

    plt.axis('off')
    plt.imshow(img.reshape(img.shape[0:2]), cmap="gray")
    plt.show()


if __name__ == "__main__":
    # Define argument parser so that the script can be executed directly
    # from console.
    ARG_PARSER = argparse.ArgumentParser("VGGNet model")
    PARENT_PARSER = argparse.ArgumentParser(add_help=False)
    SUBPARSER = ARG_PARSER.add_subparsers(help='sub-command help')

    # Shared arguments
    PARENT_PARSER.add_argument(
        "-m", "--model_path", type=str, required=True, help="path to model file."
    )

    # Arguments for plotting attention
    ATTENTION_PARSER = SUBPARSER.add_parser("attention", parents=[PARENT_PARSER])
    ATTENTION_PARSER.set_defaults(func=plt_attention)

    ATTENTION_PARSER.add_argument(
        "-i", "--img_path", type=str,
        help="path to image file. If set, use given image instead "
             "of a random on from validation set"
    )

    ATTENTION_PARSER.add_argument(
        "-is", "--img_size", type=int, help="image size to reshape to"
    )

    ATTENTION_PARSER.add_argument(
        "-bp", "--bpart", type=str,
        help="body part to use for training and prediction"
    )

    # Arguments for plotting activation
    ACTIVATION_PARSER = SUBPARSER.add_parser("activation", parents=[PARENT_PARSER])
    ACTIVATION_PARSER.set_defaults(func=plt_activation)

    ACTIVATION_PARSER.add_argument(
        "-l", "--layer_idx", type=int, help="Index of the layer to plot"
    )

    ACTIVATION_PARSER.add_argument(
        "-mi", "--max_iter", type=int, help="Index of the layer to plot"
    )

    # parse argument
    ARGS = ARG_PARSER.parse_args()
    ARGS.func(**vars(ARGS))
