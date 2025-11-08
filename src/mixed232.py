import sys
import numpy as np
import nibabel as nib
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree
from skimage.measure import label
from keras_contrib.layers import InstanceNormalization
from keras.models import load_model, Model
from skimage.segmentation import find_boundaries

from utils.losses import soft_dice_loss, soft_dice_score
from utils.utils import cersegsys_test_prt, cersegsys_train_prt
from paper_writing.visualizations_for_papers import BrainstemVisualizer
from attention.attention import cbam_block
from scipy.spatial.distance import cdist



def convert_mixed_precision_to_float32(model):
    config = model.get_config()

    for layer in config['layers']:
        if 'dtype' in layer['config']:
            layer['config']['dtype'] = 'float32'
        if 'mixed_precision' in layer['config']:
            layer['config']['mixed_precision'] = False

    new_model = tf.keras.Model.from_config(config, custom_objects={'soft_dice_score': soft_dice_score, 'soft_dice_loss': soft_dice_loss, 'InstanceNormalization': InstanceNormalization, 'cbam_block': cbam_block})

    for layer in new_model.layers:
        if layer.weights:
            original_layer = model.get_layer(layer.name)
            layer.set_weights(original_layer.weights())

    return new_model