import tensorflow as tf
import keras
from keras.models import load_model
from utils.losses import *
import os
from keras_contrib.layers import InstanceNormalization
import json

checkpoint_path = 'D:\\university\\phd\\brainchop\\backend\\models\\cerebellum-parcellation\\model.h5'
output_path = 'D:\\university\\phd\\brainchop\\backend\\models\\cerebellum-parcellation\\code.py'

def model_to_code(h5_file_path, output_py_file):

    """
    Convert a trained Keras model (.h5) to Python code and save to .py file

    Args:
        h5_file_path (str): Path to the input .h5 model file
        output_py_file (str): Path for the output .py file
    """

    model = load_model(h5_file_path, custom_objects={'soft_dice_score': soft_dice_score, 'soft_dice_loss': soft_dice_loss, 'InstanceNormalization': InstanceNormalization, 'calc_aver_dice_loss': soft_dice_loss})
    config = model.get_config()

    code  = f"""# Auto-generated model architecture from {h5_file_path}
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import {', '.join({layer['class_name'] for layer in config['layers'] if layer['class_name'] not in ['InputLayer', 'InstanceNormalization']})}
from keras_contrib.layers import InstanceNormalization
from utils.losses import *

def create_model():
    model = Sequential()
"""
    
    # Add layers to the code
    for layer in config['layers']:
        print(layer)
        if layer['class_name'] == 'InputLayer':
            continue

        layer_config = layer['config']
        layer_code = f"    model.add({layer['class_name']})("

        # Add layer parameters
        params = []
        for key, value in layer_config.items():
            if key == 'name':
                continue
            if isinstance(value, (str, list, tuple, dict)):
                params.append(f"{key}={repr(value)}")
            else:
                params.append(f"{key}={value}")

        layer_code += ', '.join(params) + ")"
        code += layer_code + "\n"

    code += """
    return model


if __name__ == '__main__':
    model = create_model()
    model.summary()
"""

    # Save to .py file
    with open(output_py_file, 'w') as f:
        f.write(code)

    print(f'Model architecture saved to {output_py_file}')

if __name__ == '__main__':
    model_to_code(checkpoint_path, output_path)
    
        
    


