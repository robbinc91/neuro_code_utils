import random
import numpy as np
from scipy.ndimage import rotate, shift
import nibabel as nib


class MRIDataAugmenter:
    def __init__(self, translation_prob=0.5, rotation_prob=0.5, flip_prob=0.5):
        self.translation_prob = translation_prob
        self.rotation_prob = rotation_prob
        self.flip_prob = flip_prob
        self.transformations = []

    def apply_transforms(self, image_path):
        image = nib.load(image_path)
        data = image.get_fdata()
        for transform in self.transformations:
            if transform['type'] == 'translation':
                data = self.translate_image(
                    data, transform['amount'], transform['axis'])
            elif transform['type'] == 'rotation':
                data = self.rotate_image(
                    data, transform['amount'], transform['axis'])
            elif transform['type'] == 'flip':
                data = self.flip_image(data, transform['axis'])
        return data

    def translate_image(self, image, amount, axis):
        shift_values = [0, 0, 0]
        shift_values[axis] = amount
        shifted_image = shift(image, shift_values)
        return shifted_image

    def rotate_image(self, image, amount, axis):
        angle = random.uniform(-amount, amount)
        rotated_image = rotate(image, angle, (axis, 1), reshape=False)
        return rotated_image

    def flip_image(self, image, axis):
        flipped_image = np.flip(image, axis)
        return flipped_image

    def add_translation(self):
        if random.random() < self.translation_prob:
            axis = random.choice([0, 1, 2])
            amount = random.uniform(-10, 10)
            self.transformations.append(
                {'type': 'translation', 'axis': axis, 'amount': amount})

    def add_rotation(self):
        if random.random() < self.rotation_prob:
            axis = random.choice([0, 1, 2])
            amount = 10
            self.transformations.append(
                {'type': 'rotation', 'axis': axis, 'amount': amount})

    def add_flip(self):
        if random.random() < self.flip_prob:
            self.transformations.append({'type': 'flip', 'axis': 2})

    def augment_image(self, image_path):
        self.transformations = []
        self.add_translation()
        self.add_rotation()
        self.add_flip()
        data = self.apply_transforms(image_path)
        return data
