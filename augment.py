import Augmentor
import argparse
import os
from tqdm import tqdm
import numpy as np
import random
from glob import glob
from PIL import Image
from collections import namedtuple
from itertools import cycle

"""
This script will apply distortions such as cropping/rotation/skew on the foreground images
and then superimpose the specified number of such foreground images on different background
images, returning the bounding boxes and labels of the foreground images on the new backgrounds.

A further augmentation of the superimposed images will be done in the form of random distortions
to color, brightness, sharpness, contrast etc. (Only distortions that will preserve the 
new bounding box and coordinates of the foregrounds on the new background images. [TODO]
"""

class ForegroundImage(object):
    """
    The foreground image class to hold data on the foreground image.
    """
    def __init__(self, image, label, xmin, ymin, xmax, ymax, angle):
        self.image = image
        self.label = label
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.angle = angle


class SuperimposedImage(object):
    """
    Wrapper class containing the pipeline of foreground images to be augmented, 
    the background image path that the augmented foreground images are to be superimposed on
    and methods to perform the superimposition.

    Each background image and a set of newly augmented foreground images correspond to a 
    SuperimposedImage class.
    """

    def __init__(self, foreground_pipeline, foreground_num, background_path):
        """
        An instance of SuperimposedImage class.

        :param foreground_pipeline: Pipeline generator object instantiated with foreground images.
        :param foreground_num: Number of foreground images to be superimposed onto the background.
        :param background_path: Path of the background image.

        """
        self.foreground_pipeline = foreground_pipeline
        self.foreground_num = foreground_num
        self.foreground_generator = self.foreground_pipeline.keras_generator(
                batch_size=foreground_num, scaled=False)

        self.foreground_images, self.foreground_labels = next(
                self.foreground_generator)

        self.class_labels = {v : k for k, v in dict(self.foreground_pipeline.class_labels).items()}
        self.foreground_labels = [self.class_labels[np.argmax(lab)] for lab in self.foreground_labels]

        # Convert foreground images that are numpy arrays to PIL images
        # and create named tuples to hold co-ordinate data.
        ForegroundImage = namedtuple(
                'ForegroundImage',
                'image, label, xmin, ymin, xmax, ymax')
        pil_foreground_images = []
        for image, label in zip(self.foreground_images, self.foreground_labels):
            image = Image.fromarray(image)
            image = ForegroundImage(image, label, 0, 0, 0, 0)
            pil_foreground_images.append(image)

        self.foreground_images = pil_foreground_images

        self.background_path = background_path
        self.background_image = Image.open(self.background_path)
        self.background_height = self.background_image.height
        self.background_width = self.background_image.width

        self.superimposed_image = self.background_image.copy()

        self.process_image()

    def process_image(self):
        """ 
        Randomly resize each augmented foreground image and paste it onto a 
        random (x, y) coordinate on the background image.
        """

        new_fg_images = []
        for fg_image in self.foreground_images:
            fg_image = self.random_resize(fg_image.image)

            xmin = random.randint(
                0,
                self.background_width - fg_image.image.width - 1)
            ymin = random.randint(
                0,
                self.background_height - fg_image.image.height - 1)

            fg_image.xmin = xmin
            fg_image.ymin = ymin
            fg_image.xmax = xmin + fg_image.image.width
            fg_image.ymax = ymin + fg_image.image.height

            self.superimposed_image.paste(
                    fg_image.image,
                    (fg_image.xmin, fg_image.ymin),
                    fg_image.image)

            new_fg_images.append(fg_image)

        self.foreground_images = new_fg_images


    def random_resize(self, fg_image):
        """
        Randomly resize the foreground image to a certain scale.
        The minimum size it will be resized to remains static and can be defined inside 
        this function. The maximum size will depend on the dimensions of the background image
        and the num of foreground images to be superimposed. 

        :param fg_image: The foreground as a PIL image that is to be resized.
        :returns: The resized foreground as a PIL image.
        """
        min_scale = 0.05
        max_scale = (1 / self.foreground_num)

        scale_factor = np.random.uniform(min_scale, max_scale)
        x = np.floor(scale_factor * self.background_width)
        y = np.floor(scale_factor * self.background_height)

        resized_fg_image = fg_image.resize((int(x), int(y)))

        return resized_fg_image

    def show(self):
        """
        Visualize the superimposed with bounding boxes
        """
        display_image = self.superimposed_image.copy()

        for image in self.foreground_images:

            draw = ImageDraw.Draw(display_image)
            draw.rectangle(((image.xmin, image.ymin), (image.xmax, image.ymax)), outline="red")
            draw.text((image.xmin, image.ymin), image.label, fill='red')

        display_image.show()


if __name__ == '__main__':
    description = "Create augmented dataset with set of foreground images imposed on different backgrounds."

    parser = argparse.ArgumentParser(description = description)

    parser.add_argument(
            "-f",
            "--foreground",
            type = str,
            required = True,
            help = "The folder containing the samples of the foreground images.")

    parser.add_argument(
            "-b",
            "--background",
            type = str,
            required = True,
            help = "The folder containing background images.")

    parser.add_argument(
            "-nf",
            "--numforeground",
            type = int,
            default = 4,
            help = "No. of foreground images to superimpose per background image")

    parser.add_argument(
            "-n",
            "--noimages",
            type = int,
            default = 10,
            help = "No. of augmented images to create per background image")

    parser.add_argument(
            "-o",
            "--output",
            type = str,
            required = True,
            help = "The output path where we will be saving the images to")

    parser.add_argument(
            "-lab",
            "--labelfile",
            type = str,
            required = True,
            help = "The label file containing the coordinate and label data of the images")

    args = parser.parse_args()

    foreground_path = args.foreground
    background_path = args.background
    no_images = args.noimages
    num_foreground = args.numforeground
    output_dir = args.output
    label_file_path = args.labelfile

    pipeline = Augmentor.Pipeline(foreground_path)

    ## Pipeline operations. We can comment out the ones we do not need.
    pipeline.flip_random(probability=0.7)
    pipeline.gaussian_distortion(
            probability=0.5,
            grid_width=2,
            grid_height=2,
            magnitude=5,
            corner='bell',
            method='in')
    pipeline.random_color(
            probability=0.5,
            min_factor=0.4,
            max_factor=1.0)
    pipeline.random_contrast(
            probability=0.5,
            min_factor=0.4,
            max_factor=1.0)
    pipeline.random_distortion(
            probability=0.9,
            grid_width=5,
            grid_height=5,
            magnitude=6)
    pipeline.random_erasing(
            probability=0.8,
            rectangle_area=0.3)
    pipeline.rotate_random_90(probability=0.5)
    pipeline.scale(probability=0.3, scale_factor=1.5)
    pipeline.shear(
            probability=0.5,
            max_shear_left=20,
            max_shear_right=20)
    pipeline.skew(probability=0.5)


    with open(args.labelfile, 'w') as label_file:
        label_file.write(
                "filename" + "," + \
                "width" + "," + \
                "height" + "," + \
                "class" + "," + \
                "xmin" + "," + \
                "ymin" + "," + \
                "xmax" + "," + \
                "ymax\n")

        backgrounds = cycle(glob(os.path.join(background_path, '*')))

        for ix in tqdm(range(1, no_images + 1)):
            image = SuperimposedImage(pipeline, num_foreground, next(backgrounds))
            image.superimposed_image.convert('RGB').save(os.path.join(output_dir, f'{ix}.jpg'))

            for fg_image in image.foreground_images:
                label_file.write(
                        f'{ix}.jpg' + ',' + \
                        f'{fg_image.image.width}' + ',' \
                        f'{fg_image.image.height}' + ',' \
                        f'{fg_image.label}' + ',' \
                        f'{fg_image.xmin}' + ',' \
                        f'{fg_image.ymin}' + ',' \
                        f'{fg_image.xmax}' + ',' \
                        f'{fg_image.ymax}\n')
