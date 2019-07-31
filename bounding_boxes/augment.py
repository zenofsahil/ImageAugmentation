import os
import numpy as np
import random
import logging
import argparse
import imgaug as ia
from imgaug import augmenters as iaa
from tqdm import tqdm
from glob import glob
from PIL import Image
from itertools import cycle

from foreground_image import ForegroundImage
from superimposed_image import SuperimposedImage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
This script will apply distortions such as cropping/rotation/skew on the foreground images
and then superimpose the specified number of such foreground images on different background
images, returning the bounding boxes and labels of the foreground images on the new backgrounds.


TODO
A further augmentation of the superimposed images will be done in the form of random distortions
to color, brightness, sharpness, contrast etc. (Only distortions that will preserve the 
new bounding box and coordinates of the foregrounds on the new background images. [TODO]
"""


### Augmentation pipeline taken as is from https://github.com/aleju/imgaug ###

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.ContrastNormalization((0.5, 2.0))
                    )
                ]),
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                # iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        )
    ],
    random_order=True
)

logger.info('ImgAug pipeline defined.')

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

    with open(args.labelfile, 'w') as label_file:
        logger.info(f'Initialising label output file {args.labelfile}')
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

        fg_image_dirs = glob(os.path.join(foreground_path, '*'))
        logger.info(f'Found foreground image directories {fg_image_dirs}')

        fg_image_classes = [glob(os.path.join(image_dir, '*')) 
            for image_dir in fg_image_dirs]
        logger.info(f'Found foreground image classes: {fg_image_classes}')

        fg_image_paths = [image_path 
            for image_class in fg_image_classes 
            for image_path in image_class]

        fg_images = [Image.open(image) for image in fg_image_paths]
        fg_image_labels = list(map(lambda x: x.split('/')[-2:][0], fg_image_paths))

        fg_sampler = list(zip(fg_images, fg_image_labels))

        for ix in tqdm(range(1, no_images + 1)):
            # Sample `foreground_num` number of images.
            fg_samples = random.choices(fg_sampler, num_foreground)
            image = SuperimposedImage(seq, fg_samples, next(backgrounds))
            image.superimposed_image.convert('RGB').save(os.path.join(output_dir, f'{ix}.jpg'))

            logger.info('Populating label file with bounding box data.')
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
