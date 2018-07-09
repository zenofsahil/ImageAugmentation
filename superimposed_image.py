import os
import numpy as np
import random
from PIL import Image
from foreground_image import ForegroundImage

class SuperimposedImage(object):
    """
    Wrapper class containing the pipeline of foreground images to be augmented, 
    the background image path that the augmented foreground images are to be superimposed on
    and methods to perform the superimposition.

    Each background image and a set of newly augmented foreground images correspond to a 
    SuperimposedImage class.
    """

    #def __init__(self, foreground_pipeline, foreground_num, background_path):
    def __init__(self, pipeline, fg_samples, background_path):
        """
        An instance of SuperimposedImage class.

        :param foreground_pipeline: Pipeline generator object instantiated with foreground images.
        :param foreground_num: Number of foreground images to be superimposed onto the background.
        :param background_path: Path of the background image.

        """
        self.background_path = background_path
        self.background_image = Image.open(self.background_path)
        self.background_height = self.background_image.height
        self.background_width = self.background_image.width

        self.pipeline = pipeline
        self.foreground_num = len(fg_samples)
        # self.foreground_generator = self.foreground_pipeline.keras_generator(
        #         batch_size=foreground_num, scaled=False)

        # self.foreground_images, self.foreground_labels = next(
        #         self.foreground_generator)

        #self.foreground_images, self.foreground_labels = self.augmented_images()
        self.foreground_images = self.populate_foreground_images(fg_samples)
        # self.foreground_images = []
        # self.populate_foreground_images()


        # self.class_labels = {v : k for k, v in dict(self.foreground_pipeline.class_labels).items()}
        # self.foreground_labels = [self.class_labels[np.argmax(lab)] for lab in self.foreground_labels]

        # Convert foreground images that are numpy arrays to PIL images
        # and create named tuples to hold co-ordinate data.
        # ForegroundImage = namedtuple(
        #         'ForegroundImage',
        #         'image, label, xmin, ymin, xmax, ymax')

        # pil_foreground_images = []

        # for image, label in zip(self.foreground_images, self.foreground_labels):
        #     image = Image.fromarray(image)
        #     image = ForegroundImage(image, label, 0, 0, 0, 0)
        #     pil_foreground_images.append(image)

        # self.foreground_images = pil_foreground_images

        self.superimposed_image = self.background_image.copy()

        self.process_image()

    def populate_foreground_images(self, image_samples):
        """
        Read, (augment) and populate the foreground images."
        """

        ### Apply augmentations to the images ###

        images = list(map(lambda x: np.array(x[0]), image_samples))
        labels = list(map(lambda x: x[1], image_samples))

        augmented_images = self.pipeline.augment_images(images)
        augmented_images = list(map(
            lambda x: Image.fromarray(x).convert('RGBA'), augmented_images))

        foreground_images = []
        for image, label in zip(augmented_images, labels):
            foreground_image = ForegroundImage(image, label, 0, 0, 0, 0, 0)
            foreground_images.append(foreground_image)

        return foreground_images

    def process_image(self):
        """ 
        Randomly resize each augmented foreground image and paste it onto a 
        random (x, y) coordinate on the background image.
        """

        new_fg_images = []
        for fg_image in self.foreground_images:
            fg_image.image = self.random_resize(fg_image.image)

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
