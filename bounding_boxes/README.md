Bounding Box Augmentation
===

This script produces a synthetic dataset for use in object detection tasks. 

**Usage**
```
usage: augment.py [-h] -f FOREGROUND -b BACKGROUND [-nf NUMFOREGROUND]
                  [-n NOIMAGES] -o OUTPUT -lab LABELFILE

Create augmented dataset with set of foreground images imposed on different
backgrounds.

optional arguments:
  -h, --help            show this help message and exit
  -f FOREGROUND, --foreground FOREGROUND
                        The folder containing the samples of the foreground
                        images.
  -b BACKGROUND, --background BACKGROUND
                        The folder containing background images.
  -nf NUMFOREGROUND, --numforeground NUMFOREGROUND
                        No. of foreground images to superimpose per background
                        image
  -n NOIMAGES, --noimages NOIMAGES
                        No. of augmented images to create per background image
  -o OUTPUT, --output OUTPUT
                        The output path where we will be saving the images to
  -lab LABELFILE, --labelfile LABELFILE
                        The label file containing the coordinate and label
                        data of the images
```

