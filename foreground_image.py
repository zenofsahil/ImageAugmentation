

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


