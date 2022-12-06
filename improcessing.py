import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

from skimage.transform import rescale as sk_rescale

class ImageProcessing:
    def __init__(self, img_path: str) -> None:
        '''
        Receive image path as argument, open image as array and
        stores it as instance attribute.
        '''

        self.image = imread(img_path)
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]

    def apply_filter(self, filter: np.array or list, display: bool = True) -> np.array:
        '''
        Apply filter in the 3 channels and return the result as an array
        '''

        # Convert filter to array if is list
        if isinstance(filter, list):
            filter = np.array(filter)

        filter_w = filter.shape[1]  # Filter width
        filter_h = filter.shape[0]  # Filter height

        horizontal_steps = self.image.shape[1] - filter_w + 1
        vertical_steps = self.image.shape[0] - filter_h + 1

        result = []
        for channel_idx in range(3):
            channel_result = []
            for vs in range(vertical_steps):
                line = []
                for hs in range(horizontal_steps):
                    sub_matrix = self.image[vs:(
                        vs + filter_h), hs:(hs + filter_w), channel_idx:(channel_idx+1)]
                    sub_matrix = sub_matrix.reshape(filter_h, filter_w)
                    dot_value = np.multiply(sub_matrix, filter)
                    line.append(dot_value.sum())
                channel_result.append(line)
            result.append(channel_result)

        joined_channels = np.dstack(np.array(result))

        if display:
            self.show(joined_channels)

        return joined_channels

    def show(self, image_array: np.array = None) -> None:
        '''
        Display image passed as argument or the main image 
        if argument is None.
        '''

        if image_array is None:
            image_array = self.image

        plt.imshow(image_array)

    def rescale(self, factor: float) -> None:
        self.image = sk_rescale(self.image, factor, multichannel=True)

    def blur(self, radius: int) -> np.array:
        '''
        Blur image and displays it, returning result as array
        '''

        res = self.apply_filter([
            [1/(radius**2) for x in range(radius)] for y in range(radius)
        ])

        return res

    def vertical_edges(self) -> np.array:
        res = self.apply_filter([
            [-1,0,1],
            [-1,0,1],
            [-1,0,1],
        ])

        return res

    def horizontal_edges(self) -> np.array:
        res = self.apply_filter([
            [1,1,1],
            [0,0,0],
            [-1,-1,-1]
        ])

        return res

    def brightness(self, value: int) -> np.array:
        '''
        Increase brightness by percentage
        '''
        res = self.image*((100+value)/100)

        self.show(res)

        return res