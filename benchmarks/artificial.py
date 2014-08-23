# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import numpy as np
import pims
import trackpy as tp


class TimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """
    timeout = 180

    @classmethod
    def run(cls, method_name):
        """Run a single benchmark"""
        sb = cls()
        sb.setup()
        getattr(sb, method_name)()

    def setup(self):
        shape = (400, 401)
        r = 3
        noise_level = 0.01
        cases = {'sparse': 10, 'dense': 10000}
        for case_name, count in cases.items():
            locations = gen_random_locations(shape, count)
            image = draw_spots(shape, locations, r, noise_level)
            setattr(self, '{0}_image'.format(case_name), image)

        # Prime FFTW (if it's there).
        tp.locate(self.sparse_image, 7)

    def time_locate_sparse(self):
        tp.locate(self.sparse_image, 7)

    def time_locate_dense(self):
        tp.locate(self.dense_image, 7)

#
# Code below is copied from trackpy/tests/test_feature.py
# That code is not importable from trackpy. Since benchmarks run on the code
# history, it will not help to make it importable now.
#

def draw_gaussian_spot(image, pos, r, max_value=None, ecc=0):
    if image.shape[0] == image.shape[1]:
        raise ValueError("For stupid numpy broadcasting reasons, don't make" +
                         "the image square.")
    ndim = image.ndim
    pos = maybe_permute_position(pos)
    coords = np.meshgrid(*np.array(map(np.arange, image.shape)) - pos,
                         indexing='ij')
    if max_value is None:
        max_value = np.iinfo(image.dtype).max - 3
    if ndim == 2:
        # Special case for 2D: implement eccentricity.
        y, x = coords
        spot = max_value*np.exp(
            -((x / (1 - ecc))**2 + (y * (1 - ecc))**2)/(2*r**2))
    else:
        if ecc != 0:
            raise ValueError("Eccentricity must be 0 if image is not 2D.")
        coords = np.asarray(coords)
        spot = max_value*np.exp(-np.sum(coords**2, 0)/(ndim*r**2))
    image += spot.astype(image.dtype)

def gen_random_locations(shape, count):
    np.random.seed(0)
    shape = np.asarray(shape)[::-1]  # TODO
    return np.array([map(np.random.randint, shape) for _ in xrange(count)])


def draw_spots(shape, locations, r, noise_level):
    np.random.seed(0)
    image = np.random.randint(0, 1 + noise_level, shape).astype('uint8')
    for x in locations:
        draw_gaussian_spot(image, x, r)
    return image

def maybe_permute_position(pos):
    ndim = len(pos)
    if ndim == 2:
        pos = np.asarray(pos)[[1, 0]]
    elif ndim == 3:
        pos = np.asarray(pos)[[2, 1, 0]]
    return pos

if __name__ == '__main__':
    TimeSuite.run('time_locate_sparse')
