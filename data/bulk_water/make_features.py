# Create HDF5 file of features, for benchmarking linking.

import trackpy as tp
import pims

images = pims.ImageSequence('*.png', as_grey=True)[:20]

features = tp.batch(images, 11, minmass=2000, invert=True)

features.to_hdf('features.h5', 'features')
