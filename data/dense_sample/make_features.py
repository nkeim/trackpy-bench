# Create HDF5 file of features, for benchmarking linking.

import trackpy as tp
import pims

images = pims.ImageSequence('*.png', as_grey=True)[:5]

features = tp.batch(images, 3, minmass=0, invert=True, percentile=1)

features.to_hdf('features.h5', 'features')
