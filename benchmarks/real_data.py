import os

import trackpy as tp
import pims

path, _ = os.path.split(os.path.abspath(__file__))
datadir = os.path.join(path, '..', 'data')

class WaterConfiguration(object):
    def configure(self):
        self.diameter = 11
        self.feature_base_args = {'minmass': 2000, 'invert': True}
        self.search_range = 5
        self.search_range_large = 10  # Sure to trigger subnet code
        self.linking_base_args = {'memory': 3}

        # Load images into memory
        self.images = list(pims.ImageSequence(
            os.path.join(datadir, 'bulk_water', '*.png'), as_grey=True))


class DenseConfiguration(object):
    def configure(self):
        self.diameter = 3
        self.feature_base_args = {'minmass': 0, 'invert': True, 'percentile': 1}
        self.search_range = 4
        self.search_range_large = 4  # Sure to trigger subnet code
        self.linking_base_args = {}

        # Load images into memory
        self.images = list(pims.ImageSequence(
            os.path.join(datadir, 'dense-sample', '*.tif'), as_grey=True))


class _SuiteBase(object):
    """Perform feature-finding and linking on the "bulk_water" sample movie."""
    def setup(self):
        self.configure()

    def setup_features(self):
        # Warm up FFTW and numba
        _ = tp.batch(self.images[:1], self.diameter, **self.feature_base_args)

    def find_features(self, **kwargs):
        opts = self.feature_base_args.copy().update(kwargs)
        return tp.batch(self.images, self.diameter, **opts)

    def setup_linking(self):
        self.features = self.find_features()
        # Warm up numba by using a large search range,
        # to ensure that subnet code will be compiled.
        _ = tp.link_df(self.features[self.features.frame < 3],
                       self.search_range_large, **self.linking_base_args)

    def link(self, **kwargs):
        opts = self.linking_base_args.copy().update(kwargs)
        return tp.link_df(self.features, self.search_range, **opts)


class _FeatureBase(_SuiteBase):
    def setup(self):
        super(WaterFeatureSuite, self).setup()
        self.setup_features()

    def time_numba(self):
        self.find_features(engine='numba')

    def time_python(self):
        self.find_features(engine='python')


class _LinkingBase(_SuiteBase):
    def setup(self):
        super(WaterLinkingSuite, self).setup()
        self.setup_linking()

    def time_numba(self):
        self.link(link_strategy='numba')

    def time_python(self):
        self.link(link_strategy='recursive')


###############
# Actual suites

class WaterFeatureSuite(_FeatureBase, WaterConfiguration):
    pass


class WaterLinkingSuite(_LinkingBase, WaterConfiguration):
    pass


class DenseFeatureSuite(_FeatureBase, DenseConfiguration):
    pass


class DenseLinkingSuite(_LinkingBase, DenseConfiguration):
    pass
