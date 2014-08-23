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
            os.path.join(datadir, 'bulk_water', '*.png'), as_grey=True)[:20])


class DenseConfiguration(object):
    def configure(self):
        self.diameter = 3
        self.feature_base_args = {'minmass': 0, 'invert': True, 'percentile': 1}
        self.search_range = 4
        self.search_range_large = 4  # Sure to trigger subnet code
        self.linking_base_args = {}

        # Load images into memory
        self.images = list(pims.ImageSequence(
            os.path.join(datadir, 'dense_sample', '*.png'), as_grey=True)[:5])


class _SuiteBase(object):
    """Perform feature-finding and linking on the "bulk_water" sample movie."""

    timeout = 180

    @classmethod
    def run(cls, method_name):
        """Run a single benchmark"""
        sb = cls()
        sb.setup()
        getattr(sb, method_name)()

    def setup(self):
        self.configure()

    def setup_features(self):
        # Warm up FFTW and numba
        _ = tp.batch(self.images[:1], self.diameter, **self.feature_base_args)

    def find_features(self, **kwargs):
        opts = self.feature_base_args.copy()
        opts.update(kwargs)
        return tp.batch(self.images, self.diameter, **opts)

    def setup_linking(self):
        self.features = self.find_features()
        # Warm up numba by using a large search range,
        # to ensure that subnet code will be compiled.
        _ = tp.link_df(self.features[self.features.frame < 3],
                       self.search_range_large, **self.linking_base_args)

    def link(self, **kwargs):
        opts = self.linking_base_args.copy()
        opts.update(kwargs)
        return tp.link_df(self.features, self.search_range, **opts)


class _FeatureBase(_SuiteBase):
    def setup(self):
        super(_FeatureBase, self).setup()
        self.setup_features()

    def time_numba(self):
        self.find_features(engine='numba')


class _LinkingBase(_SuiteBase):
    def setup(self):
        super(_LinkingBase, self).setup()
        self.setup_linking()

    def time_numba(self):
        self.link()


###############
# Actual suites

class WaterFeatureSuite(_FeatureBase, WaterConfiguration):
    def time_python(self):
        self.find_features(engine='python')


class WaterLinkingSuite(_LinkingBase, WaterConfiguration):
    def time_python(self):
        self.link(link_strategy='recursive')


class DenseFeatureSuite(_FeatureBase, DenseConfiguration):
    pass  # Numba only


class DenseLinkingSuite(_LinkingBase, DenseConfiguration):
    pass  # Numba only


# For debugging
if __name__ == '__main__':
    DenseLinkingSuite.run('time_numba')
