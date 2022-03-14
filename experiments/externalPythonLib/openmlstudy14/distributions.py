import numpy as np
from scipy.stats._distn_infrastructure import rv_continuous, rv_discrete
import scipy.stats


class OpenMLDistributionHelper(object):
    def _cdf(self, x, *args):
        raise NotImplementedError()

    def _sf(self, x, *args):
        raise NotImplementedError()

    def _ppf(self, q, *args):
        raise NotImplementedError()

    def _isf(self, q, *args):
        raise NotImplementedError()

    def _stats(self, *args, **kwds):
        raise NotImplementedError()

    def _munp(self, n, *args):
        raise NotImplementedError()

    def _entropy(self, *args):
        raise NotImplementedError()


class loguniform_gen(OpenMLDistributionHelper, rv_continuous):
    def _pdf(self, x, base, low, high):
        raise NotImplementedError()

    def _argcheck(self, base, low, high):
        self.base = base
        self.a = low
        self.b = high
        return (high > low) and low > 0 and high > 0 and base >= 2

    def logspace(self, num):
        start = np.log(self.a) / np.log(self.base)
        stop = np.log(self.b) / np.log(self.base)
        return np.logspace(start, stop, num=num, endpoint=True, base=self.base)

    def _rvs(self, base, low, high):
        low = np.log(low) / np.log(base)
        high = np.log(high) / np.log(base)
        return np.power(self.base,
                        self._random_state.uniform(low=low, high=high,
                                                   size=self._size))
loguniform = loguniform_gen(name='loguniform')


class loguniform_int_gen(OpenMLDistributionHelper, rv_discrete):
    def _pmf(self, x, base, low, high):
        raise NotImplementedError()

    def _argcheck(self, base, low, high):
        self.base = base
        self.a = low
        self.b = high
        return (high > low) and low >= 1 and high >= 1 and base >= 2

    def _rvs(self, base, low, high):
        assert self.a >= 1
        low = np.log(low - 0.4999) / np.log(base)
        high = np.log(high + 0.4999) / np.log(base)
        return np.rint(np.power(base, self._random_state.uniform(
            low=low, high=high, size=self._size))).astype(int)
loguniform_int = loguniform_int_gen(name='loguniform_int')


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    # Assumes that high is the target high + 1
    r = scipy.stats.randint(low=-10, high=11)
    samples = r.rvs(size=10000)
    assert np.max(samples) == 10, np.max(samples)
    assert np.min(samples) == -10

    r = scipy.stats.uniform(loc=0.1, scale=0.8)
    samples = r.rvs(size=1000000)
    assert 0.90001 >= np.max(samples) >= 0.8999, np.max(samples)
    assert 0.09999 <= np.min(samples) <= 0.1001, np.min(samples)

    r = loguniform(base=2, low=2**-12, high=2**12)
    samples = r.rvs(size=1000000)
    assert np.max(samples) <= 4096
    assert np.min(samples) >= 0.0000001

    r = loguniform(base=10, low=1e-7, high=1e-1)
    samples = r.rvs(size=1000000)
    assert np.max(samples) <= 0.1
    assert np.min(samples) >= 0.0000001

    r = loguniform_int(base=2, low=1, high=2**12)
    samples = r.rvs(size=1000000)
    assert np.max(samples) == 4096
    assert np.min(samples) == 1

    r = loguniform_int(base=10, low=1, high=1000)
    samples = r.rvs(size=1000000)
    assert np.max(samples) == 1000
    assert np.min(samples) == 1

    r = loguniform_int(base=300, low=300**0.1, high=300**0.9)
    samples = r.rvs(size=1000000)
    assert np.max(samples) == 170, np.max(samples)
    assert np.min(samples) == 1, np.min(samples)








