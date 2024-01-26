import copy


class HyperParameters:
    """
    Kind of a useless class for the moment as it just represents a dict.
    An instance of parameters needs to be passed to the optimizer

    This class will be used in the future to improve the parameters and do some parameters optimization
    and analysis on the parameters. It might even allow to do AutoML in the future.

    THIS CLASS IS IMMUTABLE !!!! TRUST ME!
    """

    def __init__(self, **kwargs):
        self._params = copy.deepcopy(kwargs)

    def __getattr__(self, item):
        if item in self._params:
            return self._params[item]
        else:
            raise AttributeError("'HyperParameters' object has no attribute " + str(item))

