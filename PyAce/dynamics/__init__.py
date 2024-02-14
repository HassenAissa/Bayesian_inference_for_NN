from dynamics.control import NNPolicyOptimizer
from dynamics.control import Policy
from dynamics.deep_pilco import NNPolicy
from dynamics.deep_pilco import BayesianDynamics
from dynamics.simple_policy import Policy


from os.path import dirname, basename, isfile, join
import glob
modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]