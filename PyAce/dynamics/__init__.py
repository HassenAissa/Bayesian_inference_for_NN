from .control import NNPolicyOptimizer
from .control import Policy
from .deep_pilco import NNPolicy
from .deep_pilco import BayesianDynamics


from os.path import dirname, basename, isfile, join
import glob
modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]