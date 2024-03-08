from os.path import dirname, basename, isfile, join
from .DeepPilco import DeepPilco
from .control import Control, Policy
from .deep_pilco import BayesianDynamics, NNPolicy, DynamicsTraining
from .PolicyNN import PolicyNN
from .Policy import Policy
import glob
modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]