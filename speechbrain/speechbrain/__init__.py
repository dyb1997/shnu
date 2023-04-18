""" Comprehensive speech processing toolkit
"""
import os
from .core import Stage, Brain, create_experiment_directory, parse_arguments
from .core1 import Stage, Brain, create_experiment_directory, parse_arguments
from .core1cutmix import Stage, Brain, create_experiment_directory, parse_arguments
from .core1stage import Stage, Brain, create_experiment_directory, parse_arguments
from .core4 import Stage, Brain, create_experiment_directory, parse_arguments
from .core7 import Stage, Brain, create_experiment_directory, parse_arguments
from .core22 import Stage, Brain, create_experiment_directory, parse_arguments
from .core23 import Stage, Brain, create_experiment_directory, parse_arguments
from .corese import Stage, Brain, create_experiment_directory, parse_arguments
from .corese2 import Stage, Brain, create_experiment_directory, parse_arguments
from .corese3 import Stage, Brain, create_experiment_directory, parse_arguments
from .corese4 import Stage, Brain, create_experiment_directory, parse_arguments
from .coreCL import Stage, Brain, create_experiment_directory, parse_arguments
from .coreCLtd import Stage, Brain, create_experiment_directory, parse_arguments
from .coreBYOL import Stage, Brain, create_experiment_directory, parse_arguments
from .coreBYOLTSMFA import Stage, Brain, create_experiment_directory, parse_arguments
from .coreBYOLTS import Stage, Brain, create_experiment_directory, parse_arguments
from .coreCL1 import Stage, Brain, create_experiment_directory, parse_arguments
from .coreCLtd1 import Stage, Brain, create_experiment_directory, parse_arguments
from .corefinetuning import Stage, Brain, create_experiment_directory, parse_arguments
from .coreseunsup import Stage, Brain, create_experiment_directory, parse_arguments
from .coreonlyaug import Stage, Brain, create_experiment_directory, parse_arguments
#from .core1class import Stage, Brain, create_experiment_directory, parse_arguments
from .coreAP import Stage, Brain, create_experiment_directory, parse_arguments
from . import alignment  # noqa
from . import dataio  # noqa
from . import decoders  # noqa
from . import lobes  # noqa
from . import lm  # noqa
from . import nnet  # noqa
from . import processing  # noqa
from . import tokenizers  # noqa
from . import utils  # noqa

with open(os.path.join(os.path.dirname(__file__), "version.txt")) as f:
    version = f.read().strip()

__all__ = [
    "Stage",
    "Brain",
    "create_experiment_directory",
    "parse_arguments",
]

__version__ = version
