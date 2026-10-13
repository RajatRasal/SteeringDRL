"""Debug file loading params and models"""

import os
import sys
import importlib

import torch

from spatial_disentanglement.nerv.models import build_model
from spatial_disentanglement.nerv.method import build_method
from spatial_disentanglement.nerv.datasets import build_dataset


def load_params(params):
    if params.endswith('.py'):
        params = params[:-3]
    sys.path.append(os.path.dirname(params))
    params = importlib.import_module(os.path.basename(params))
    params = params.BaseParams()
    model = build_model(params)
    return params, model
