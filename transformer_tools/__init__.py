# Copyright (c) 2021 Philip May
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Transformer-Tools main package."""

from transformer_tools.transformers import LabeledDataset


# Versioning follows the `Semantic Versioning Specification <https://semver.org/>`__ and
# `PEP 440 -- Version Identification and Dependency Specification <https://www.python.org/dev/peps/pep-0440/>`__.  # noqa: E501
__version__ = "0.1.1rc1"

__all__ = ["LabeledDataset", "__version__"]