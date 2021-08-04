# Copyright (c) 2021 Timothy Wolff-Piggott, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""This module defines a custom "transformer_pretrained" model flavor for MLflow."""

import importlib
import logging
import os
import posixpath
import shutil
import sys
from enum import Enum
from typing import Any, Dict, Union

import cloudpickle
import mlflow.pyfunc.utils as pyfunc_utils
import numpy as np
import pandas as pd
import seldon_core
import torch
import transformers
import yaml
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.pytorch import pickle_module as mlflow_pytorch_pickle_module
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir, _copy_file_or_tree
from mlflow.utils.model_utils import _get_flavor_configuration


FLAVOR_NAME = "transformer_pretrained"

_PICKLE_MODULE_INFO_FILE_NAME = "pickle_module_info.txt"
_EXTRA_FILES_KEY = "extra_files"
_REQUIREMENTS_FILE_KEY = "requirements_file"

_logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Model type."""

    SEQUENCE_CLASSIFICATION = 0
    SEQ_2_SEQ = 1


def get_default_conda_env():
    """TODO: add docstring.

    Returns:
        The default Conda environment as a dictionary for MLflow Models produced by calls to
             :func:`save_model` and :func:`log_model`.
    """
    return _mlflow_conda_env(
        additional_pip_deps=[
            "torch=={}".format(torch.__version__),
            "transformers=={}".format(transformers.__version__),
            # We include CloudPickle in the default environment because
            # it's required by the default pickle module used by `save_model()`
            # and `log_model()`: `mlflow.pytorch.pickle_module`.
            "cloudpickle=={}".format(cloudpickle.__version__),
        ],
    )


def log_model(
    transformer_model,
    tokenizer,
    artifact_path,
    conda_env=None,
    code_paths=None,
    pickle_module=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    requirements_file=None,
    extra_files=None,
    model_type: ModelType = ModelType.SEQUENCE_CLASSIFICATION,
    **kwargs,
):
    """TODO: add docstring."""
    # TODO update when pickle_module can be passed as an arg in save_pretrained
    # https://github.com/huggingface/transformers/blob/4b919657313103f1ee903e32a9213b48e6433afe/src/transformers/modeling_utils.py#L784
    if pickle_module is not None:
        _logger.warning(
            "Cannot currently pass pickle_module to save_pretrained;"
            " using mlflow_pytorch_pickle_module"
        )
    pickle_module = mlflow_pytorch_pickle_module
    Model.log(
        artifact_path=artifact_path,
        # flavor=mlflow.transformers,
        flavor=sys.modules[__name__],
        transformer_model=transformer_model,
        tokenizer=tokenizer,
        conda_env=conda_env,
        code_paths=code_paths,
        pickle_module=pickle_module,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        requirements_file=requirements_file,
        extra_files=extra_files,
        model_type=model_type,
        **kwargs,
    )


def save_model(
    transformer_model,
    tokenizer,
    path,
    conda_env=None,
    mlflow_model=None,
    code_paths=None,
    pickle_module=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    requirements_file=None,
    extra_files=None,
    # **kwargs,
):
    """TODO: add docstring."""
    # TODO update when pickle_module can be passed as an arg in save_pretrained
    # https://github.com/huggingface/transformers/blob/4b919657313103f1ee903e32a9213b48e6433afe/src/transformers/modeling_utils.py#L784
    if pickle_module is not None:
        _logger.warning(
            "Cannot currently pass pickle_module to save_pretrained;"
            " using mlflow_pytorch_pickle_module"
        )
    pickle_module = mlflow_pytorch_pickle_module

    if not isinstance(transformer_model, transformers.PreTrainedModel):
        raise TypeError("Argument 'transformer_model' should be a transformers.PreTrainedModel")
    if code_paths is not None:
        if not isinstance(code_paths, list):
            raise TypeError(
                "Argument code_paths should be a list, not {}".format(type(code_paths))
            )
    path = os.path.abspath(path)
    if os.path.exists(path):
        raise RuntimeError("Path '{}' already exists".format(path))

    if mlflow_model is None:
        mlflow_model = Model()

    os.makedirs(path)
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    model_data_subpath = "data"
    model_data_path = os.path.join(path, model_data_subpath)
    os.makedirs(model_data_path)
    # Persist the pickle module name as a file in the model's `data` directory. This is necessary
    # because the `data` directory is the only available parameter to `_load_pyfunc`, and it
    # does not contain the MLmodel configuration; therefore, it is not sufficient to place
    # the module name in the MLmodel
    #
    # TODO: Stop persisting this information to the filesystem once we have a mechanism for
    # supplying the MLmodel configuration to `mlflow.transformers._load_pyfunc`
    pickle_module_path = os.path.join(model_data_path, _PICKLE_MODULE_INFO_FILE_NAME)
    with open(pickle_module_path, "w") as f:
        f.write(pickle_module.__name__)
    # Save transformer model
    transformer_model.save_pretrained(model_data_path)
    # save tokenizer
    tokenizer.save_pretrained(model_data_path)

    torchserve_artifacts_config: Dict[str, Any] = {}

    if requirements_file:
        if not isinstance(requirements_file, str):
            raise TypeError("Path to requirements file should be a string")

        with TempDir() as tmp_requirements_dir:
            _download_artifact_from_uri(
                artifact_uri=requirements_file, output_path=tmp_requirements_dir.path()
            )
            rel_path = os.path.basename(requirements_file)
            torchserve_artifacts_config[_REQUIREMENTS_FILE_KEY] = {"path": rel_path}
            shutil.move(tmp_requirements_dir.path(rel_path), path)

    if extra_files:
        torchserve_artifacts_config[_EXTRA_FILES_KEY] = []
        if not isinstance(extra_files, list):
            raise TypeError("Extra files argument should be a list")

        with TempDir() as tmp_extra_files_dir:
            for extra_file in extra_files:
                _download_artifact_from_uri(
                    artifact_uri=extra_file, output_path=tmp_extra_files_dir.path()
                )
                rel_path = posixpath.join(
                    _EXTRA_FILES_KEY,
                    os.path.basename(extra_file),
                )
                torchserve_artifacts_config[_EXTRA_FILES_KEY].append({"path": rel_path})
            shutil.move(
                tmp_extra_files_dir.path(),
                posixpath.join(path, _EXTRA_FILES_KEY),
            )

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    code_dir_subpath = None
    if code_paths is not None:
        code_dir_subpath = "code"
        for code_path in code_paths:
            _copy_file_or_tree(src=code_path, dst=path, dst_dir=code_dir_subpath)

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        model_data=model_data_subpath,
        pytorch_version=torch.__version__,
        transformers_version=transformers.__version__,
        **torchserve_artifacts_config,
    )
    pyfunc.add_to_model(
        mlflow_model,
        # loader_module="mlflow.transformers",
        loader_module="mlflow_transformers_flavor",
        data=model_data_subpath,
        pickle_module_name=pickle_module.__name__,
        code=code_dir_subpath,
        env=conda_env_subpath,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def _load_model(path, model_type: ModelType = ModelType.SEQUENCE_CLASSIFICATION, **kwargs):
    """TODO: add docstring.

    Args:
        path: The path to a serialized PyTorch model.
        model_type: Type of the model.
        kwargs: Additional kwargs to pass to the PyTorch ``torch.load`` function.
    """
    if not os.path.isdir(path):
        raise ValueError(
            "transformers.AutoModelForSequenceClassification.from_pretrained"
            " should be called with a path to a model directory."
        )

    # `path` is a directory containing a serialized PyTorch model and a text file containing
    # information about the pickle module that should be used by PyTorch to load it
    pickle_module_path = os.path.join(path, _PICKLE_MODULE_INFO_FILE_NAME)
    with open(pickle_module_path, "r") as f:
        pickle_module_name = f.read()
    if "pickle_module" in kwargs and kwargs["pickle_module"].__name__ != pickle_module_name:
        _logger.warning(
            "Attempting to load the PyTorch model with a pickle module, '%s', that does not"
            " match the pickle module that was used to save the model: '%s'.",
            kwargs["pickle_module"].__name__,
            pickle_module_name,
        )
    else:
        try:
            kwargs["pickle_module"] = importlib.import_module(pickle_module_name)
        except ImportError as exc:
            raise MlflowException(
                message=(
                    "Failed to import the pickle module that was used to save the PyTorch"
                    " model. Pickle module name: `{pickle_module_name}`".format(
                        pickle_module_name=pickle_module_name
                    )
                ),
                error_code=RESOURCE_DOES_NOT_EXIST,
            ) from exc

    if model_type == ModelType.SEQUENCE_CLASSIFICATION:
        # pylint: disable=no-value-for-parameter
        model = transformers.AutoModelForSequenceClassification.from_pretrained(path)
    elif model_type == ModelType.SEQ_2_SEQ:
        # pylint: disable=no-value-for-parameter
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(path)
    else:
        raise ValueError(f"Invalid 'model_type'! model_type: {model_type}")

    return model


def _load_tokenizer(path, **kwargs):
    """TODO: add docstring."""
    if not os.path.isdir(path):
        raise ValueError(
            "transformers.AutoTokenizer.from_pretrained"
            " should be called with a path to a model directory."
        )

    return transformers.AutoTokenizer.from_pretrained(path, **kwargs)


def load_model(model_uri, model_type: ModelType = ModelType.SEQUENCE_CLASSIFICATION, **kwargs):
    """TODO: add docstring."""
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    try:
        pyfunc_conf = _get_flavor_configuration(
            model_path=local_model_path, flavor_name=pyfunc.FLAVOR_NAME
        )
    except MlflowException:
        pyfunc_conf = {}
    code_subpath = pyfunc_conf.get(pyfunc.CODE)
    if code_subpath is not None:
        pyfunc_utils._add_code_to_system_path(
            code_path=os.path.join(local_model_path, code_subpath)
        )

    transformers_conf = _get_flavor_configuration(
        model_path=local_model_path, flavor_name=FLAVOR_NAME
    )
    if torch.__version__ != transformers_conf["pytorch_version"]:
        _logger.warning(
            "Stored model version '%s' does not match installed PyTorch version '%s'",
            transformers_conf["pytorch_version"],
            torch.__version__,
        )
    if transformers.__version__ != transformers_conf["transformers_version"]:
        _logger.warning(
            "Stored model version '%s' does not match installed transformers version '%s'",
            transformers_conf["transformers_version"],
            transformers.__version__,
        )
    transformers_model_artifacts_path = os.path.join(
        local_model_path, transformers_conf["model_data"]
    )
    return _load_model(path=transformers_model_artifacts_path, model_type=model_type, **kwargs)


def mlflow_transformers_flavor(
    path="/mnt/models", model_type: ModelType = ModelType.SEQUENCE_CLASSIFICATION, **kwargs
):
    """TODO: add docstring."""
    model_path = os.path.join(seldon_core.Storage.download(path))
    return _load_pyfunc(model_path, model_type=model_type, **kwargs)


def _load_pyfunc(path, model_type: ModelType = ModelType.SEQUENCE_CLASSIFICATION, **kwargs):
    """Load PyFunc implementation.

    Called by ``pyfunc.load_pyfunc``.

    Args:
        path: Local filesystem path to the MLflow Model with the
            ``transformer_pretrained`` flavor.
        model_type: Model type.
    """
    model_wrapper: Union[_TransformerSequenceClassificationWrapper, _TransformerSeq2SeqWrapper]
    if model_type == ModelType.SEQUENCE_CLASSIFICATION:
        model_wrapper = _TransformerSequenceClassificationWrapper(
            _load_model(path, model_type=model_type, **kwargs), _load_tokenizer(path, **kwargs)
        )
    elif model_type == ModelType.SEQ_2_SEQ:
        model_wrapper = _TransformerSeq2SeqWrapper(
            _load_model(path, model_type=model_type, **kwargs), _load_tokenizer(path, **kwargs)
        )
    else:
        raise ValueError(f"Invalid 'model_type'! model_type: {model_type}")

    return model_wrapper


class _TransformerSeq2SeqWrapper:
    def __init__(self, transformer_model, tokenizer):
        self.tokenizer = tokenizer
        self.transformer_model = transformer_model

    def predict(self, data: np.ndarray, dev="cpu"):
        """TODO: add docstring.

        Args:
            data: The input text texts. Must be a 1d numpy array.
            dev: This argument will not be taken into account.
        """
        # Seldon for some reason passes [] instead of cpu.
        # So ignoring dev and just hardcoding "cpu"
        del dev
        assert isinstance(data, np.ndarray)
        assert data.ndim == 1
        assert data.size > 0

        results = []

        for text in data:
            result = ""
            if (text is not None) and (len(text) > 0):
                text = text.strip().replace("\n", " ")
                text = "summarize: " + text
                device = torch.device("cpu")
                tokenized_text = self.tokenizer.encode(text, return_tensors="pt").to(device)
                # summarize
                summary_ids = self.transformer_model.generate(
                    tokenized_text,
                    num_beams=3,
                    no_repeat_ngram_size=2,
                    min_length=30,
                    max_length=100,
                    early_stopping=True,
                )
                result = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            results.append(result)

        return results


class _TransformerSequenceClassificationWrapper:
    """TODO: add docstring."""

    class ListDataset(torch.utils.data.Dataset):
        """TODO: add docstring."""

        def __init__(self, data):
            self.data = data

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    class Collate:
        """TODO: add docstring."""

        def __init__(self, tokenizer, device):
            self.tokenizer = tokenizer
            self.device = device

        def __call__(self, batch):
            # batch is a list of str with length of batch_size from DataLoader
            encoding = self.tokenizer(
                batch,
                return_token_type_ids=False,
                return_tensors="pt",
                truncation=True,
                padding="longest",
            )

            # taken from here: https://github.com/huggingface/transformers/blob/
            # 34fcfb44e30284186ece3f4ac478c1e6444eb0c7/src/transformers/pipelines.py#L605
            encoding = {name: tensor.to(self.device) for name, tensor in encoding.items()}
            return encoding

    def __init__(self, transformer_model, tokenizer):
        self.tokenizer = tokenizer
        self.transformer_model = transformer_model

    def predict(self, data, dev="cpu"):
        """TODO: add docstring."""
        # Seldon for some reason passes [] instead of cpu.
        # So ignoring dev and just hardcoding "cpu"
        del dev
        device = torch.device("cpu")
        collate = self.Collate(tokenizer=self.tokenizer, device=device)

        if isinstance(data, pd.DataFrame):
            inp_data = data.values.ravel()
        elif isinstance(data, np.ndarray):
            inp_data = data
        elif isinstance(data, (list, dict)):
            raise TypeError(
                "The MLflow python_function format does not support List or Dict input types. "
                "Please use a pandas.DataFrame or a numpy.ndarray"
            )
        else:
            raise TypeError("Input data should be pandas.DataFrame or numpy.ndarray")

        if inp_data.ndim != 1:
            raise ValueError("Input data should have ndim == 1 after casting to numpy")
        if inp_data.size == 0:
            raise ValueError("Input data should be non-empty")

        data_loader = torch.utils.data.DataLoader(
            dataset=self.ListDataset(inp_data),
            batch_size=10,
            shuffle=False,
            collate_fn=collate,
            pin_memory=False,  # TODO: should this be True with GPU usage?
        )

        self.transformer_model.eval()
        self.transformer_model.to(device)

        results = []

        with torch.no_grad():
            for d in data_loader:
                outputs = self.transformer_model(**d)[0].cpu().detach().numpy()

                # since we only want to know whick output is the highest
                # we can skip the sigmoid or softmax function
                # we can use the model outputs directly
                predictions = [
                    self.transformer_model.config.id2label[item.argmax()] for item in outputs
                ]
                results.extend(predictions)
            if isinstance(data, pd.DataFrame):
                predicted = pd.DataFrame(results)
                predicted.index = data.index
            else:
                predicted = np.array(results)

        return predicted
