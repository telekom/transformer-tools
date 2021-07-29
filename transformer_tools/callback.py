# Copyright (c) 2021 Philip May, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Transformers callbacks.

Also see :class:`transformers.TrainerCallback`.
"""

import logging
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from ml_cloud_tools.s3 import copy_dir_to_s3_dir, copy_s3_dir_to_dir, list_s3_files
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint


_logger = logging.getLogger(__name__)
PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"\/" + PREFIX_CHECKPOINT_DIR + r"-(\d+)\/")


def _best_checkpoint_number(files: List[str]):
    best_cp_number = -1
    for file in files:
        cp_search = _re_checkpoint.search(file)
        if cp_search is not None:
            cp_number = int(cp_search.groups()[0])
            if cp_number > best_cp_number:
                best_cp_number = cp_number
    return best_cp_number if best_cp_number > -1 else None


class S3CheckpointSyncCallback(TrainerCallback):
    """Save and recover checkpoints by using S3."""

    def __init__(
        self,
        s3_dir_name: str,
        s3_bucket_name: Optional[str] = None,
        s3_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.s3_dir_name = s3_dir_name
        self.s3_bucket_name = s3_bucket_name
        self.s3_kwargs = s3_kwargs

    def on_init_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """Event called at the end of the initialization of the :class:`~transformers.Trainer`."""
        source_s3_dir_path = Path(self.s3_dir_name) / Path(args.output_dir).name
        files_on_s3 = list_s3_files(
            source_s3_dir_path.as_posix(), s3_bucket_name=self.s3_bucket_name
        )
        if len(files_on_s3) > 0:
            checkpoint_number = _best_checkpoint_number(files_on_s3)
            if checkpoint_number is not None:
                source_s3_dir_path = source_s3_dir_path / Path(
                    f"{PREFIX_CHECKPOINT_DIR}-{checkpoint_number}"
                )
                target_dir_path = Path(args.output_dir)
                target_dir_path.mkdir(parents=True, exist_ok=True)
                copy_s3_dir_to_dir(
                    source_s3_dir_path.as_posix(),
                    target_dir_path.as_posix(),
                    s3_bucket_name=self.s3_bucket_name,
                    s3_kwargs=self.s3_kwargs,
                )

    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """Event called after a checkpoint save."""
        last_checkpoint_dir = get_last_checkpoint(args.output_dir)
        if last_checkpoint_dir is not None:
            source_dir_path = Path(last_checkpoint_dir)
            target_s3_dir_path = Path(self.s3_dir_name) / Path(args.output_dir).name
            copy_dir_to_s3_dir(
                source_dir_path.as_posix(),
                target_s3_dir_path.as_posix(),
                s3_bucket_name=self.s3_bucket_name,
                s3_kwargs=self.s3_kwargs,
            )

        # remove local checkpoint directory
        try:
            shutil.rmtree(args.output_dir)
        except Exception as e:
            error_msg = "Exception raised while deleting checkpoints! Exception: {}".format(e)
            _logger.error(error_msg, exc_info=True)
            warnings.warn(error_msg, RuntimeWarning)
