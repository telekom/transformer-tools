# Copyright (c) 2021 Philip May, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Transformers callbacks.

Also see :class:`transformers.TrainerCallback`.
"""

import logging
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

from ml_cloud_tools.s3 import copy_dir_to_s3_dir, copy_s3_dir_to_dir, list_s3_files
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint


_logger = logging.getLogger(__name__)


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
        # TODO: only copy last CP
        source_s3_dir_path = Path(self.s3_dir_name) / Path(args.output_dir).name
        files_on_s3 = list_s3_files(
            source_s3_dir_path.as_posix(), s3_bucket_name=self.s3_bucket_name
        )
        if len(files_on_s3) > 0:
            target_dir_path = Path(args.output_dir).resolve().parent
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
            target_s3_dir_path = Path(self.s3_dir_name) / source_dir_path.name
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
