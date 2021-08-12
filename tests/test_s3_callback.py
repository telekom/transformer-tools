# Copyright (c) 2021 Philip May, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

from transformer_tools.s3_callback import _best_checkpoint_number


def test_best_checkpoint_number_happy_case():
    checkpoint_dirs = [
        "something/checkpoints/checkpoint-100/something-else1.txt",
        "something/checkpoints/checkpoint-100/something-else2.txt",
        "something/checkpoints/checkpoint-101/something-else.txt",
        "something/checkpoints/checkpoint-10000001/something-else1.txt",
        "something/checkpoints/checkpoint-10000001/something-else2.txt",
    ]

    checkpoint_number = _best_checkpoint_number(checkpoint_dirs)

    assert checkpoint_number == 10000001


def test_best_checkpoint_number_no_valid_checkpoints():
    checkpoint_dirs = [
        "something/checkpoints/checkpoint-/something-else1.txt",
        "something/checkpoints/something-else2.txt",
    ]

    checkpoint_number = _best_checkpoint_number(checkpoint_dirs)

    assert checkpoint_number is None


def test_best_checkpoint_number_mixed_case():
    checkpoint_dirs = [
        "something/checkpoints/checkpoint-/something-else1.txt",
        "something/checkpoints/something-else2.txt",
        "something/checkpoints/checkpoint-101/something-else.txt",
        "something/checkpoints/checkpoint-10000001/something-else1.txt",
        "something/checkpoints/checkpoint-10000001/something-else2.txt",
    ]

    checkpoint_number = _best_checkpoint_number(checkpoint_dirs)

    assert checkpoint_number == 10000001
