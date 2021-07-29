# Copyright (c) 2021 Philip May, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

from transformer_tools.callback import _select_best_checkpoint_dir


def test_select_best_checkpoint_dir_happy_case():
    checkpoint_dirs = [
        "something/checkpoints/checkpoint-100/something-else1.txt",
        "something/checkpoints/checkpoint-100/something-else2.txt",
        "something/checkpoints/checkpoint-101/something-else.txt",
        "something/checkpoints/checkpoint-10000001/something-else1.txt",
        "something/checkpoints/checkpoint-10000001/something-else2.txt",
    ]

    best_checkpoint_dir = _select_best_checkpoint_dir(checkpoint_dirs)

    assert best_checkpoint_dir == "something/checkpoints/checkpoint-10000001"


def test_select_best_checkpoint_dir_no_valid_checkpoints():
    checkpoint_dirs = [
        "something/checkpoints/checkpoint-/something-else1.txt",
        "something/checkpoints/something-else2.txt",
    ]

    best_checkpoint_dir = _select_best_checkpoint_dir(checkpoint_dirs)

    assert best_checkpoint_dir is None


def test_select_best_checkpoint_dir_mixed_case():
    checkpoint_dirs = [
        "something/checkpoints/checkpoint-/something-else1.txt",
        "something/checkpoints/something-else2.txt",
        "something/checkpoints/checkpoint-101/something-else.txt",
        "something/checkpoints/checkpoint-10000001/something-else1.txt",
        "something/checkpoints/checkpoint-10000001/something-else2.txt",
    ]

    best_checkpoint_dir = _select_best_checkpoint_dir(checkpoint_dirs)

    assert best_checkpoint_dir == "something/checkpoints/checkpoint-10000001"
