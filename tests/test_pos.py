# Copyright (c) 2021 Philip May
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

from transformer_tools.augmentation.pos import SpacyPosTagger


def test_tag_text():
    spt = SpacyPosTagger("de_core_news_sm")
    text = "Das ist ein Test!"
    tagged_text = spt.tag_text(text)

    assert len(tagged_text) == 5 + 3  # 5 token and 3 spaces
    assert "".join([t.text for t in tagged_text]) == text
