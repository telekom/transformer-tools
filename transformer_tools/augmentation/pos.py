# Copyright (c) 2021 Philip May
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""POS tagging."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import spacy


@dataclass
class PosToken:
    """A POS token."""

    text: str
    pos_type: str


class SpacyPosTagger:
    """paCy POS tagger.

    To use a spaCy pipeline you must install it first.
    Use: `python -m spacy download de_core_news_sm` for example.
    """

    def __init__(self, spacy_pipeline: Union[str, Path]):
        # see https://spacy.io/api/top-level#spacy.load
        # TODO: maybe disable some pipeline components to speed up POS tagging
        self._nlp = spacy.load(spacy_pipeline)

    def tag_text(self, text) -> List[PosToken]:
        """Do POS tagging with text."""
        doc = self._nlp(text)
        result = []
        for token in doc:
            # print(t.text, "<->", t.whitespace_ == " ", t.pos_)
            result.append(PosToken(token.text, token.pos_))
            if token.whitespace_ != "":
                result.append(PosToken(" ", "SPACE"))
        return result
