# Copyright (c) 2021 Michal Harakal, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""Text utils."""

from somajo import SoMaJo


class BlockSplitter:
    """Splittinglongers texts into a blocks with complete sentences limited by gived character length."""

    def __init__(self, is_german: bool = True) -> None:
        """Constructor.

        Args:
            is_german: Indicate if a german vocabulary has to be loaded for senteces
            splitting with SoMaJo. Otherwise load english.
        """
        self.tokenizer = (
            SoMaJo("de_CMC", split_camel_case=True)
            if is_german
            else SoMaJo(language="en_PTB")
        )

    def split_text_to_blocks(self, text: str, block_size: int = 800) -> list[str]:
        """Split text to blocks with length limited be "block_size".

        Args:
            text: Text to be splitted.
            block_size: desired maximal block length.
        """
        # tokenize text into senteces
        sentences = self.tokenizer.tokenize_text([text])
        counter = 0
        result = list()
        block = ""
        # process sentences
        for sentence_tokens in sentences:
            sentence_text_list = list()
            # join tokens of one sentes into single string
            for token in sentence_tokens:
                if token.text != ".":
                    sentence_text_list.append(token.text)
            sentence_text = " ".join(sentence_text_list)

            if len(sentence_text) + len(block) < block_size:
                block = block + ". " + sentence_text
                counter += len(sentence_text)
            else:
                result.append(block)
                block = sentence_text
                counter = 0
        if len(block) > 0:
            result.append(block)
        return result
