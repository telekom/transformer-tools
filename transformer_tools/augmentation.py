# Copyright (c) 2021 Sitong Ye, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Text augmentation module."""

import functools
import os
import random
import re
import time

import pandas as pd
import spacy
import torch
import yaml  # type: ignore
from transformers import AutoModelForMaskedLM, AutoTokenizer


class RamdomAugGenerator:
    """TODO: add docstring."""

    def __init__(self, aug_config, object_map, shuffle_weight=None, swap_prob=0.8):
        """TODO: fix docstring.

        input:
            aug_config: takes in both ".yaml" (specific for augmentation) or directly dictionary
                type
            object_map: dictionary, which maps the namespace in config file to the object
        """
        shuffle_weight = [0.5, 0.5] if shuffle_weight is None else shuffle_weight
        if isinstance(aug_config, str):
            if aug_config.endswith(".yaml") or aug_config.endswith(".yml"):
                with open(aug_config, "r") as config:
                    self.cfg = yaml.safe_load(config)
        elif isinstance(aug_config, dict):
            self.cfg = aug_config
        self.object_map = object_map
        self.shuffle_weight = shuffle_weight
        assert len(self.object_map) == len(shuffle_weight)
        self.swap_prob = swap_prob

        # initialize everything
        self.object_factory = {}
        for method in self.cfg:
            self.object_factory[method] = self.object_map[method](**self.cfg[method])

    def __call__(self, text):
        """TODO: add docstring."""
        # when the superclass is called after initialization,
        # it randomly choice from available subclasses,
        # currently it's hard coded here, which class is available. can be moved to yaml file
        if random.random() > self.swap_prob:
            # it does not swap, in this case, just return the input text
            # print("not augmented")
            return text
        # otherwise it will be swapped
        # shuffle the methods
        selected_method = random.choices(
            list(self.object_factory.keys()), weights=self.shuffle_weight
        )
        print("selected augmentation method:", selected_method[0])
        # initialise the selected_method correspondently
        out = self.object_factory[selected_method[0]].generate_augmentation(text)[0]
        return out


class TextAug:
    """TODO: add docstring."""

    def __init__(self, nr_aug_per_sent):
        self.nr_aug_per_sent = nr_aug_per_sent

    def _generate(self, sent):
        raise NotImplementedError

    def generate_augmentation(self, input_text):
        """TODO: add docstring."""
        # takes in list of text and return list
        output_list = []
        if isinstance(input_text, str):
            input_text = [input_text]
        for sentence in input_text:
            for _ in range(self.nr_aug_per_sent):
                output_list.append(self._generate(sentence))
        return output_list

    def generate_augmentated_dataframe(self, input_df, text_column, label_column):
        """TODO: add docstring."""
        # input dataframe
        # output dataframe does not contain original data
        augmented_df = pd.DataFrame(columns=["augmented_text", "label"])
        augmented_texts = []
        labels = []
        for row in input_df.itertuples():
            ori_text = getattr(row, text_column)
            ori_label = getattr(row, label_column)
            for _ in range(self.nr_aug_per_sent):
                try:
                    augmented_texts.append(self._generate(ori_text))
                    labels.append(ori_label)
                except ValueError:  # noqa: E722
                    continue
        # populated them into a dataframe
        augmented_df["augmented_text"] = augmented_texts
        augmented_df["label"] = labels
        return augmented_df


class TextaugWord(TextAug):
    """TODO: add docstring."""

    def __init__(self, nr_aug_per_sent, pos_model_path, swap_dice=0.2):
        super().__init__(nr_aug_per_sent)
        self.pos_filtering = True
        self.pos_model_path = pos_model_path
        self.de_model = spacy.load(self.pos_model_path)
        # decides with which probability a word is swapped or it stays the same
        self.swap_dice = swap_dice

    @functools.lru_cache(maxsize=10_000)
    @staticmethod
    def _is_sameword(original_word, new_word):
        # clean up and lowercase at the same time
        # only be used at word level
        def only_word(text_chunk):
            if len(re.findall(r"[a-z]+", text_chunk.lower())) != 0:
                return re.findall(r"[a-z]+", text_chunk.lower())[0]
            return ""

        ori = only_word(original_word.lower())
        new = only_word(new_word.lower())
        return ori == new

    @functools.lru_cache(maxsize=10_000)
    def _gen_spacy_token(self, sent):
        doc = self.de_model(sent)
        return list(doc)

    @functools.lru_cache(maxsize=1000)
    @staticmethod
    def _is_validword(spacy_token, valid_pos=None):
        """TODO: fix docstring.

        :param spacy_token: spacy token of the word
        :param valid_pos:
        :return: Boolean: True / False
        """
        valid_pos = ["VERB", "ADV", "NOUN", "ADJ"] if valid_pos is None else valid_pos
        return spacy_token.pos_ in valid_pos

    def _swap_with_weights(self, ori_word, prob):
        # swap a word with candidates or stay the same depending on given probability
        # prob: probability of being swapped
        swap = self.get_candidates(ori_word)
        return random.choices(population=[ori_word, random.choice(swap)], weights=[1 - prob, prob])

    def get_candidates(self, word):
        """TODO: add docstring."""
        raise NotImplementedError

    def _generate(self, sent):
        # candidate is based on part of speech filtering with spacy pos tag
        # candidate_generation is applied on every sent (one instance)
        # generate one augmentation
        # aug_text is initialized as list of tokens from the input sent
        aug_text = [token.text for token in self._gen_spacy_token(sent)]
        # print("tokens: ", aug_text)
        # for token in self._gen_spacy_token(sent):
        token_list = self._gen_spacy_token(sent)
        valid_token_idx = [idx for (idx, tok) in enumerate(token_list) if self._is_validword(tok)]
        # print("valid tokens: ", [(idx, token_list[idx].text) for idx in valid_token_idx])
        # select the token to be swapped
        if len(valid_token_idx) != 0:
            # print("selected_index", selected_index)
            # then we swap the selected...
            aug_text = [
                self._swap_with_weights(aug_text[idx], self.swap_dice)
                if idx in valid_token_idx
                else word.lower()
                for (idx, word) in enumerate(aug_text)
            ]
        out = " ".join(aug_text).strip()
        out = re.sub(" +", " ", out)
        out = re.sub(" ,", ",", out)
        out = re.sub(r" \.", ".", out)
        return out


class TextAugEmbedding(TextaugWord):
    """TODO: add docstring."""

    def __init__(
        self,
        nr_aug_per_sent,
        pos_model_path,
        embedding_path,
        score_threshold=0.5,
        base_embedding="fasttext",
        swap_dice=0.2,
        from_local=True,
        language="de",
    ):
        super().__init__(nr_aug_per_sent, pos_model_path, swap_dice)  # TODO: test online import
        self.base_embedding = base_embedding
        self.aug_model = None
        self.score_threshold = score_threshold
        self.embedding_path = embedding_path
        self.language = language
        if base_embedding == "fasttext":
            try:
                import fasttext  # pylint: disable=import-outside-toplevel
                import fasttext.util  # pylint: disable=import-outside-toplevel

                if from_local is False:
                    # load german model from fasttext
                    fasttext.util.download_model(self.language, if_exists="ignore")
                    # if from_local is false, embedding_path configured in the configuration file
                    # should be the name of model, for example: "cc.de.300.bin"
                else:
                    if isinstance(self.embedding_path, list):
                        # in order not to mess up the random seed set up in the main script
                        state = random.getstate()
                        timestamp = 1000 * time.time()  # current time in milliseconds
                        random.seed(int(timestamp) % 2 ** 32)
                        self.embedding_path = random.choice(self.embedding_path)
                        random.setstate(state)
                print("selected embedding: ", self.embedding_path)
                self.aug_model = fasttext.load_model(self.embedding_path)
                # print("fasttext embedding loaded")
                # print("embedding dimension: ", ft.get_dimension())
            except Exception as err:
                raise Exception("Import / load fasttext model unsuccessful!") from err
        else:
            raise ValueError("not supported embedding (yet)")

    @functools.lru_cache(maxsize=1000)
    def get_candidates(self, word, nr=5):
        """TODO: add docstring."""
        candidates = [
            i[1]
            for i in self.aug_model.get_nearest_neighbors(word, k=nr)
            if i[0] >= self.score_threshold
        ]
        return candidates


class TextaugBackTrans(TextAug):
    """TODO: add docstring."""

    def __init__(
        self,
        nr_aug_per_sent,
        ori_mid_model_path,
        ori_mid_checkpoints,
        mid_ori_model_path,
        mid_ori_checkpoints,
        print_mid_text=False,
        from_local=True,
    ):
        # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        super().__init__(nr_aug_per_sent)
        self.from_local = from_local
        self.ori_mid_model_path = ori_mid_model_path
        self.mid_ori_model_path = mid_ori_model_path
        self.ori_mid_checkpoints = ori_mid_checkpoints
        self.mid_ori_checkpoints = mid_ori_checkpoints
        self.ori2mid_model = self._load_transmodel(
            self.ori_mid_model_path, self.ori_mid_checkpoints, self.from_local
        )
        self.mid2ori_model = self._load_transmodel(
            self.mid_ori_model_path, self.mid_ori_checkpoints, self.from_local
        )
        self.print_mid_text = print_mid_text
        print("back translation object initiated")

    @staticmethod
    def _load_transmodel(source2target_modelpath, checkpoint_files, from_local):
        if from_local is True:
            from fairseq.models.transformer import (  # pylint: disable=import-outside-toplevel
                TransformerModel,
            )

            model = TransformerModel.from_pretrained(
                model_name_or_path=source2target_modelpath,
                checkpoint_file=checkpoint_files,
                data_name_or_path=source2target_modelpath,
                bpe="fastbpe",
                bpe_codes=os.path.join(source2target_modelpath, "bpecodes"),
            )
        else:
            if source2target_modelpath in torch.hub.list("pytorch/fairseq"):
                try:
                    model = torch.hub.load(
                        "pytorch/fairseq",
                        source2target_modelpath,
                        checkpoint_file=checkpoint_files,
                        tokenizer="moses",
                        bpe="subword_nmt",
                    )
                except Exception as err:
                    raise Exception("Import / load fairseq model unsuccessful!") from err
            else:
                raise ValueError("Model can not be found")
        return model

    @staticmethod
    def _translate(transmodel, text):
        return transmodel.translate(text)

    def _generate(self, sent):
        mid_text = self._translate(self.ori2mid_model, sent)
        if self.print_mid_text is True:
            print("mid_text: ", mid_text)
        out = self._translate(self.mid2ori_model, mid_text)
        return out


class TextaugContextEmbed(TextAug):
    """TODO: add docstring."""

    def __init__(
        self,
        nr_aug_per_sent,
        local_model_path,
        model="bert-base-german-cased",
        from_local=True,
        nr_candidates=5,
    ):
        """TODO: fix docstring.

        :param nr_aug_per_sent: int
        :param model: registry name from the transformers library
        :param model_path: e.g. "./model/bert-case-german-cased/" directory should include both
            model and tokenizer
        :param from_local: Boolean
        :param nr_candidates: int number of candidates for each masked word
        """
        super().__init__(nr_aug_per_sent)
        if from_local is False:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            # pylint: disable=no-value-for-parameter
            self.model = AutoModelForMaskedLM.from_pretrained(model)
        else:
            # load from local
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            # pylint: disable=no-value-for-parameter
            self.model = AutoModelForMaskedLM.from_pretrained(local_model_path)
        self.nr_candidates = nr_candidates

    def _generate(self, sent):
        try:
            _input = self.tokenizer.encode(sent, return_tensors="pt")
            random_chosen_index = random.choice(range(1, len(_input[0]) - 2))
            # print(random_chosen_index)
            _input[0][random_chosen_index] = self.tokenizer.mask_token_id
            mask_token_index = torch.where(_input == self.tokenizer.mask_token_id)[1]
            token_logits = self.model(_input).logits
            mask_token_logits = token_logits[0, mask_token_index, :]
            top_tokens = (
                torch.topk(mask_token_logits, self.nr_candidates, dim=1).indices[0].tolist()
            )
            _input[0][random_chosen_index] = random.choice(top_tokens)
            output = self.tokenizer.decode(_input[0], skip_special_tokens=True)
        except ValueError:  # noqa: E722
            return sent
        return output
