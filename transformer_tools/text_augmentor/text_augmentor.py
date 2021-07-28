# Copyright (c) 2021 Sitong Ye, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import pandas as pd
from tqdm import tqdm
import re
from preprocess import preprocess
import math
import functools
import time
import spacy
import torch
from fairseq.models.transformer import TransformerModel
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM
import random


class TextAug:

    def __init__(self, nr_aug_per_sent):
        self.nr_aug_per_sent = nr_aug_per_sent

    def _generate(self, sent):
        raise NotImplementedError

    def generate_augmentation(self, input_text):
        ## takes in list of text and return list
        output_list = []
        if isinstance(input_text, str):
            input_text = [input_text]
        for sentence in input_text:
            for k in range(self.nr_aug_per_sent):
                output_list.append(self._generate(sentence))
        return output_list

    def generate_augmentated_dataframe(self, input_df, text_column, label_column):
        ## input dataframe
        # output dataframe does not contain original data
        augmented_df = pd.DataFrame(columns=["augmented_text", "label"])
        augmented_texts = []
        labels = []
        for row in tqdm(input_df.itertuples()):
            ori_text = getattr(row, text_column)
            ori_label = getattr(row, label_column)
            for k in range(self.nr_aug_per_sent):
                try:
                    augmented_texts.append(self._generate(ori_text))
                    labels.append(ori_label)
                except:
                    continue
        # populated them into a dataframe
        augmented_df["augmented_text"] = augmented_texts
        augmented_df["label"] = labels
        return augmented_df


class TextaugWord(TextAug):

    def __init__(self, nr_aug_per_sent, pos_model_path, swap_proportion=0.2):
        super().__init__(nr_aug_per_sent)
        self.pos_filtering = True
        # decides with which probability a word is swapped or it stays the same
        self.pos_model_path = pos_model_path
        self.de_model = spacy.load(self.pos_model_path)
        # portion of valid token to be swapped
        self.swap_propotion = swap_proportion

    @functools.lru_cache(maxsize=10_000)
    def _is_sameword(self, original_word, new_word):
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
        sent = preprocess(sent)
        doc = self.de_model(sent)
        return [token for token in doc]

    @functools.lru_cache(maxsize=1000)
    def _is_validword(self, spacy_token, valid_pos=["VERB", "ADV", "NOUN", "ADJ"]):
        """
        :param spacy_token: spacy token of the word
        :param valid_pos:
        :return: Boolean: True / False
        """
        return (spacy_token.pos_ in valid_pos)

    def _swap_with_weights(self, ori_word, prob):
        # swap a word with candidates or stay the same depending on given probability
        # prob: probability of being swapped
        swap = self.candidate_dict.get(ori_word)
        return random.choices(population=[ori_word, random.choice(swap)], weights=[1 - prob, prob])

    def get_candidates(self, word):
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
        valid_token_idx = [idx for (idx, tok) in enumerate(token_list) if self._is_validword(tok) == True]
        # print("valid tokens: ", [(idx, token_list[idx].text) for idx in valid_token_idx])
        # select the token to be swapped
        if len(valid_token_idx) != 0:
            selected_index = random.sample(valid_token_idx, math.ceil(self.swap_propotion * len(valid_token_idx)))
            # print("selected_index", selected_index)
            # then we swap the selected...
            for idx in selected_index:
                # insert token as key
                # only insert key if the token is not dictionary
                cand_list = [w.lower() for w in self.get_candidates(token_list[idx].text.lower()) if
                             self._is_sameword(token_list[idx].text.lower(), w.lower()) is False]
                if (len(cand_list) > 0) and (cand_list is not None):
                    # print("cand_list", cand_list)
                    aug_text[idx] = random.choice(cand_list)

        out = " ".join(aug_text).strip()
        out = re.sub(" +", " ", out)
        out = re.sub(" ,", ",", out)
        out = re.sub(r" \.", ".", out)
        return out

    def generate_augmentation(self, input_text):
        return super().generate_augmentation(input_text)

    def generate_augmentated_dataframe(self, input_df, text_column, label_column):
        return super().generate_augmentated_dataframe(input_df, text_column, label_column)


class TextAugEmbedding(TextaugWord):
    def __init__(self, nr_aug_per_sent, pos_model_path, embedding_path, score_threshold=0.5, base_embedding="fasttext",
                 swap_proportion=0.2):
        super().__init__(nr_aug_per_sent, pos_model_path, swap_proportion)
        self.base_embedding = base_embedding
        self.aug_model = None
        self.score_threshold = score_threshold
        if base_embedding == "fasttext":
            try:
                import fasttext
                import fasttext.util
                if isinstance(embedding_path, list):
                    # in order not to mess up the random seed set up in the main script
                    state = random.getstate()
                    t = 1000 * time.time()  # current time in milliseconds
                    random.seed(int(t) % 2 ** 32)
                    embedding_path = random.choice(embedding_path)
                    print("selected embedding: ", embedding_path)
                    random.setstate(state)
                self.aug_model = fasttext.load_model(embedding_path)
                # print("fasttext embedding loaded")
                # print("embedding dimension: ", ft.get_dimension())
            except Exception as err:
                raise Exception("Import / load fasttext model unsuccessful!", err)
        else:
            raise ValueError("not supported embedding (yet)")

    @functools.lru_cache(maxsize=1000)
    def get_candidates(self, word):
        candidates = [k[1] for k in self.aug_model.get_nearest_neighbors(word) if k[0] >= self.score_threshold]
        return candidates

    def _generate(self, sent):
        return super()._generate(sent)

    def generate_augmentation(self, input_text):
        return super().generate_augmentation(input_text)

    def generate_augmentated_dataframe(self, input_df, text_column, label_column):
        return super().generate_augmentated_dataframe(input_df, text_column, label_column)


class TextaugBackTrans(TextAug):

    def __init__(self, nr_aug_per_sent, ori_mid_model_path, ori_mid_checkpoints,
                 mid_ori_model_path, mid_ori_checkpoints, print_mid_text=False):
        #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        super().__init__(nr_aug_per_sent)
        self.ori_mid_model_path = ori_mid_model_path
        self.mid_ori_model_path = mid_ori_model_path
        self.ori_mid_checkpoints = ori_mid_checkpoints
        self.mid_ori_checkpoints = mid_ori_checkpoints
        self.ori2mid_model = self._load_transmodel(self.ori_mid_model_path, self.ori_mid_checkpoints)
        self.mid2ori_model = self._load_transmodel(self.mid_ori_model_path, self.mid_ori_checkpoints)
        self.print_mid_text = print_mid_text
        print("back translation object initiated")

    def _load_transmodel(self, source2target_modelpath,
                         checkpoint_files):
        return TransformerModel.from_pretrained(
            model_name_or_path=source2target_modelpath,
            checkpoint_file=checkpoint_files,
            data_name_or_path=source2target_modelpath,
            bpe="fastbpe",
            bpe_codes=os.path.join(source2target_modelpath, "bpecodes"))

    def _translate(self, transmodel, text):
        return transmodel.translate(text)

    def _generate(self, sent):
        mid_text = self._translate(self.ori2mid_model, sent)
        if self.print_mid_text is True:
            print("mid_text: ", mid_text)
        out = self._translate(self.mid2ori_model, mid_text)
        return out

    def generate_augmentation(self, input_text):
        return super().generate_augmentation(input_text)

    def generate_augmentated_dataframe(self, input_df, text_column,
                                       label_column):
        return super().generate_augmentated_dataframe(input_df,
                                                      text_column,
                                                      label_column)


class TextaugContextEmbed(TextAug):

    def __init__(self, nr_aug_per_sent, local_model_path, model="bert-base-german-cased", from_local=True):
        """

        :param nr_aug_per_sent: int
        :param model: registry name from the transformers library
        :param model_path: e.g. "./model/bert-case-german-cased/" directory should include both model and tokenizer
        :param from_local: Boolean
        """
        super().__init__(nr_aug_per_sent)
        if from_local is False:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.model = AutoModelForMaskedLM.from_pretrained(model)
        else:
            # load from local
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.model = AutoModelForMaskedLM.from_pretrained(local_model_path)

    def _generate(self, sequence, nr_candidates=5):
        try:
            input = self.tokenizer.encode(sequence, return_tensors="pt")
            random_chosen_index = random.choice(range(1, len(input[0]) - 2))
            # print(random_chosen_index)
            input[0][random_chosen_index] = self.tokenizer.mask_token_id
            mask_token_index = torch.where(input == self.tokenizer.mask_token_id)[1]
            token_logits = self.model(input).logits
            mask_token_logits = token_logits[0, mask_token_index, :]
            top_tokens = torch.topk(mask_token_logits, nr_candidates, dim=1).indices[0].tolist()
            input[0][random_chosen_index] = random.choice(top_tokens)
            output = self.tokenizer.decode(input[0], skip_special_tokens=True)
        except:
            return sequence
        return output

    def generate_augmentation(self, input_text):
        return super().generate_augmentation(input_text)

    def generate_augmentated_dataframe(self, input_df, text_column,
                                       label_column):
        return super().generate_augmentated_dataframe(input_df,
                                                      text_column,
                                                      label_column)
