# ------------------------------------------------------------------------
# Modified from OFA (https://github.com/OFA-Sys/OFA)
# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.
# ------------------------------------------------------------------------
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
import torch.utils.data
from fairseq.data import FairseqDataset

logger = logging.getLogger(__name__)


class BaseDataset(FairseqDataset):
    def __init__(self, split, dataset, bpe, src_dict, tgt_dict, task="rec"):
        self.split = split
        self.dataset = dataset
        self.bpe = bpe
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        self.bos = src_dict.bos()
        self.eos = src_dict.eos()
        self.pad = src_dict.pad()
        self.bos_item = torch.LongTensor([self.bos])
        self.eos_item = torch.LongTensor([self.eos])
        self.task = task

    def __len__(self):
        return len(self.dataset)

    def encode_text(self, text, length=None, append_bos=False, append_eos=False, use_bpe=True):
        s = self.tgt_dict.encode_line(line=self.bpe.encode(text) if use_bpe else text, add_if_not_exist=False, append_eos=False).long()
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([self.bos_item, s])
        if append_eos:
            s = torch.cat([s, self.eos_item])
        return s

    def pre_question(self, question, max_ques_words):
        question = question.lower().lstrip(",.!?*#:;~").replace("-", " ").replace("/", " ")

        question = re.sub(
            r"\s{2,}",
            " ",
            question,
        )
        question = question.rstrip("\n")
        question = question.strip(" ")

        # truncate question
        question_words = question.split(" ")
        if len(question_words) > max_ques_words:
            question = " ".join(question_words[:max_ques_words])

        return question

    def pre_caption(self, caption, max_words):
        caption = caption.lower().lstrip(",.!?*#:;~").replace("-", " ").replace("/", " ").replace("<person>", "person")

        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > max_words:
            caption = " ".join(caption_words[:max_words])

        return caption
