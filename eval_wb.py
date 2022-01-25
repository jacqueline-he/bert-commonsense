import logging
import os
import random
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
from time import time
import torch.distributed
import numpy as np

from pathlib import Path

import argparse
from argparse import Namespace

import pickle

import yaml
import glob
import logging

import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig

from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler

from torch.nn import CrossEntropyLoss

from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from data_reader import InputExample, DataProcessor
from scorer import scorer

from transformers import BertPreTrainedModel, BertModel  
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.modeling_outputs import MaskedLMOutput

from train import BertForMaskedLM, get_data
from scorer import scorer

logger = logging.getLogger(__name__)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt = '%m/%d/%Y %H:%M:%S',
					level = logging.INFO)

def convert_examples_to_features(examples, max_seq_len, tokenizer):
    logger.info('Converting examples to features...')
    features = []
    i = 0
    all_input_ids_1, all_attention_mask_1, all_masked_lm_1 = [],[],[]
    # tokens_1 is sentence containing cand_a
    # tokens_2 is sentence containing cand_b
    # right answer is always cand_a i think
    for (ex_index, example) in enumerate(examples):
        tokens_sent = tokenizer.tokenize(example.text_a)
        tokens_a = tokenizer.tokenize(example.candidate_a)

        tokens_1, type_1, attention_mask_1, masked_lm_1 = [],[],[],[]
        tokens_1.append("[CLS]")
        for token in tokens_sent:
            if token=="_":
                tokens_1.extend(["[MASK]" for _ in range(len(tokens_a))])
            else:
                tokens_1.append(token)

        tokens_1 = tokens_1[:max_seq_len-1]#-1 because of [SEP]
        if tokens_1[-1]!="[SEP]":
            tokens_1.append("[SEP]")
        # TODO - check: may not be right
        attention_mask_1 = (len(tokens_1)*[1])+((max_seq_len-len(tokens_1))*[0])

        #sentences
        input_ids_1 = tokenizer.convert_tokens_to_ids(tokens_1)
        #replacements
        input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)

        for token in tokens_1:
            if token=="[MASK]":
                if len(input_ids_a)<=0:
                    continue#broken case
                masked_lm_1.append(input_ids_a[0])
                input_ids_a = input_ids_a[1:]
            else:
                masked_lm_1.append(-100)
        while len(masked_lm_1)<max_seq_len:
            masked_lm_1.append(-100)


        # Zero-pad up to the sequence length.
        while len(input_ids_1) < max_seq_len:
            input_ids_1.append(0)
        # while len(input_ids_2) < max_seq_len:
        #     input_ids_2.append(0)
        # assert len(input_ids_1) == max_seq_len
        # assert len(input_ids_2) == max_seq_len
        # assert len(attention_mask_1) == max_seq_len
        # assert len(attention_mask_2) == max_seq_len
        # assert len(masked_lm_1) == max_seq_len
        # assert len(masked_lm_2) == max_seq_len

        all_input_ids_1.append(input_ids_1)
        all_attention_mask_1.append(attention_mask_1)
        all_masked_lm_1.append(masked_lm_1)
    
    return all_input_ids_1, all_attention_mask_1, all_masked_lm_1


def get_test_data(examples, tokenizer, args):
    for index in random.sample(range(len(examples)), 5):
        logger.warning(f"Sample {index} of the training set: {examples[index]}.")
        ex = examples[index]
        logger.info(ex.guid)
        logger.info(ex.text_a)
        logger.info(ex.candidate_a)
        logger.info(ex.candidate_b)

    all_input_ids_1, all_attention_mask_1, all_masked_lm_1 = convert_examples_to_features(examples=examples, max_seq_len=args.max_seq_length, tokenizer=tokenizer)

    all_input_ids_1 = torch.LongTensor([i for i in all_input_ids_1])
    all_attention_mask_1 = torch.LongTensor([i for i in all_attention_mask_1])
    all_masked_lm_1 = torch.LongTensor([i for i in all_masked_lm_1])
    data = TensorDataset(all_input_ids_1, all_attention_mask_1, all_masked_lm_1)
    return data

def test(test_set, model, processor, tokenizer, args):
    test_examples = processor.get_examples(args.data_dir, test_set)
    
    test_data = get_data(test_examples, tokenizer, args)

    eval_dataloader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True)
    logger.info("***** Running evaluation for %s *****", test_set)

    ans_stats=[]

    device = model.device
    sum_corr = 0
    len_corr = 0
    for batch in tqdm(eval_dataloader,desc="Evaluation"):
        (input_ids_1, attention_mask_1, masked_lm_1, input_ids_2, attention_mask_2, masked_lm_2) = (tens.to(device) for tens in batch)

        with torch.no_grad():    
            output_1 = model(input_ids=input_ids_1, labels=masked_lm_1)
            output_2 = model(input_ids=input_ids_2, labels=masked_lm_2)

        loss_1_ind = output_1['individual_loss']
        loss_2_ind = output_2['individual_loss']


        # 1 (correct) should have smaller loss than 2 (incorrect), so result should be neg.
        correct = (loss_1_ind - loss_2_ind < 0).tolist()
        #eval_loss = loss_1.to('cpu').numpy()

        sum_corr += sum(correct)
        len_corr += len(correct)


        for loss in correct:
            curr_id = len(ans_stats)
            # print(test_examples[curr_id].guid)
            # print(test_examples[curr_id].ex_true)
            # print(loss)
            # # assert(0 == 1)
            ans_stats.append((test_examples[curr_id].guid,test_examples[curr_id].ex_true,loss))
    
    #print(sum_corr / len_corr)
    #return scorer(ans_stats, test_set)
    return (sum_corr / len_corr)




def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default="data/",
                        type=str,
                        help="The input data dir. Should contain the files for the task.")
    parser.add_argument('--dataparallel', default=False)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--print_every', default=10)
    parser.add_argument('--eval_interval', default=100)
    parser.add_argument('--model_name', default='bert-base-uncased')
    parser.add_argument('--file_name', default='model-step={step}-acc={best_score:.2f}.pt')
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--max_seq_length', default=128)
    parser.add_argument('--fair_filter', default='/n/fs/scratch/jh70/fairfil_tok_layer_2/fairfil_tok_ckpt_800_0.pth')
    parser.add_argument('--ckpt_dir', default='/n/fs/scratch/jh70/fairfil-tok-coref-0121')
    parser.add_argument('--num_epochs', default=5)
    parser.add_argument("--alpha_param",
                        default=10,
                        type=float,
                        help="Discriminative penalty hyper-parameter.")
    parser.add_argument("--beta_param",
                        default=0.4,
                        type=float,
                        help="Discriminative intolerance interval hyper-parameter.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help = "local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--load_from_file",
                        default='/n/fs/scratch/jh70/bert-coref-0121-new-20/model-step=1300-acc=67.12.pt')

    args = parser.parse_args()

    processor = DataProcessor()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, cache_dir='/n/fs/scratch/jh70')

    

    config = AutoConfig.from_pretrained(args.model_name, cache_dir='/n/fs/scratch/jh70')
    config.model_name = args.model_name
    config.fair_filter = args.fair_filter if args.fair_filter else None
    #config.fair_filter=None
    model = BertForMaskedLM.from_pretrained(args.model_name, config=config, cache_dir='/n/fs/scratch/jh70')
    model.load_state_dict(torch.load(args.load_from_file)['states'])

    model = model.to(args.device)
    model.eval()
    

    

    
    # m_corr = 0
    # m_tot = 0
    # f_corr = 0
    # f_tot = 0
    # for a in ans_stats:
    #     if a[1] == 'M':
    #         if a[2] == True:
    #             m_corr += 1
    #         m_tot += 1
    #     else:
    #         if a[2] == True:
    #             f_corr += 1
    #         f_tot += 1

    # print(f'm: {m_corr / m_tot}')
    # print(f'f: {f_corr / f_tot}')
    
    #scorer(ans_stats,'gap-test',output_file=os.path.join(args.ckpt_dir, "gap-answers.tsv"))
    print("Winogender: ", test("winogender", model, processor, tokenizer, args))
    print("WinoBias Anti Stereotyped Type 1: ", test("winobias-anti1", model, processor, tokenizer, args))
    print("WinoBias Pro Stereotyped Type 1: ",test("winobias-pro1", model, processor,tokenizer, args))
    print("WinoBias Anti Stereotyped Type 2: ",test("winobias-anti2", model, processor,tokenizer, args))
    print("WinoBias Pro Stereotyped Type 2: ",test("winobias-pro2", model, processor,tokenizer, args))


if __name__ == "__main__":
    
    main()