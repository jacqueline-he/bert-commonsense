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

import wandb
wandb.init(project="coref")


gap_train_path = '/n/fs/nlp-jh70/bert-commonsense/data/gap-development.tsv'
gap_test_path = '/n/fs/nlp-jh70/bert-commonsense/data/gap-test.tsv'

logger = logging.getLogger(__name__)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt = '%m/%d/%Y %H:%M:%S',
					level = logging.INFO)


class MLP(nn.Module):
    def __init__(self, D_in, D_out, n_layer=2):
        super().__init__()
        if n_layer == 1:
            self.linear = nn.Linear(D_in, D_out)
        elif n_layer == 2:
            self.linear = nn.Sequential(
                nn.Linear(D_in, D_out),
                nn.ReLU(),
                nn.Linear(D_out, D_out)
            )
        self.activation = nn.ReLU()
        
    def forward(self,x):
        x = self.linear(x)
        x = self.activation(x)
        return x

class BertForMaskedLM(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.init_weights()
        self.fairfil = None
        if config.fair_filter:
            print('Applying fair filter')
            self.fairfil = MLP(768, 768, 2)
            self.fairfil.linear.load_state_dict(torch.load(config.fair_filter, map_location=torch.device('cuda')))

            for param in self.fairfil.parameters():
                param.requires_grad = False


    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        #print(sequence_output.shape)

        # if self.fairfil is not None:
        #     fair_emb = self.fairfil(outputs.last_hidden_state[:,0]) # CLS token
        #     fair_emb = fair_emb.unsqueeze(1)
        #     #print(fair_emb.shape)
        #     fair_output = torch.hstack((fair_emb, sequence_output[:, 1:]))
        #     #print(fair_output.shape)
        #     prediction_scores = self.cls(fair_output)
        if self.fairfil:
            fair_output = []
            s = sequence_output.size(0)
            #print(s)
            for i in range(s):
                res = self.fairfil(sequence_output[i, :])
                #print(res.shape)
                fair_output.append(res)

            fair_output = torch.stack((fair_output))

            #print(fair_output.shape)
            prediction_scores = self.cls(fair_output)
        else:
            prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            #loss_fct = CrossEntropyLoss()  # -100 index = padding token
            #masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

            loss_fct_ind = CrossEntropyLoss(reduction='none')

            individual_masked_lm_loss = loss_fct_ind(prediction_scores.permute(0,2,1), labels)
            individual_masked_lm_loss = individual_masked_lm_loss.mean(axis=1)
            masked_lm_loss = individual_masked_lm_loss.sum()
            #individual_masked_lm_loss = torch.mean(sums)


        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return {
            'loss': masked_lm_loss,
            'logits': prediction_scores,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
            'individual_loss': individual_masked_lm_loss
        }

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

def convert_examples_to_features(examples, max_seq_len, tokenizer):
    logger.info('Converting examples to features...')
    features = []
    i = 0
    all_input_ids_1, all_attention_mask_1, all_masked_lm_1 = [],[],[]
    all_input_ids_2, all_attention_mask_2, all_masked_lm_2 = [],[],[]
    # tokens_1 is sentence containing cand_a
    # tokens_2 is sentence containing cand_b
    # right answer is always cand_a i think
    for (ex_index, example) in enumerate(examples):
        if example.candidate_b is None:
            continue
        tokens_sent = tokenizer.tokenize(example.text_a)
        tokens_a = tokenizer.tokenize(example.candidate_a)
        tokens_b = tokenizer.tokenize(example.candidate_b)
        tokens_1, type_1, attention_mask_1, masked_lm_1 = [],[],[],[]
        tokens_2, type_2, attention_mask_2, masked_lm_2 = [],[],[],[]
        tokens_1.append("[CLS]")
        tokens_2.append("[CLS]")
        for token in tokens_sent:
            if token=="_":
                tokens_1.extend(["[MASK]" for _ in range(len(tokens_a))])
                tokens_2.extend(["[MASK]" for _ in range(len(tokens_b))])
            else:
                tokens_1.append(token)
                tokens_2.append(token)
        tokens_1 = tokens_1[:max_seq_len-1]#-1 because of [SEP]
        tokens_2 = tokens_2[:max_seq_len-1]
        if tokens_1[-1]!="[SEP]":
            tokens_1.append("[SEP]")
        if tokens_2[-1]!="[SEP]":
            tokens_2.append("[SEP]")
        # TODO - check: may not be right
        attention_mask_1 = (len(tokens_1)*[1])+((max_seq_len-len(tokens_1))*[0])
        attention_mask_2 = (len(tokens_2)*[1])+((max_seq_len-len(tokens_2))*[0])

        #sentences
        input_ids_1 = tokenizer.convert_tokens_to_ids(tokens_1)
        input_ids_2 = tokenizer.convert_tokens_to_ids(tokens_2)
        #replacements
        input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
        input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)

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

        for token in tokens_2:
            if token=="[MASK]":
                if len(input_ids_b)<=0:
                    continue#broken case
                masked_lm_2.append(input_ids_b[0])
                input_ids_b = input_ids_b[1:]
            else:
                masked_lm_2.append(-100)
        while len(masked_lm_2)<max_seq_len:
            masked_lm_2.append(-100)

        # Zero-pad up to the sequence length.
        while len(input_ids_1) < max_seq_len:
            input_ids_1.append(0)
        while len(input_ids_2) < max_seq_len:
            input_ids_2.append(0)
        assert len(input_ids_1) == max_seq_len
        assert len(input_ids_2) == max_seq_len
        assert len(attention_mask_1) == max_seq_len
        assert len(attention_mask_2) == max_seq_len
        assert len(masked_lm_1) == max_seq_len
        assert len(masked_lm_2) == max_seq_len

        all_input_ids_1.append(input_ids_1)
        all_attention_mask_1.append(attention_mask_1)
        all_masked_lm_1.append(masked_lm_1)
        all_input_ids_2.append(input_ids_2)
        all_attention_mask_2.append(attention_mask_2)
        all_masked_lm_2.append(masked_lm_2)
    
    return all_input_ids_1, all_attention_mask_1, all_masked_lm_1, all_input_ids_2, all_attention_mask_2, all_masked_lm_2

def get_data(examples, tokenizer, args):
    # for index in random.sample(range(len(examples)), 5):
    #     logger.warning(f"Sample {index} of the training set: {examples[index]}.")
    #     ex = examples[index]
    #     logger.info(ex.guid)
    #     logger.info(ex.text_a)
    #     logger.info(ex.candidate_a)
    #     logger.info(ex.candidate_b)

    all_input_ids_1, all_attention_mask_1, all_masked_lm_1, \
    all_input_ids_2, all_attention_mask_2, all_masked_lm_2 = convert_examples_to_features(examples=examples, max_seq_len=args.max_seq_length, tokenizer=tokenizer)

    all_input_ids_1 = torch.LongTensor([i for i in all_input_ids_1])
    all_attention_mask_1 = torch.LongTensor([i for i in all_attention_mask_1])
    all_masked_lm_1 = torch.LongTensor([i for i in all_masked_lm_1])

    all_input_ids_2 = torch.LongTensor([i for i in all_input_ids_2])
    all_attention_mask_2 = torch.LongTensor([i for i in all_attention_mask_2])
    all_masked_lm_2 = torch.LongTensor([i for i in all_masked_lm_2])

    data = TensorDataset(all_input_ids_1, all_attention_mask_1, all_masked_lm_1, all_input_ids_2, all_attention_mask_2, all_masked_lm_2)
    return data

def train(model, dl, eval_dl, optimizer, args):
    model.train()
    num_batches = len(dl)
    
    t = time()
    total_loss = 0
    total_correct = []

    if args.dataparallel:
        device = model.module.device
    else:
        device = model.device

    for epoch in tqdm(range(args.num_epochs)):
        for i, batch in enumerate(tqdm(dl)):

            (input_ids_1, attention_mask_1, masked_lm_1, input_ids_2, attention_mask_2, masked_lm_2) = (tens.to(device) for tens in batch)
            
            output_1 = model(input_ids=input_ids_1, labels=masked_lm_1)
            output_2 = model(input_ids=input_ids_2, labels=masked_lm_2)
            loss_1 = output_1['loss']
            loss_2 = output_2['loss']
            loss = loss_1 + args.alpha_param * torch.max(torch.zeros(loss_1.size(), device=device), (torch.ones(loss_1.size(), device=device) * args.beta_param + loss_1 - loss_2))
            loss_1_ind = output_1['individual_loss']
            loss_2_ind = output_2['individual_loss']
            # 1 (correct) should have smaller loss than 2 (incorrect), so result should be neg.
            correct = (loss_1_ind - loss_2_ind < 0).tolist()
            total_correct += correct
            
            if args.dataparallel:
                total_loss += loss.mean().item()
                loss.mean().backward()
            else:
                total_loss += loss.item()
                loss.backward()
            
            optimizer.step()
            model.zero_grad()
            optimizer.zero_grad()

            if args.global_step % args.print_every == 0:
                train_acc = sum(total_correct) / len(total_correct)
                # train_loss = total_loss / len(total_correct)
                print(f'train_acc: {train_acc}, train_loss: {loss}')
                wandb.log({'loss': loss, 'train acc': train_acc})

            if args.global_step % args.eval_interval == 0 and args.global_step != 0:
                dev_acc = evaluate('dev', model, eval_dl, epoch, args)
                if dev_acc > args.best_score:
                    args.best_score = dev_acc
                    save_ckpt(model, optimizer, args, latest=False)
            
            args.global_step += 1

def evaluate(mode, model, dl, epoch, args):
    model.eval()
    print('running eval')

    total_loss = 0
    total_correct = []
    device = model.device
    for i, batch in enumerate(tqdm(dl)):
        (input_ids_1, attention_mask_1, masked_lm_1, input_ids_2, attention_mask_2, masked_lm_2) = (tens.to(device) for tens in batch)

        with torch.no_grad():    
            output_1 = model(input_ids=input_ids_1, labels=masked_lm_1)
            output_2 = model(input_ids=input_ids_2, labels=masked_lm_2)

        loss_1 = output_1['loss']
        loss_2 = output_2['loss']
        loss = loss_1 + args.alpha_param * torch.max(torch.zeros(loss_1.size(), device=device), (torch.ones(loss_1.size(), device=device) * args.beta_param + loss_1 - loss_2))
        loss_1_ind = output_1['individual_loss']
        loss_2_ind = output_2['individual_loss']
            # 1 (correct) should have smaller loss than 2 (incorrect), so result should be neg.
        correct = (loss_1_ind - loss_2_ind < 0).tolist()
        total_correct += correct
            
        if args.dataparallel:
            total_loss += loss.mean().item()
        else:
            total_loss += loss.item()
            
    acc = sum(total_correct) / len(total_correct)
    loss = total_loss / len(total_correct)    
    log_dict = {
        'epoch': epoch,
        'eval_acc': acc,
        'eval_loss': loss,
        'global_step': args.global_step,
    }
    print(mode, log_dict)
    return acc


        
        

def save_args(args):
    path = os.path.join(args.ckpt_dir, 'args.yaml')
    with open(path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    print(f'Arg file saved at: {path}')

def save_ckpt(model, optimizer, args, latest=False):
    if not latest:
        best_ckpt_path = args.file_path.format(
            step=args.global_step,
            best_score=args.best_score * 100
        )
        checkpoint = {'ckpt_path': best_ckpt_path}
    else:
        checkpoint = {'ckpt_path': os.path.join(args.ckpt_dir, 'latest.ckpt')}

    states = model.state_dict() if not args.dataparallel else model.module.state_dict()
    checkpoint['states'] = states
    checkpoint['optimizer_states'] = optimizer.state_dict()

    if not latest:
        for rm_path in glob.glob(os.path.join(args.ckpt_dir, '*.pt')):
            os.remove(rm_path)

    torch.save(checkpoint, checkpoint['ckpt_path'])
    print(f"Model saved at: {checkpoint['ckpt_path']}")

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
    parser.add_argument('--batch_size', default=24)
    parser.add_argument('--max_seq_length', default=128)
    parser.add_argument('--fair_filter', default='/n/fs/scratch/jh70/fairfil_tok_layer_2/fairfil_tok_ckpt_800_0.pth')
    parser.add_argument('--ckpt_dir', default='/n/fs/scratch/jh70/actual-fairfil-tok-coref-0121')
    parser.add_argument('--num_epochs', 
                        type=int,
                        default=5)
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

    args = parser.parse_args()
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    args.best_score = float('-inf')
    args.global_step = 0
    args.arg_path = os.path.join(args.ckpt_dir, 'args.yaml')
    args.file_path = os.path.join(args.ckpt_dir, args.file_name)

    '''Print GPU information'''
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')     
    logger.info("device: {} n_gpu: {} distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

    processor = DataProcessor()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, cache_dir='/n/fs/scratch/jh70')

    
    train_examples = processor.get_examples(args.data_dir, "gap-train")
    train_data = get_data(train_examples, tokenizer, args)

    source = os.path.join(args.data_dir,"gap-test.tsv")
    test_examples = processor.gap_train(source)
    test_data = get_data(test_examples, tokenizer, args)


    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True)

    config = AutoConfig.from_pretrained(args.model_name, cache_dir='/n/fs/scratch/jh70')
    config.model_name = args.model_name
    #config.fair_filter=None
    config.fair_filter = args.fair_filter if args.fair_filter else None
    model = BertForMaskedLM.from_pretrained(args.model_name, config=config, cache_dir='/n/fs/scratch/jh70')

    if torch.cuda.device_count() and args.dataparallel:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    model = model.to(args.device)

    if args.dataparallel:
        print(f'Trainable params: {sum(p.numel() for p in model.module.parameters() if p.requires_grad)}')
        print(f'All params      : {sum(p.numel() for p in model.module.parameters())}')
    else:
        print(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        print(f'All params      : {sum(p.numel() for p in model.parameters())}')

    if args.dataparallel:
        optimizer = AdamW(model.module.parameters(), lr=0.00005) # only cls parameters
    else:
        optimizer = AdamW(model.parameters(), lr=0.00005) # only cls parameters

    save_args(args)
    train(model, train_loader, eval_loader, optimizer, args)

    logger.info('Finished training')
    save_ckpt(model, optimizer, args, True)




if __name__ == "__main__":
    
    main()


