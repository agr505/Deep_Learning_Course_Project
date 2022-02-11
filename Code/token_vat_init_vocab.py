""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids,subtoken_ids,sent_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.subtoken_ids = subtoken_ids
        self.sent_id = sent_id

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels

import argparse
import glob
import json
import logging as log
import os
import random
from dataclasses import dataclass, field
from typing import Optional
from torch.nn import CrossEntropyLoss
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    get_linear_schedule_with_warmup,BertForTokenClassification, BertTokenizer
)
from transformers.modeling_bert import *
#from transformers.models.bert.modeling_bert import *
#from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
logger = log.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def read_data(tokenizer, labels, pad_token_label_id, mode, train_examples = -1, 
              omit_sep_cls_token=False,
              pad_subtoken_with_real_label=False,
              semi = False):
 
    examples = read_examples_from_file("data/conll2003",mode)
     

    
   
    print(mode)
    print('data num: {}'.format(len(examples)))

    features = convert_examples_to_features(examples, labels, 256, tokenizer, 
                                cls_token = tokenizer.cls_token, sep_token =  tokenizer.sep_token, pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                cls_token_segment_id = 2 if "BertForTokenClassification" in ["xlnet"] else 0, 
                                sequence_a_segment_id = 0, pad_token_segment_id=4 if "BertForTokenClassification" in ["xlnet"] else 0, 
                                pad_token_label_id = pad_token_label_id,
                                omit_sep_cls_token=omit_sep_cls_token,
                                pad_subtoken_with_real_label=pad_subtoken_with_real_label,
                                subtoken_label_type="real",
                                label_sep_cls=True)
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_subtoken_ids = torch.tensor([f.subtoken_ids for f in features], dtype=torch.long)
    all_sent_id = torch.tensor([f.sent_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_subtoken_ids,all_sent_id)
    
    return dataset

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, 
                cls_token='[CLS]', sep_token='[SEP]', pad_token=0, 
                cls_token_segment_id=0, sequence_a_segment_id=0, pad_token_segment_id=0,
                pad_token_label_id=-100, mask_padding_with_zero=True,
                                 omit_sep_cls_token=False,
                                 pad_subtoken_with_real_label=True,
                                subtoken_label_type='real',label_sep_cls=False):
    print('process training data',len(examples))
    
    label_sep_cls=False
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    max_len=0
    

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))


        tokens = []
        label_ids = []
        subtoken_ids=[]
        # this subtoken_ids array is used to mark whether the token is a subtoken of a word or not
        for word, label in zip(example.words, example.labels):

            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            """
            'real': the defaul one, same as above:
                    O word: Oxx->OOO
                    I word: Ixx->III
                    B word: Bxx->BII
            'repeat': repeat the label:
                    O word: Oxx->OOO
                    I word: Ixx->III
                    B word: Bxx->BBB
            'O': change to O
                    O word: Oxx->OOO
                    I word: Ixx->IOO
                    B word: Bxx->BOO            
            """
            
            if len(word_tokens) > 0:
                if pad_subtoken_with_real_label:
                    if subtoken_label_type=='real':
                        if label[0]=='B':
                            pad_label='I'+label[1:]
                            label_ids.extend([label_map[label]] + [label_map[pad_label]] * (len(word_tokens) - 1))
                            subtoken_ids.extend([1] + [0] * (len(word_tokens) - 1))
                        else: # 'I' and 'O'
                            label_ids.extend([label_map[label]]*len(word_tokens))
                            subtoken_ids.extend([1] + [0] * (len(word_tokens) - 1))
                    elif subtoken_label_type=='repeat':
                        label_ids.extend([label_map[label]]*len(word_tokens))
                        subtoken_ids.extend([1] + [0] * (len(word_tokens) - 1))
                    elif subtoken_label_type=='O':
                        label_ids.extend([label_map[label]] + [label_map['O']] * (len(word_tokens) - 1))
                        subtoken_ids.extend([1] + [0] * (len(word_tokens) - 1))                        
                else:                    
                    label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                    subtoken_ids.extend([1] + [0] * (len(word_tokens) - 1))


        if len(tokens) > max_len:
            max_len=len(tokens)
            
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length -2)]
            label_ids = label_ids[: (max_seq_length -2)]
            subtoken_ids=subtoken_ids[:(max_seq_length -2)]
        
        if omit_sep_cls_token:
            segment_ids = [sequence_a_segment_id] * len(tokens)
            
        elif label_sep_cls:
            tokens += [sep_token]
            label_ids += [label_map['SEP']]
            segment_ids = [sequence_a_segment_id] * len(tokens)
            subtoken_ids+=[-1]

            tokens = [cls_token] + tokens
            label_ids = [label_map['CLS']] + label_ids
            segment_ids = [sequence_a_segment_id] + segment_ids
            subtoken_ids=[-1]+subtoken_ids

            
        else:
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            segment_ids = [sequence_a_segment_id] * len(tokens)
            subtoken_ids+=[-1]

            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids
            subtoken_ids=[-1]+subtoken_ids
        
            
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        if label_sep_cls:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [label_map['PAD']] * padding_length  
            subtoken_ids+=[-1]*padding_length
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length
            subtoken_ids+=[-1]*padding_length
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(subtoken_ids)==max_seq_length
        

        if ex_index < 2:
            print("******"*10)
            print("*** Example ***")
            print("guid: %s", example.guid)
            print("tokens: %s", " ".join([str(x) for x in tokens]))
            print("input_ids: %s", " ".join([str(x) for x in input_ids]))
            print("input_mask: %s", " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            print("label_ids: %s", " ".join([str(x) for x in label_ids]))
            print("subtoken_ids: %s", " ".join([str(x) for x in subtoken_ids]))
        try:
            sent_id = int(example.guid.split('-')[1])
            assert sent_id==ex_index,('sent_id',sent_id,'ex_index',ex_index,'example.guid',example.guid)
        except:
            print(example.guid)
            print(example.words)
            print(example.labels)
            
            

        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids, subtoken_ids=subtoken_ids,sent_id = sent_id)
        )
    print('=*'*40)
    print('max_len',max_len)
    return features

def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format( mode))
    guid_index = 0
    examples = []
  
    para =  None
 
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        pos=[]
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(
                        guid="{}-{}".format(mode, guid_index), words=words,  labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
                    pos=[]
            else:
                splits = line.split(" ")
                words.append(splits[0])
             
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
              
                    labels.append("O")
        if words:
            examples.append(InputExample(
                guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
   
    return examples

def train(args,args2, train_dataset, model, tokenizer):
    """ Train the model """
    
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    #print(args)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join("bert-base-cased", "optimizer.pt")) and os.path.isfile(
        os.path.join("bert-base-cased", "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join("bert-base-cased", "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join("bert-base-cased", "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    #10 25 if os.path.exists("bert-base-cased"):
    #10 25    # set global_step to global_step of last saved checkpoint from model path
    #10 25    try:
    #10 25        global_step = int("bert-base-cased".split("-")[-1].split("/")[0])
    #10 25    except ValueError:
    #10 25        global_step = 0
    #10 25    epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
    #10 25    steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

    #10 25    logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #10 25    logger.info("  Continuing training from epoch %d", epochs_trained)
    #10 25    logger.info("  Continuing training from global step %d", global_step)
    #10 25    logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    global best_f1
    best_f1 = 0

    ##
    # initialize embedding delta

    #delta_global_embedding = torch.zeros([119547, args.hidden_size]).uniform_(-1,1)

    # 30522 bert
    # 50265 roberta
    # 21128 bert-chinese

    dims = torch.tensor([args.hidden_size]).float() # (768^(1/2))
    mag = args.adv_init_mag / torch.sqrt(dims) # 1 const (small const to init delta)
    #delta_global_embedding = (delta_global_embedding * mag.view(1, 1))
    #delta_global_embedding = delta_global_embedding.to(args.device)

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0], ncols=80)
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            # adaptive seq len
            #max_seq_len = torch.max(torch.sum(batch[1], 1)).item()#
          
            #batch = [t[:, :max_seq_len] for t in batch[:3]] + [batch[3]]#

            # BERT -only
            inputs = {"attention_mask": batch[1], "token_type_ids":batch[2], "labels": batch[3]}

            # Adv-Train

            # initialize delta
            input_ids = batch[0]
            input_ids_flat = input_ids.contiguous().view(-1)

            embeds_init=model.bert.embeddings.word_embeddings(input_ids)
            #embeds_init=model.module.bert.embeddings.word_embeddings(input_ids)
       

            # embeds_init = embeds_init.clone().detach()
            input_mask = inputs['attention_mask'].float()
            input_lengths = torch.sum(input_mask, 1) # B 

            bs,seq_len = embeds_init.size(0), embeds_init.size(1)

            #
            delta_lb, delta_tok, total_delta = None, None, None
            

            dims = input_lengths * embeds_init.size(-1) # B x(768^(1/2))
            mag = args.adv_init_mag / torch.sqrt(dims) # B
            delta_lb = torch.zeros_like(embeds_init).uniform_(-1,1) * input_mask.unsqueeze(2)
            delta_lb.requires_grad_()
            
            delta_lb = (delta_lb * mag.view(-1, 1, 1)).detach()



            #gathered = torch.index_select(delta_global_embedding, 0, input_ids_flat) # B*seq-len D
            #print("shape of gathered")
            #print(gathered.size())
            
            #gathered= torch.rand([4096, 768])
            #gathered= gathered.to(args.device)
            #delta_tok = gathered.view(bs, seq_len, -1).detach() # B seq-len D
            
            delta_tok = torch.rand([bs, seq_len, 768]).detach()
            delta_tok = delta_tok.to(args.device)
            #print("shape of delta_tok")
            #print(delta_tok.size())
            delta_tok.requires_grad_()
            denorm = torch.norm(delta_tok.view(-1,delta_tok.size(-1))).view(-1, 1, 1)
            delta_tok = delta_tok / denorm # B seq-len D  normalize delta obtained from global embedding

            # B seq-len 1

            if args.adv_train == 0:
                # inputs['inputs_embeds'] = embeds_init
                inputs['input_ids'] = input_ids
                outputs = model(**inputs)
                loss = outputs[0]

                # 1) loss backward

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                tr_loss += loss.item()
                loss.backward()

            else:

                # Adversarial-Training Loop
                for astep in range(args.adv_steps):

                                    
                    # craft input embedding
                    delta_lb.requires_grad_()
                    delta_tok.requires_grad_()

                    inputs_embeds = embeds_init + delta_lb + delta_tok

                    inputs['inputs_embeds'] = inputs_embeds
                        
                    outputs= model(token_type_ids=None, attention_mask= batch[1],     labels= batch[3], inputs_embeds=inputs_embeds)
                    #outputs = model(inputs)

                    loss = outputs[0]  # model outputs are always tuple in transformers (see doc)


                    # 1) loss backward

                    if args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    tr_loss += loss.item()
                    delta_lb.retain_grad()
                    delta_tok.retain_grad()
                    loss.backward(retain_graph=True)

                    if astep == args.adv_steps - 1:
                        # further updates on delta

                        delta_tok = delta_tok.detach()
                        #delta_global_embedding = delta_global_embedding.index_put_((input_ids_flat,), delta_tok, True)

                        break

                    # 2) get grad on delta
                    if delta_lb is not None:
                        delta_lb_grad = delta_lb.grad.clone().detach()
                    if delta_tok is not None:
                        delta_tok_grad = delta_tok.grad.clone().detach()


                    # 3) update and clip

                        
                    denorm_lb = torch.norm(delta_lb_grad.view(bs, -1), dim=1).view(-1, 1, 1)
                    denorm_lb = torch.clamp(denorm_lb, min=1e-8)
                    denorm_lb = denorm_lb.view(bs, 1, 1)


                    denorm_tok = torch.norm(delta_tok_grad, dim=-1) # B seq-len 
                    denorm_tok = torch.clamp(denorm_tok, min=1e-8)
                    denorm_tok = denorm_tok.view(bs, seq_len, 1) # B seq-len 1


                    delta_lb = (delta_lb + args.adv_lr * delta_lb_grad / denorm_lb).detach()
                    delta_tok = (delta_tok + args.adv_lr * delta_tok_grad / denorm_tok).detach()

                    # calculate clip

                    delta_norm_tok = torch.norm(delta_tok, p=2, dim=-1).detach() # B seq-len
                    mean_norm_tok, _ = torch.max(delta_norm_tok, dim=-1, keepdim=True) # B,1 
                    reweights_tok = (delta_norm_tok / mean_norm_tok).view(bs, seq_len, 1) # B seq-len, 1

                    delta_tok = delta_tok * reweights_tok

                    total_delta = delta_tok + delta_lb

                    delta_norm = torch.norm(total_delta.view(bs, -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > args.adv_max_norm).to(embeds_init)
                    reweights = (args.adv_max_norm / delta_norm * exceed_mask \
                                 + (1-exceed_mask)).view(-1, 1, 1) # B 1 1

                    # clip

                    delta_lb = (delta_lb * reweights).detach()
                    delta_tok = (delta_tok * reweights).detach()


            # *************************** END *******************
            # End (2) '''

                

            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(epoch_iterator) <= args.gradient_accumulation_steps
                and (step + 1) == len(epoch_iterator)
            ):
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    #logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        labels=["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
                        pad_token_label_id = CrossEntropyLoss().ignore_index
                        results,_ = evaluate(args, args2, model, tokenizer, pad_token_label_id,  parallel = False, mode="dev", prefix = str(global_step))
                        #print(results)
                        #for key, value in results.items():
                        #    eval_key = "eval_{}".format(key)
                        #    logs[eval_key] = value

                        logger.info("Model name: %s", args.output_dir)
                        logger.info("Epoch is %s", epoch)
                        if results['f1'] >= best_f1:
                            best_f1 = results['f1']
                            
                            
                            output_dir = os.path.join(args.output_dir, "best")
                           
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            logger.info("Saving best model to %s", output_dir)
                            logger.info("Epochs trained is %s", epoch)
                          
                            model_to_save = (
                                model.module if hasattr(model, "module") else model)  
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving model checkpoint to %s", output_dir)

                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)






                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    #logs["learning_rate"] = learning_rate_scalar
                    #logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    #for key, value in logs.items():
                    #    tb_writer.add_scalar(key, value, global_step)
                    #print(json.dumps({**logs, **{"step": global_step}}))

                if True==False:#if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        #########################
        if (args.evaluate_during_training):
            
            labels=["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
            pad_token_label_id = CrossEntropyLoss().ignore_index
            results,_ = evaluate(args, args2, model, tokenizer, pad_token_label_id,parallel = False, mode="dev", prefix = str(global_step))
            #for key, value in results.items():
            #    tb_writer.add_scalar("eval_{}".format(key), value, global_step)

            
            if results['f1'] >= best_f1:
                best_f1 = results['f1']
              
                
                output_dir = os.path.join(args.output_dir, "best")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
              
                logger.info("Saving best model to %s", output_dir)
              
                logger.info("Epoch is %s", epoch)
                model_to_save = (model.module if hasattr(model, "module") else model)  
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)

                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)



        ################################
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()
    # print('saving gloabl embedding')
    # torch.save(delta_global_embedding, os.path.join("global_embedding.pt"))
    return global_step, tr_loss / global_step


 



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

  
    model_type: str = field(metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_TYPES)})
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pre-trained models downloaded from s3"}
    )


@dataclass
class DataProcessingArguments:
    task_name: str = field(
        metadata={"help": "The name of the task to train selected in the list: " + ", ".join(processors.keys())}
    )
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

@dataclass
class ATArgs:
    adv_train: int = field()
    adv_steps: int = field()
    adv_init_mag: float = field()
    adv_max_norm: float = field()
    adv_lr: float = field()
    data_size: int = field(default=0)
    vocab_size: int = field(default=28996)
    hidden_size: int = field(default=768)
    


def main():
    parser = HfArgumentParser((ModelArguments, DataProcessingArguments, TrainingArguments, ATArgs))
    model_args, dataprocessing_args, training_args, at_args = parser.parse_args_into_dataclasses()

    # For now, let's merge all the sets of args into one,
    # but soon, we'll keep distinct sets of args, with a cleaner separation of concerns.


    args = argparse.Namespace(**vars(model_args), **vars(dataprocessing_args), **vars(training_args), **vars(at_args))
    args2=OtherArgs()
    
    if (os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir):
        raise ValueError( "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    
    logger.setLevel(log.INFO)
    formatter = log.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    
    if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
            
    fh = log.FileHandler(args.output_dir  +'/'  + 'log.txt')
    fh.setLevel(log.INFO)
    fh.setFormatter(formatter)

    ch = log.StreamHandler()
    ch.setLevel(log.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    
    logger.info("------NEW RUN-----")

    

   
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    #####################################################
    #Before edit 10/24/21
    #tokenizer = BertTokenizer.from_pretrained(
    #    'bert-base-uncased'
    #)

    #model = BertForTokenClassification.from_pretrained(
    #    "bert-base-uncased", 
    #    num_labels = 9, 
    #    output_attentions = False,
    #    output_hidden_states = False, 
    #)
    ########################################################
    config = BertConfig.from_pretrained(
        args2.config_name if args2.config_name else args2.model_name,
        num_labels=num_labels,
    )
    tokenizer = BertTokenizer.from_pretrained(
        args2.tokenizer_name if args2.tokenizer_name else args2.model_name,
        do_lower_case=args2.do_lower_case,
    )
    model = BertForTokenClassification.from_pretrained(
        args2.model_name,
        from_tf=bool(".ckpt" in args2.model_name),
        config=config)











    best_f1 = 0

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        #train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False, data_size=args.data_size)
        train_dataset = read_data( tokenizer=tokenizer, labels=label_list, 
                                                           pad_token_label_id=CrossEntropyLoss().ignore_index, mode = 'train',
                                                          pad_subtoken_with_real_label=args2.pad_subtoken_with_real_label) 
        global_step, tr_loss = train(args,args2, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)





    print("Doing Predict!!!!")
    test_dataset_regular = read_data(tokenizer, labels=label_list, pad_token_label_id=CrossEntropyLoss().ignore_index, mode = 'test', pad_subtoken_with_real_label=args2.pad_subtoken_with_real_label)

    #test_dataset_challenging = read_data_rule_based_aug(args, args2, tokenizer, labels=label_list, pad_token_label_id=CrossEntropyLoss().ignore_index, mode = 'test', pad_subtoken_with_real_label=args2.pad_subtoken_with_real_label)
    
    output_dir = os.path.join(args.output_dir, "best")
    tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=False)
    

    model = BertForTokenClassification.from_pretrained(output_dir)
    model.to(args.device)
    pad_token_label_id = CrossEntropyLoss().ignore_index
    result, predictions = evaluate(args, args2, model, tokenizer, pad_token_label_id ,test_dataset_regular,  mode="test", prefix = 'final')
    
    print("Regular test set results: ",result)
    #result, predictions = evaluate(args, args2, model, tokenizer,  pad_token_label_id ,test_dataset_challenging,  mode="test", prefix = 'final')
    #print("Challenging test set results: ",result)











######################################################################################################
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    #10/24 if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #10/24    # Create output directory if needed
    #10/24    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    #10/24        os.makedirs(args.output_dir)

    #10/24    logger.info("Saving model checkpoint to %s", args.output_dir)
    #10/24    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    #10/24    # They can then be reloaded using `from_pretrained()`
    #10/24    model_to_save = (
    #10/24        model.module if hasattr(model, "module") else model
    #10/24    )  # Take care of distributed/parallel training
    #10/24#10/24    model_to_save.save_pretrained(args.output_dir)
    #10/24#10/24    tokenizer.save_pretrained(args.output_dir)

    #10/24    # Good practice: save your training arguments together with the trained model
    #10/24    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    #10/24    # Load a trained model and vocabulary that you have fine-tuned
    #10/24    model = AutoModelForTokenClassification.from_pretrained(args.output_dir)
    #10/24    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    #10/24    model.to(args.device)

    # Evaluation
    #10/24results = {}
    #if args.do_eval and args.local_rank in [-1, 0]:
    #tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    #10/24checkpoints = [args.output_dir]
    #10/24if args.eval_all_checkpoints:
    #10/24    checkpoints = list(
    #10/24        os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
    #10/24    )
    #10/24    logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #10/24logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #10/24for checkpoint in checkpoints:
    #10/24    global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
    #10/24    prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

    #10/24    model = AutoModelForTokenClassification.from_pretrained(checkpoint)
    #10/24    model.to(args.device)
    #10/24    result = evaluate(args, model, tokenizer, prefix=prefix)
    #10/24    result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
    #10/24   results.update(result)


class OtherArgs:
    model_name= 'bert-base-cased'
    weight_decay=0.0
    adam_epsilon=1e-8
    do_lower_case=False
    max_grad_norm=1.0
    learning_rate=5e-5
    max_steps=-1
    warmup_steps=0
    #warmup_steps=5000
    logging_steps=150
    #logging_steps=1000
    gradient_accumulation_steps=1
    eval_batch_size=128
    labels='./data/conll2003/labels.txt'
    pad_subtoken_with_real_label=True
    subtoken_label_type='real'
    label_sep_cls=True
    evaluate_during_training=True
    eval_pad_subtoken_with_first_subtoken_only=True
    tokenizer_name=''
    config_name=''
def evaluate(args,args2,model, tokenizer,  pad_token_label_id,  eval_dataset = None, parallel = True, mode = 'dev', prefix = ''):
    labels = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    if eval_dataset is None and mode=='dev':
        eval_dataset = read_data( tokenizer, labels, pad_token_label_id, mode = mode)

    eval_dataloader = DataLoader(eval_dataset, batch_size = args2.eval_batch_size, shuffle = False)

    if parallel:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s *****", mode + '-' + prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args2.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    all_subtoken_ids=None
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3],'sent_id' : batch[5]}
            inputs["token_type_ids"] = batch[2]
            target=inputs['labels']
            
            result  = model(inputs['input_ids'], 
                            token_type_ids=None, 
                            attention_mask=inputs["attention_mask"])
     
            logits=result[0]

          
        nb_eval_steps += 1
        
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            all_subtoken_ids=batch[4].detach().cpu().numpy()
            sent_id=inputs['sent_id'].detach().cpu().numpy()
            input_ids=inputs['input_ids'].detach().cpu().numpy()
        else:
       
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            all_subtoken_ids = np.append(all_subtoken_ids, batch[4].detach().cpu().numpy(), axis=0)
            sent_id = np.append(sent_id, inputs['sent_id'].detach().cpu().numpy(), axis=0)
            input_ids= np.append(input_ids, inputs['input_ids'].detach().cpu().numpy(), axis=0)

    #eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]
    input_id_list = [[] for _ in range(input_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if args2.pad_subtoken_with_real_label  or args2.label_sep_cls:

                if args2.eval_pad_subtoken_with_first_subtoken_only:
                    if all_subtoken_ids[i,j] ==1: 
                        out_label_list[i].append(label_map[out_label_ids[i][j]])
                        preds_list[i].append(label_map[preds[i][j]])
                        tid=input_ids[i][j]
                        input_id_list[i].append(tokenizer.convert_ids_to_tokens([tid])[0])


                else:
                    if all_subtoken_ids[i,j] in [0,1] and out_label_ids[i, j] != pad_token_label_id:# in this case, we consider all the tokens.
                        out_label_list[i].append(label_map[out_label_ids[i][j]])
                        preds_list[i].append(label_map[preds[i][j]])            
                        input_id_list[i].append(tokenizer.convert_ids_to_tokens([input_ids[i][j]]))    
            else:
                if all_subtoken_ids[i,j] in [0,1] and out_label_ids[i, j] != pad_token_label_id:# in this case, we consider all the tokens.
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])                
                    input_id_list[i].append(tokenizer.convert_ids_to_tokens([input_ids[i][j]]))
    file_name=os.path.join(args.output_dir,'{}_pred_results.tsv'.format(mode))
    #output_eval_results(out_label_list,preds_list,input_id_list,file_name)
    from eval_utils import f1_score as f1_scoree, precision_score, recall_score, classification_report, macro_score as macro_scoree
    macro_scores=macro_scoree(out_label_list, preds_list)
    results = {
      
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_scoree(out_label_list, preds_list),
        'macro_f1':macro_scores['macro_f1'],
        'macro_precision':macro_scores['macro_precision'],
        'macro_recall':macro_scores['macro_recall']
    }

    logger.info("***** Eval results %s *****", mode + '-' + prefix)

    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list
def read_data_rule_based_aug(args,args2, tokenizer, labels, pad_token_label_id, mode,  
              omit_sep_cls_token=False,
              pad_subtoken_with_real_label=False):
    logger.info("Creating features from dataset file at %s", args.data_dir)
    examples = read_examples_from_file_excel("data/conll2003",mode) 
    
    if mode =='train':       
        augexamples=[]
        if args.augmented_train_percentage>0:
            file_name="augexamples"+str(args.augmented_train_percentage)+"percent.pkl"
        else:
            augmented_train_percentage=-1*args.augmented_train_percentage
            file_name="zeroshotaugexamples"+str(augmented_train_percentage)+"percent_no_held_out_phrases.pkl"
        
            file_path = os.path.join(args.data_dir, file_name)
            open_file = open(file_path, "rb")
            augexamples = pickle.load(open_file)
            open_file.close()
    
            logger.info("Number of Augmented Examples: %s", len(augexamples))    
                    
            for augexample in augexamples:
                
                augsentid=len(examples)

                augexample.guid="train-"+str(augsentid)
            
                examples.append(augexample)
        
      


    elif mode =='test':
        examples = read_examples_from_file_excel("data/conll2003",mode)

    if  mode =='train':
        #examples = examples[0]
        print(mode)
        print('data num: {}'.format(len(examples)))

        features = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer, 
                                    cls_token = tokenizer.cls_token, sep_token =  tokenizer.sep_token, pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                    cls_token_segment_id = 2 if args.model_type in ["xlnet"] else 0, 
                                    sequence_a_segment_id = 0, pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0, 
                                    pad_token_label_id = pad_token_label_id,
                                    omit_sep_cls_token=omit_sep_cls_token,
                                    pad_subtoken_with_real_label=pad_subtoken_with_real_label,
                                    subtoken_label_type=args2.subtoken_label_type,
                                    label_sep_cls=args2.label_sep_cls)
        
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_subtoken_ids = torch.tensor([f.subtoken_ids for f in features], dtype=torch.long)
        all_sent_id = torch.tensor([f.sent_id for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_subtoken_ids,all_sent_id)
        
      
        
        return dataset
    

    if  mode =='test':
        #examples = examples[0]
        print(mode)
        print('data num: {}'.format(len(examples)))

        features = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer, 
                                    cls_token = tokenizer.cls_token, sep_token =  tokenizer.sep_token, pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                    cls_token_segment_id = 2 if args.model_type in ["xlnet"] else 0, 
                                    sequence_a_segment_id = 0, pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0, 
                                    pad_token_label_id = pad_token_label_id,
                                    omit_sep_cls_token=omit_sep_cls_token,
                                    pad_subtoken_with_real_label=pad_subtoken_with_real_label,
                                    subtoken_label_type=args2.subtoken_label_type,
                                    label_sep_cls=args2.label_sep_cls)
        
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_subtoken_ids = torch.tensor([f.subtoken_ids for f in features], dtype=torch.long)
        all_sent_id = torch.tensor([f.sent_id for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_subtoken_ids,all_sent_id)
        
        return dataset

def read_examples_from_file_excel(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 0
    examples = []

                

    fromexcel=True  
    if fromexcel==False or mode !='test':
        with open(file_path, encoding="utf-8") as f:
            words = []
            labels = []
            pos=[]
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        examples.append(InputExample(
                            guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
                        guid_index += 1
                        words = []
                        labels = []
                        pos=[]
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    pos.append(splits[1])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                examples.append(InputExample(
                    guid="{}-{}".format(mode, guid_index), words=words,labels=labels))
    else:
    
        import pandas as pd

        file_path = os.path.join(data_dir, "{}.xlsx".format("Challenge Set"))
        train = pd.read_excel(file_path)
        firstrun=True
        exampleindex=5
        examples = []
        guidloopstop=''
        guid_index=0
        while guidloopstop!='test-3442':

            if firstrun:
                guidloopstop='test-2'
                qualityr=train.iloc[0]     
                quality=qualityr.iloc[1]
                augtyper=train.iloc[1]     
                augtype=augtyper.iloc[1]
                wordr=train.iloc[2]  
                words=[]
                labelsr=train.iloc[3]  
                labels=[]
                i=1
                while 1==1:
                    c=wordr.iloc[i]
                    label=labelsr.iloc[i]
                    if c!='^':
                        words.append(c)
                        labels.append(label)
                        i=i+1
                    else:
                        break
                if    quality==1:
                    examples.append(InputExample(
                        guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
                    guid_index += 1
                firstrun=False
            else:
               
                guidr=train.iloc[exampleindex]     
                guidloopstop=guidr.iloc[1]
             
               
                qualityr=train.iloc[exampleindex+1]     
                quality=qualityr.iloc[1]
                augtyper=train.iloc[exampleindex+2]     
                augtype=augtyper.iloc[1]
                wordr=train.iloc[exampleindex+3]  
                words=[]
                labelsr=train.iloc[exampleindex+4]  
                labels=[]
                i=1
                while 1==1:
                    c=wordr.iloc[i]
                    label=labelsr.iloc[i]
                    if c!='^':
                        if c=='s':
                            c='\'s'
                        words.append(c)
                        labels.append(label)
                        i=i+1
                    else:
                        break
                if    quality==1:
                    examples.append(InputExample(
                        guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
                    guid_index += 1
                exampleindex=exampleindex+6
            

    print("Number of examples ",len(examples))
    return examples

def output_eval_results(out_label_list,preds_list,input_id_list,file_name):
    with open(file_name,'w') as fout:
        for i in range(len(out_label_list)):
            label=out_label_list[i]
            pred=preds_list[i]
            tokens=input_id_list[i]
            for j in range(len(label)):
                if tokens[j]=='[PAD]':
                    continue
                fout.write('{}\t{}\t{}\n'.format(tokens[j] ,label[j],pred[j]))
            fout.write('\n')
if __name__ == "__main__":
    main()





