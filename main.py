import argparse
import json
import logging
import math
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset, Dataset

from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from transformers import set_seed, get_scheduler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

from peft import LoraConfig, TaskType, get_peft_model


class Dataset2Tensor(Dataset):
    def __init__(self, dataset, examples, tokenizer, max_seq_length):
        super().__init__()
        self.data = []
        if dataset == 'sst2':
            for example in examples:
                text_a = example['sentence']
                text_b = None
                template = (text_a, text_b)
                input = tokenizer(*template, max_length=max_seq_length, padding='max_length', truncation=True, return_tensors='pt')
                label = torch.tensor(example['label'], dtype=torch.long)
                self.data.append({
                    'input_ids': input['input_ids'][0],
                    'attention_mask': input['attention_mask'][0],
                    'labels': label,
                })
        elif dataset == 'qqp':
            for example in examples:
                text_a = example['question1']
                text_b = example['question2']
                template = (text_a, text_b)
                input = tokenizer(*template, max_length=max_seq_length, padding='max_length', truncation=True, return_tensors='pt')
                label = torch.tensor(example['label'], dtype=torch.long)
                self.data.append({
                    'input_ids': input['input_ids'][0],
                    'attention_mask': input['attention_mask'][0],
                    'labels': label,
                })
        elif dataset == 'mnli':
            for example in examples:
                text_a = example['premise']
                text_b = example['hypothesis']
                template = (text_a, text_b)
                input = tokenizer(*template, max_length=max_seq_length, padding='max_length', truncation=True, return_tensors='pt')
                label = torch.tensor(example['label'], dtype=torch.long)
                self.data.append({
                    'input_ids': input['input_ids'][0],
                    'attention_mask': input['attention_mask'][0],
                    'labels': label,
                })
        else:
            raise ValueError(dataset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Client:
    def __init__(self, args, id, model_name, train_dataset=None, eval_dataset=None):
        self.args = args
        self.id = id
        self.name = 'client{}'.format(id)

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # PLMs
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=args.num_labels)

        # Hyperparameters
        self.device = args.device
        self.max_seq_length = args.max_seq_length
        self.batch_size = args.batch_size
        self.epochs = args.E_k
        self.lr = args.lr_k
        self.tao = args.tao

    def local_update(self):
        self.model.to(self.device)

        train_dataset = Dataset2Tensor(self.args.dataset, self.train_dataset, self.tokenizer, self.max_seq_length)
        eval_dataset = Dataset2Tensor(self.args.dataset, self.eval_dataset[0], self.tokenizer, self.max_seq_length)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
        eval_loader = DataLoader(eval_dataset, shuffle=False, batch_size=self.batch_size)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        # scheduler = get_scheduler('linear', optimizer=optimizer, num_training_steps=len(train_loader) * self.epochs)

        best_acc = -1
        for epoch in range(1, self.epochs + 1):
            # Training
            self.model.train()
            train_loss = 0
            train_ground_truths, train_predictions = [], []
            for batch in tqdm(train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                loss.backward()
                optimizer.step()
                # scheduler.step()

                train_loss += loss.item()
                train_predictions.extend(torch.argmax(logits, dim=-1).tolist())
                train_ground_truths.extend(batch['labels'].tolist())
            train_loss /= len(train_loader)
            train_acc = accuracy_score(y_pred=train_predictions, y_true=train_ground_truths)

            # Testing
            self.model.eval()
            eval_loss = 0
            eval_ground_truths, eval_predictions = [], []
            for batch in eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits

                eval_loss += loss.item()
                eval_predictions.extend(torch.argmax(logits, dim=-1).tolist())
                eval_ground_truths.extend(batch['labels'].tolist())
            eval_loss /= len(eval_loader)
            eval_acc = accuracy_score(y_pred=eval_predictions, y_true=eval_ground_truths)

            if eval_acc > best_acc:
                best_acc = eval_acc
                model_dir = os.path.join(args.output_dir, f'{self.name}_ {self.model_name}')
                os.makedirs(model_dir, exist_ok=True)
                self.model.save_pretrained(model_dir)
                self.tokenizer.save_pretrained(model_dir)

            if self.args.dataset == 'qqp':
                train_f1 = f1_score(y_pred=train_predictions, y_true=train_ground_truths)
                eval_f1 = f1_score(y_pred=eval_predictions, y_true=eval_ground_truths)
                logging.info("Epoch: {}/{}\tTrain Loss: {:.4f}, Train Acc: {:.4f}, Train F1: {:.4f}, Eval Loss: {:.4f}, Eval Acc: {:.4f}, Eval F1: {:.4f}, *Best Acc: {:.4f}".format(epoch, self.epochs, train_loss, train_acc, train_f1, eval_loss, eval_acc, eval_f1, best_acc))
            elif self.args.dataset == 'mnli':
                eval_dataset = Dataset2Tensor(self.args.dataset, self.eval_dataset[1], self.tokenizer, self.max_seq_length)
                eval_loader = DataLoader(eval_dataset, shuffle=False, batch_size=self.batch_size)

                self.model.eval()
                mismatched_eval_loss = 0
                mismacthed_ground_truths, mismacthed_predictions = [], []
                for batch in eval_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = self.model(**batch)
                    loss = outputs.loss
                    logits = outputs.logits

                    mismatched_eval_loss += loss.item()
                    mismacthed_predictions.extend(torch.argmax(logits, dim=-1).tolist())
                    mismacthed_ground_truths.extend(batch['labels'].tolist())
                mismatched_eval_loss /= len(eval_loader)
                mismatched_eval_acc = accuracy_score(y_pred=mismacthed_predictions, y_true=mismacthed_ground_truths)
                logging.info("Epoch: {}/{}\tTrain Loss: {:.4f}, Train Acc: {:.4f}, Eval Loss-m: {:.4f}, Eval Acc-m: {:.4f}, Eval Loss-mm: {:.4f}, Eval Acc-mm: {:.4f},*Best Acc: {:.4f}".format(epoch, self.epochs, train_loss, train_acc, eval_loss, eval_acc, mismatched_eval_loss, mismatched_eval_acc, best_acc))
            else:
                logging.info("Epoch: {}/{}\tTrain Loss: {:.4f}, Train Acc: {:.4f}, Eval Loss: {:.4f}, Eval Acc: {:.4f}, *Best Acc: {:.4f}".format(epoch, self.epochs, train_loss, train_acc, eval_loss, eval_acc, best_acc))

        self.model.cpu()
        return best_acc

    def compute_logits(self, pseudo_sentences):
        self.model.to(self.device)

        self.model.eval()
        pseudo_inputs = self.tokenizer.batch_encode_plus(pseudo_sentences, max_length=self.max_seq_length, padding=True, truncation=True, return_tensors='pt')
        pseudo_inputs = {k: v.to(self.device) for k, v in pseudo_inputs.items()}
        with torch.no_grad():
            logits = self.model(input_ids=pseudo_inputs['input_ids'], attention_mask=pseudo_inputs['attention_mask']).logits

        self.model.cpu()
        return logits

    def local_distillation(self, pseudo_sentences, pseudo_labels, ensemble_logits):
        self.model.to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        self.model.train()
        pseudo_inputs = self.tokenizer.batch_encode_plus(pseudo_sentences, max_length=self.max_seq_length, padding=True, truncation=True, return_tensors='pt')
        pseudo_inputs = {k: v.to(self.device) for k, v in pseudo_inputs.items()}
        optimizer.zero_grad()
        logits = self.model(input_ids=pseudo_inputs['input_ids'], attention_mask=pseudo_inputs['attention_mask']).logits
        loss = F.cross_entropy(logits, pseudo_labels) + F.kl_div(F.log_softmax(logits / args.tao, dim=-1), F.softmax(ensemble_logits.to(self.device) / args.tao, dim=-1), reduction='batchmean') * (args.tao ** 2)
        loss.backward()
        optimizer.step()

        self.model.cpu()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Federated
    parser.add_argument("--algorithm", default="FedSTG", type=str, help="Type of algorithms")
    parser.add_argument("--seed", default=0, type=int, help="Random seed for initialization")
    parser.add_argument("--K", default=10, type=int, help="Number of clients: K")
    parser.add_argument("--C", default=1, type=float, help="Fraction of clients: C")
    parser.add_argument("--T", default=10, type=int, help="Number of communication rounds: T")
    # Data
    parser.add_argument("--dataset", default="sst2", type=str, choices=['sst2', 'qqp', 'mnli'], help="Type of datasets")
    parser.add_argument("--max_seq_length", default=128, type=int, help="Maximum sequence length")
    parser.add_argument("--batch_size", default=16, type=int, help="Input batch size")
    parser.add_argument("--partition", default="dirichlet", type=str, choices=['iid', 'dirichlet'], help="iid data or non-iid data with Dirichlet distribution")
    parser.add_argument("--alpha", default=1, type=float, help="Ratio of Dirichlet distribution")
    # Model
    parser.add_argument("--global_model", default="bert-base-uncased", type=str, help="Type of global model")
    parser.add_argument("--local_models", default="bert-base-uncased", type=str, choices=['distilbert-base-uncased', 'bert-base-uncased'], help="Type of local model")
    parser.add_argument("--generator", default='LLama-3.1-8B-Instruct', type=str, choices=['Llama-3.2-1B-Instruct', 'Llama-3.2-3B-Instruct', 'Llama-3.1-8B-Instruct'], help="Type of generator")
    # Optimization
    # parser.add_argument("--optimizer", default="adamw", type=str, choices=['adam', 'adamw'], help="Type of optimizer")
    # parser.add_argument("--scheduler", default="linear", type=str, choices=['linear', 'cosine'], help="Type of scheduler")
    parser.add_argument("--lr", default=2e-5, type=float, help="Learning rate of the global model: η")
    parser.add_argument("--lr_k", default=2e-5, type=float, help="Learning rate of the local model: η_k")
    parser.add_argument("--lr_g", default=3e-6, type=float, help="Learning rate of the generative model: η_g")
    # parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay if we apply")
    # parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum for optimizer')
    # parser.add_argument("--warmup_steps", type=int, default=0, help="Step of training to perform learning rate warmup for if set for cosine and linear decay")
    # parser.add_argument("--E", default=3, type=int, help="Number of global update epochs: E")
    parser.add_argument("--E_k", default=3, type=int, help="Number of local update epochs: E_k")
    # parser.add_argument("--E_k_dis", default=3, type=int, help="Number of local distillation epochs: E'_k")
    # parser.add_argument("--E_g", default=3, type=int, help="Number of generator update epochs: E_g")
    # Customized
    parser.add_argument("--tao", default=1, type=float, help="Temperature for distillation")
    # parser.add_argument("--N_pub", default=5000, type=int, help="Number of public proxy data")
    parser.add_argument("--N_syn", default=25000, type=int, help="Number of synthetic data")
    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=100, help="maximum length of the output sequence")
    parser.add_argument("--do_sample", type=bool, default=True, help="whether to sample the output sequence or take the maximum probability")
    parser.add_argument("--top_k", type=int, default=50, help="top k sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="top p sampling")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="the number of sequences to output")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3, help="if set, ngrams containing the same last-n tokens will be filtered out")
    parser.add_argument("--temperature", type=float, default=0.8, help="temperature for top-k sampling")
    # parser.add_argument("--early_stopping", type=bool, default=True, help="whether to stop decoding when all ngrams have been filtered")
    # Output
    parser.add_argument("--output_dir", default="runs", type=str, help="The output directory where checkpoints/results/logs will be written.")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set dir
    args.output_dir = os.path.join(args.output_dir, args.dataset, args.algorithm)
    os.makedirs(args.output_dir, exist_ok=True)

    # Set log
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="[%(levelname)s](%(asctime)s) %(message)s",
                        datefmt="%Y/%m/%d %H:%M:%S",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, 'log.txt')), logging.StreamHandler(sys.stdout)])

    # Set data
    if args.dataset == 'sst2':
        raw_datasets = load_dataset("glue", "sst2")
        train_dataset = raw_datasets['train']  # 67349
        eval_dataset = raw_datasets['validation']  # 872
        eval_dataset = (eval_dataset,)
        del raw_datasets

        context = "You are a professional synthetic text generator for sentiment analysis tasks."
        prompts = {
            "negative": [
                "Please generate a negative movie review without any extra explanation.",
            ],
            "positive": [
                "Please generate a positive movie review without any extra explanation.",
            ]
        }
    elif args.dataset == 'qqp':
        raw_datasets = load_dataset("glue", "qqp")
        train_dataset = raw_datasets['train']  # 363846
        eval_dataset = raw_datasets['validation']  # 40430
        eval_dataset = (eval_dataset,)
        del raw_datasets

        context = "You are a professional synthetic text generator for paraphrasing detection tasks."
        prompts = {
            "not_equivalent": [
                "Please generate two semantically unequal sentences separated by '|' without any extra explanation.",

            ],
            "equivalent": [
                "Please generate two semantically equivalent sentences separated by '|' without any extra explanation.",
            ]
        }
    elif args.dataset == 'mnli':
        raw_datasets = load_dataset("glue", "mnli")
        train_dataset = raw_datasets['train']  # 392702
        eval_matched_dataset = raw_datasets['validation_matched']  # 9815
        eval_mismatched_dataset = raw_datasets['validation_mismatched']  # 9832
        eval_dataset = (eval_matched_dataset, eval_mismatched_dataset)
        del raw_datasets

        context = "You are a professional synthetic text generator for natural language inference tasks."
        prompts = {
            "entailment": [
                "Please generate a premise sentence and a entailed hypothesis sentence separated by '|' without any extra explanation.",
            ],
            "neutral": [
                "Please generate a premise sentence and a neutral hypothesis sentence separated by '|' without any extra explanation.",
            ],
            "contradiction": [
                "Please generate a premise sentence and a contradictory hypothesis sentence separated by '|' without any extra explanation.",
            ]
        }
    else:
        raise NotImplementedError()
    args.int2class = {i: clas for i, clas in enumerate(prompts.keys())}
    args.class2int = {clas: i for i, clas in args.int2class.items()}
    args.num_labels = len(args.int2class)

    train_labels = np.array(train_dataset['label'])
    if args.partition == 'iid':  # iid distribution
        idxs = np.random.permutation(len(train_labels))
        idx_batch = np.array_split(idxs, args.K)
        client_dataidx_map = {k: idx_batch[k] for k in range(args.K)}
    elif args.partition == 'dirichlet':  # non-iid with Dirichlet distribution
        idx_batch = [[] for _ in range(args.K)]
        for c in range(args.num_labels):
            # get a list of batch indexes which are belong to label c
            idx_c = np.where(train_labels == c)[0]
            np.random.shuffle(idx_c)
            # using dirichlet distribution to determine the unbalanced proportion for each client (num_clients in total)
            proportions = np.random.dirichlet(np.repeat(args.alpha, args.K))
            # get the index in idx_c according to the dirichlet distribution
            proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            # generate the batch list for each client
            idx_batch = [idx_c_k + idx.tolist() for idx_c_k, idx in zip(idx_batch, np.split(idx_c, proportions))]
        client_dataidx_map = {}
        total = 0
        for k in range(args.K):
            np.random.shuffle(idx_batch[k])
            client_dataidx_map[k] = idx_batch[k]
            total += len(idx_batch[k])
        assert total == len(train_labels)
    else:
        raise ValueError(args.partition)

    local_datasets = {}
    for k, dataidx in client_dataidx_map.items():
        local_datasets[k] = Subset(train_dataset, indices=dataidx)

    client_cls_counts = {}
    for k, dataidx in client_dataidx_map.items():
        unq, unq_cnt = np.unique(train_labels[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        client_cls_counts[k] = tmp

    del train_dataset, train_labels
    assert len(local_datasets) == len(client_cls_counts) == args.K

    # Set model
    local_models = args.local_models.split(',')
    if len(local_models) == 1:
        local_models = [local_models[0]] * args.K
    clients = {k + 1: Client(args, id=k + 1, model_name=local_models[k], train_dataset=local_datasets[k], eval_dataset=eval_dataset) for k in range(args.K)}

    # PLM
    tokenizer = AutoTokenizer.from_pretrained(args.global_model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.global_model, num_labels=args.num_labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # LLM
    generator_tokenizer = AutoTokenizer.from_pretrained(f'meta-llama/{args.generator}')
    generator_tokenizer.pad_token_id = 0
    generator_tokenizer.bos_token_id = 1
    generator_tokenizer.eos_token_id = 2
    generator_tokenizer.padding_side = 'right'
    generator = AutoModelForCausalLM.from_pretrained(f'meta-llama/{args.generator}', torch_dtype=torch.bfloat16)
    generator.tie_weights()
    old_generator = AutoModelForCausalLM.from_pretrained(f'meta-llama/{args.generator}', torch_dtype=torch.bfloat16)
    old_generator.tie_weights()
    for param in old_generator.parameters():
        param.requires_grad = False
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                             r=16,
                             lora_alpha=32,
                             lora_dropout=0.1,
                             bias='none',
                             target_modules=['q_proj', 'v_proj'])
    generator = get_peft_model(generator, lora_config)
    old_generator = get_peft_model(old_generator, lora_config)
    generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=args.lr_g)

    # Federated training
    logger.info("Algorithm: {}".format(args.algorithm))
    logger.info("Device: {}".format(args.device))
    logger.info("Dataset: {}".format(args.dataset))
    if args.partition == "iid":
        logger.info("Partition: {}".format(args.partition))
    elif args.partition == "dirichlet":
        logger.info("Partition: {}, Alpha: {}".format(args.partition, args.alpha))
    logger.info("Number of clients: {}".format(args.K))
    logger.info("Number of train datasets: {}".format([len(local_datasets[k]) for k in range(args.K)]))
    logger.info("Data statistics: %s" % str(client_cls_counts))
    if len(eval_dataset) == 1:
        logger.info("Number of eval dataset: {}".format(len(eval_dataset[0])))
    else:
        logger.info("Number of eval dataset: {},\t{}".format(len(eval_dataset[0]), len(eval_dataset[1])))
    # logger.info("Number of public dataset: {}".format(len(public_dataset)))
    logger.info("Number of synthetic dataset: {}".format(args.N_syn))
    logger.info("Number of communication rounds: {}".format(args.T))
    logger.info("Number of local update epochs: {}".format(args.E_k))
    # if args.algorithm != 'FedAvg':
    #       logger.info("Number of global update epochs: {}".format(args.E))
    #       logger.info("Number of generator update epochs: {}".format(args.E_g))
    logger.info("Local models: {},\tGlobal model: {},\tGenerator: {}".format(', '.join(local_models), args.global_model, args.generator))
    for k in range(1, len(clients) + 1):
        logging.info('Model parameters of %s_%s: %2.2fM' % (clients[k].name, clients[k].model_name, (sum(p.numel() for p in clients[k].model.parameters()) / (1000 * 1000))))
    logging.info('Model parameters of %s_%s: %2.2fM' % ('server', args.global_model, (sum(p.numel() for p in model.parameters()) / (1000 * 1000))))
    logging.info('Model parameters of %s_%s: %2.2fB' % ('server', args.generator, (sum(p.numel() for p in generator.parameters()) / (1000 * 1000 * 1000))))

    current_accuracies = {k: None for k in range(args.K + 1)}
    test_accuracies = pd.DataFrame(columns=range(args.K + 1))
    communication_budgets = 0
    best_acc = -1

    for t in range(1, args.T + 1):
        logger.info('===============The {:d}-th round==============='.format(t))

        # the server randomly samples m = max(C*K, 1) active clients to participate federated training
        m = max(math.ceil(args.C * args.K), 1)
        selected_clients = sorted(np.random.choice(range(1, args.K + 1), m, replace=False))

        logger.info('#################### Client Update ####################')
        local_data_sizes = []
        for k in selected_clients:
            client = clients[k]
            logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_name))
            eval_acc = client.local_update()
            current_accuracies[k] = '{:.4f}'.format(eval_acc)
            local_data_sizes.append(len(client.train_dataset))  # get the quantity of clients joined in the federated training for updating the clients weights
        local_weights = [local_data_size / sum(local_data_sizes) for local_data_size in local_data_sizes]

        logger.info('#################### Server Update ####################')
        logger.info("# Node{:d}: {}_{}".format(0, 'server', args.global_model))
        # the server generates synthetic text by prompting the LLM
        labels = range(args.num_labels)
        train_dataset = []
        random_labels = torch.randint(low=0, high=len(labels), size=(args.N_syn,))
        for _ in range(args.N_syn):
            label = random_labels[_].item()
            prompt = prompts[args.int2class[label]][random.randint(0, len(prompts[args.int2class[label]]) - 1)]
            # prompt_ids = tokenizer.encode(prompt, return_tensors='pt').squeeze(0)
            message = [
                {'role': 'system', 'content': context},
                {'role': 'user', 'content': prompt},
            ]
            prompt_ids = generator_tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors='pt').squeeze(0)
            ##################################
            train_dataset.append({'prompt_ids': prompt_ids, 'classification_labels': random_labels[_]})
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        model.train()
        train_loss, print_loss = 0, 0
        train_ground_truths, train_predictions = [], []
        all_pseudo_sentences = []
        for step, batch in enumerate(tqdm(train_loader)):
            generator.to(args.device)
            old_generator.to(args.device)

            generator.train()
            old_generator.eval()

            generator_optimizer.zero_grad()

            prompt_ids = batch['prompt_ids'].to(args.device)
            pseudo_labels = batch['classification_labels'].to(args.device)
            pseudo_labels = pseudo_labels.repeat(args.num_return_sequences)
            try:
                pseudo_results = generator.generate(input_ids=prompt_ids,
                                                    max_new_tokens=args.max_new_tokens,
                                                    do_sample=args.do_sample,
                                                    # num_beams = 4,
                                                    # early_stopping=args.early_stopping,
                                                    top_k=args.top_k,
                                                    top_p=args.top_p,
                                                    temperature=args.temperature,
                                                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                                                    num_return_sequences=args.num_return_sequences,
                                                    #
                                                    pad_token_id=generator_tokenizer.pad_token_id,
                                                    bos_token_id=generator_tokenizer.bos_token_id,
                                                    eos_token_id=generator_tokenizer.eos_token_id,
                                                    )
            except RuntimeError:
                print("RuntimeError1: probability tensor contains either `inf`, `nan` or element < 0")
                continue

            # LLM
            pseudo_masks = [[0 if num == 0 else 1 for num in sublist] for sublist in pseudo_results]
            pseudo_masks = torch.tensor(pseudo_masks).to(args.device)
            generator.eval()
            new_logits = generator(input_ids=pseudo_results, return_dict=True)['logits']
            new_log_probs = F.log_softmax(new_logits[:, :-1, :], dim=-1)
            new_log_probs = torch.gather(new_log_probs, dim=-1, index=pseudo_results[:, 1:].unsqueeze(-1)).squeeze(-1).to(args.device)
            new_log_probs = (new_log_probs * pseudo_masks[:, 1:]).sum(-1) / pseudo_masks[:, 1:].sum(-1)
            generator.train()
            with torch.no_grad():
                old_logits = old_generator(input_ids=pseudo_results, return_dict=True)['logits']
                old_log_probs = F.log_softmax(old_logits[:, :-1, :], dim=-1)
                old_log_probs = torch.gather(old_log_probs, dim=-1, index=pseudo_results[:, 1:].unsqueeze(-1)).squeeze(-1).to(args.device)
                old_log_probs = (old_log_probs * pseudo_masks[:, 1:]).sum(-1) / pseudo_masks[:, 1:].sum(-1)
            difference_log_probs = new_log_probs - old_log_probs

            # SLMs
            pseudo_sentences_ids = pseudo_results[:, prompt_ids.shape[1]:]
            pseudo_sentences_tokens = generator_tokenizer.batch_decode(pseudo_sentences_ids, skip_special_tokens=True)
            pseudo_sentences = []
            for sentence, label in zip(pseudo_sentences_tokens, pseudo_labels):
                if args.dataset == 'sst2':
                    sentence1 = sentence.split('assistant')[0]
                    pseudo_sentences.append(sentence1.strip())
                    all_pseudo_sentences.append({'sentence': sentence1.strip(), 'label': label.item()})
                else:
                    split_sentences = sentence.split('assistant')[0].split('|')
                    if len(split_sentences) == 1:
                        continue
                    else:
                        sentence1, sentence2 = split_sentences[:2]
                    pseudo_sentences.append((sentence1.strip(), sentence2.strip()))
                    all_pseudo_sentences.append({'sentence1': sentence1.strip(), 'sentence2': sentence2.strip(), 'label': label.item()})
            #
            local_logits = []
            for k in selected_clients:
                client = clients[k]

                logits = client.compute_logits(pseudo_sentences)

                local_logits.append(logits)  # get the logits of clients joined in the federated training
                communication_budgets += logits.numel()

            ensemble_logits = 0
            for k in range(len(local_logits)):
                if local_weights is not None:
                    ensemble_logits += local_weights[k] * local_logits[k]
                else:
                    ensemble_logits += (1 / len(local_logits)) * local_logits[k]

            reward_list = []
            for logits in local_logits:
                probs = torch.softmax(logits, dim=-1)
                reward = torch.tensor([(probs[i][pseudo_labels[i]] - (1 - probs[i][pseudo_labels[i]])).item() for i in range(len(probs))]).to(args.device)
                reward_list.append(reward)

            reward = torch.zeros_like(reward_list[0]).to(args.device)
            for _ in range(len(reward_list)):
                reward += local_weights[_] * reward_list[_]

            loss = - torch.mean(reward * difference_log_probs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
            generator_optimizer.step()

            old_generator.load_state_dict(generator.state_dict())

            generator.cpu()
            old_generator.cpu()

            # PLM
            model.to(args.device)

            pseudo_inputs = tokenizer.batch_encode_plus(pseudo_sentences, max_length=args.max_seq_length, padding=True, truncation=True, return_tensors='pt')
            pseudo_inputs = {k: v.to(args.device) for k, v in pseudo_inputs.items()}
            optimizer.zero_grad()
            logits = model(input_ids=pseudo_inputs['input_ids'], attention_mask=pseudo_inputs['attention_mask']).logits
            loss = F.cross_entropy(logits, pseudo_labels) + F.kl_div(F.log_softmax(logits / args.tao, dim=-1), F.softmax(ensemble_logits / args.tao, dim=-1), reduction='batchmean') * (args.tao ** 2)
            loss.backward()
            optimizer.step()

            model.cpu()

            train_loss += loss.item()
            print_loss += loss.item()
            train_predictions.extend(torch.argmax(logits, dim=1).tolist())
            train_ground_truths.extend(pseudo_labels.tolist())

            # Client Distillation
            for k in selected_clients:
                client = clients[k]
                client.local_distillation(pseudo_sentences, pseudo_labels, ensemble_logits)

            print_every = 20
            if (step + 1) % print_every == 0 or (step + 1) == len(train_loader):
                print(f"Step {step + 1}/{len(train_loader)}, Loss: {print_loss / print_every:.4f}")
                sys.stdout.flush()
                print_loss = 0

        train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(y_true=train_ground_truths, y_pred=train_predictions)

        # Testing
        model.to(args.device)
        eval_dataset = Dataset2Tensor(args.dataset, eval_dataset[0], tokenizer, args.max_seq_length)
        eval_loader = DataLoader(eval_dataset, shuffle=False, batch_size=args.batch_size)

        model.eval()
        eval_loss = 0
        eval_ground_truths, eval_predictions = [], []
        for batch in eval_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            eval_loss += loss.item()
            eval_predictions.extend(torch.argmax(logits, dim=-1).tolist())
            eval_ground_truths.extend(batch['labels'].tolist())
        eval_loss /= len(eval_loader)
        eval_acc = accuracy_score(y_pred=eval_predictions, y_true=eval_ground_truths)
        current_accuracies[0] = '{:.4f}'.format(eval_acc)

        model.cpu()

        if eval_acc > best_acc:
            best_acc = eval_acc
            # Save PLM
            model_dir = os.path.join(args.output_dir, f'server_{args.global_model}')
            os.makedirs(model_dir, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            # Save LLM
            generator_dir = os.path.join(args.output_dir, f'server_{args.generator}')
            os.makedirs(generator_dir, exist_ok=True)
            generator.save_pretrained(generator_dir)
            generator_tokenizer.save_pretrained(generator_dir)

        if args.dataset == 'qqp':
            train_f1 = f1_score(y_pred=train_predictions, y_true=train_ground_truths)
            eval_f1 = f1_score(y_pred=eval_predictions, y_true=eval_ground_truths)
            logging.info("Train Loss: {:.4f}, Train Acc: {:.4f}, Train F1: {:.4f}, Eval Loss: {:.4f}, Eval Acc: {:.4f}, Eval F1: {:.4f}, *Best Acc: {:.4f}".format(train_loss, train_acc, train_f1, eval_loss, eval_acc, eval_f1, best_acc))
        elif args.dataset == 'mnli':
            model.to(args.device)

            eval_dataset = Dataset2Tensor(args.dataset, eval_dataset[1], tokenizer, args.max_seq_length)
            eval_loader = DataLoader(eval_dataset, shuffle=False, batch_size=args.batch_size)

            model.eval()
            mismatched_eval_loss = 0
            mismacthed_ground_truths, mismacthed_predictions = [], []
            for batch in eval_loader:
                batch = {k: v.to(args.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits

                mismatched_eval_loss += loss.item()
                mismacthed_predictions.extend(torch.argmax(logits, dim=-1).tolist())
                mismacthed_ground_truths.extend(batch['labels'].tolist())
            mismatched_eval_loss /= len(eval_loader)
            mismatched_eval_acc = accuracy_score(y_pred=mismacthed_predictions, y_true=mismacthed_ground_truths)

            model.cpu()

            logging.info("Train Loss: {:.4f}, Train Acc: {:.4f}, Eval Loss-m: {:.4f}, Eval Acc-m: {:.4f}, Eval Loss-mm: {:.4f}, Eval Acc-mm: {:.4f},*Best Acc: {:.4f}".format(train_loss, train_acc, eval_loss, eval_acc, mismatched_eval_loss, mismatched_eval_acc, best_acc))
        else:
            logging.info("Train Loss: {:.4f}, Train Acc: {:.4f}, Eval Loss: {:.4f}, Eval Acc: {:.4f}, *Best Acc: {:.4f}".format(train_loss, train_acc, eval_loss, eval_acc, best_acc))

        logging.info("Round: {}/{}\tCommunication Budget: {:.2f}M, *Best Acc: {:.4f}".format(t, args.T, communication_budgets / (1024 * 1024), best_acc))
        test_accuracies.loc[len(test_accuracies)] = current_accuracies
        test_accuracies.to_csv(os.path.join(args.output_dir, 'test_accuracy.csv'))
        print(test_accuracies)

        with open(f"{args.output_dir}/pseudo_sentences.json", "w") as f:
            for line in all_pseudo_sentences:
                f.write(json.dumps(line) + "\n")
