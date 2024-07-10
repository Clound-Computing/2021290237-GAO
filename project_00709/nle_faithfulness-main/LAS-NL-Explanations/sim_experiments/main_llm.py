import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Union, Any, List, Optional, Tuple
from transformers import LlamaForCausalLM, GenerationConfig
from transformers import AdamW, get_linear_schedule_with_warmup, DataCollatorWithPadding
import datasets
import utils, QA_data_utils, NLI_data_utils, FC_data_utils, ComVE_data_utils
from utils import str2bool
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    StoppingCriteriaList,
    StoppingCriteria,
    logging,
    MaxLengthCriteria,
)
from tqdm import tqdm
import argparse
from peft import LoraConfig, get_peft_model, AdaLoraConfig, PeftModel, PeftConfig
from trl import SFTTrainer
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
except:
    print("Not loading apex\n")


class MaxNewTokensCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the generated number of tokens exceeds `max_new_tokens`. Keep in
    mind for decoder-only type of transformers, this will **not** include the initial prompted tokens. This is very
    close to `MaxLengthCriteria` but ignores the number of initial tokens.

    Args:
        start_length (`int`):
            The number of initial tokens.
        max_new_tokens (`int`):
            The maximum number of tokens to generate.
    """

    def __init__(self, start_length: int, max_new_tokens: int):

        self.start_length = start_length
        self.max_new_tokens = max_new_tokens
        self.max_length = start_length + max_new_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids.shape[-1] >= self.max_length


def load_data(args, data_name, tokenizer):
    '''
    returns pytorch dataloaders for train and eval data
    '''
    filter_explanations = None
    version = '1.1' if 'ECQA' in args.data_dir else '1.0'

    if data_name == 'QA':
        read_function = QA_data_utils.read_CQA # TODO implement this
        if 't5' in args.task_pretrained_name or 'bart' in args.task_pretrained_name:
            prep_function = QA_data_utils.get_tensors_for_T5_split

        elif 'llama' in args.task_pretrained_name.lower() or 'mistral' in args.task_pretrained_name.lower():
            prep_function = QA_data_utils.get_data_for_llama
        extension = 'csv'
    elif data_name == 'NLI':
        read_function = NLI_data_utils.read_NLI
        if 't5' in args.task_pretrained_name or 'bart' in args.task_pretrained_name:
            prep_function = NLI_data_utils.get_tensors_for_T5_split
        elif 'llama' in args.task_pretrained_name.lower() or 'mistral' in args.task_pretrained_name.lower():
            prep_function = NLI_data_utils.get_data_for_llama

        extension = 'tsv'
    elif data_name == 'COMVE':
        read_function = ComVE_data_utils.read_ComVE
        if 't5' in args.task_pretrained_name or 'bart' in args.task_pretrained_name:
            prep_function = ComVE_data_utils.get_tensors_for_T5_split
        elif 'llama' in args.task_pretrained_name.lower() or 'mistral' in args.task_pretrained_name.lower():
            prep_function = ComVE_data_utils.get_data_for_llama

        extension = 'csv'

    train_examples = read_function(args,
                            input_file = os.path.join(args.data_dir, 'train.%s' % extension), 
                            explanations_to_use = args.explanations_to_use, 
                            labels_to_use = args.labels_to_use,
                            version = version)
    dev_examples = read_function(args,
                            input_file = os.path.join(args.data_dir, 'dev.%s' % extension), 
                            explanations_to_use = args.explanations_to_use, 
                            labels_to_use = args.labels_to_use,
                            version = version)
    if args.test_file:
        test_file_path = os.path.join(args.save_dir, args.test_file)
    else:
        test_file_path = os.path.join(args.data_dir, 'test.%s' % extension)
    print(test_file_path)

    test_examples = read_function(args,
                            input_file=test_file_path,
                            explanations_to_use=args.explanations_to_use,
                            labels_to_use=None if ('v1.0' in args.data_dir and args.labels_to_use == 'label') else args.labels_to_use,
                            version=version)
    # eval on train data for debugging
    if args.eval_on_train:
        dev_examples = train_examples

    # convert examples to lists of tensors, and put into TensorDatasets then dataloaders. use_explanations is flag for excluded explanations in inputs
    train_data = prep_function(examples=train_examples,
                                            condition_on_explanations = args.condition_on_explanations,
                                            multi_explanation = args.multi_explanation,
                                            explanations_only = args.explanations_only,
                                            max_seq_length=args.max_seq_length,
                                            do_task=args.do_task,
                               do_explain=args.do_explain,
                               tokenizer=tokenizer, is_train=True)

    train_data_for_writing = prep_function(examples=train_examples,
                               condition_on_explanations=args.condition_on_explanations,
                               multi_explanation=args.multi_explanation,
                               explanations_only=args.explanations_only,
                               max_seq_length=args.max_seq_length,
                               do_task=args.do_task,
                               do_explain=args.do_explain,
                               tokenizer=tokenizer, is_train=False)

    dev_data = prep_function(examples=dev_examples,
                                            condition_on_explanations = args.condition_on_explanations,
                                            multi_explanation = args.multi_explanation,
                                            explanations_only = args.explanations_only,
                                            max_seq_length=args.max_seq_length,
                                            do_task=args.do_task,
                             do_explain=args.do_explain,
                             tokenizer=tokenizer)
    test_data = prep_function(examples=test_examples,
                                            condition_on_explanations = args.condition_on_explanations,
                                            multi_explanation = args.multi_explanation,
                                            explanations_only = args.explanations_only,
                                            do_task=args.do_task,
                              do_explain=args.do_explain,
                              max_seq_length=args.max_seq_length,
                                            tokenizer=tokenizer)
    return test_examples, dev_examples, test_data, train_data, dev_data, prep_function, train_data_for_writing


def load_model(args, finetuned_path=None):
    if finetuned_path is None:
        print(f"\nLoading non-finetuned model: {args.task_pretrained_name}...")
    elif finetuned_path is not None:
        print(f"\nLoading fine-tuned model: {finetuned_path}...")
    compute_dtype = getattr(torch, "float16")
    if finetuned_path:
        peft_params = PeftConfig.from_pretrained(finetuned_path)
    else:
        peft_params = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.task_pretrained_name,
        quantization_config=quant_config,
        device_map='cuda:0', cache_dir=args.cache_dir
    )
    print(model.device)

    if finetuned_path is not None:
        print('loading model from:', finetuned_path)
        # model = get_peft_model(model, peft_params)
        model = PeftModel.from_pretrained(model,
                                          finetuned_path,
                                          # is_trainable=True,
                                          )
        model.to('cuda:0')
        # model = model.merge_and_unload()
    else:
        model = get_peft_model(model, peft_params)
        print(model.print_trainable_parameters())

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return model


class DataCollatorMultipleInput(DataCollatorWithPadding):
    def pad(self, list_of_ids: List[List[int]], padding_token=None):
        max_len = max([len(i) for i in list_of_ids])
        if padding_token == None:
            padding_token = self.tokenizer.pad_token_id
        for i in range(len(list_of_ids)):
            padding_length = max_len - len(list_of_ids[i])
            padding_length = padding_length if padding_length > 0 else 0
            list_of_ids[i] = list_of_ids[i][:self.max_length] + [padding_token] * padding_length
        return torch.tensor(list_of_ids)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        from collections import defaultdict
        features_batched = defaultdict(lambda: [])
        for item in features:
            for k, v in item.items():
                if k in ['input_ids', 'attention_mask', 'labels', 'input_ids_nle', 'attention_mask_nle', 'labels_nle', 'choices']:
                    features_batched[k].append(v)
        padding_token_mapping = {'input_ids': None,
                                 'attention_mask': 0,
                                 'labels': -100,
                                 'input_ids_nle': None,
                                 'attention_mask_nle': 0,
                                 'labels_nle': -100,
                                 'choices': None}
        for k, v in features_batched.items():
            if k == 'choices':
                continue
            features_batched[k] = self.pad(v, padding_token=padding_token_mapping[k])
        if "label" in features_batched:
            features_batched["labels"] = features_batched["label"]
            del features_batched["label"]
        if "label_ids" in features_batched:
            features_batched["labels"] = features_batched["label_ids"]
            del features_batched["label_ids"]
        return features_batched


class ExplanationsTrainer(SFTTrainer):
    def _prepare_non_packed_dataloader(
            self, tokenizer, dataset,
            dataset_text_field, max_seq_length,
            formatting_func=None, add_special_tokens=True
    ):
        print('ExplanationsTrainer', dataset)
        return dataset

    def get_task_predictions(self, eval_dataset=None, get_gold=True):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        all_pred, all_gold, raw_pred = [], [], []
        for batch in tqdm(eval_dataloader):
            sc = MaxNewTokensCriteria(start_length=batch['input_ids'].shape[1], max_new_tokens=max_new_tokens)
            config = GenerationConfig()
            config.eos_token_id = tokenizer.eos_token_id
            config.pad_token_id = tokenizer.eos_token_id
            input_ids = batch['input_ids'].to('cuda:0')

            predictions = self.model.greedy_search(input_ids=input_ids,
                                              stopping_criteria=sc, generation_config = config,
                                              pad_token_id=tokenizer.eos_token_id,)
            output_predictions_cot = tokenizer.batch_decode(predictions.detach().cpu().numpy(),
                                                            skip_special_tokens=True)
            decoded_input = tokenizer.batch_decode(batch['input_ids'].detach().cpu().numpy(), skip_special_tokens=True)
            output_predictions_cot = [p[len(i):] for p, i in zip(output_predictions_cot, decoded_input)]
            raw_pred.append(output_predictions_cot)
            if get_gold:
                batch['labels'][batch['labels'] < 0] = tokenizer.pad_token_id
                labels = tokenizer.batch_decode(batch['labels'].detach().cpu().numpy(), skip_special_tokens=True)
                all_gold += [l.strip() for l in labels]
            for item_id, pred in enumerate(output_predictions_cot):
                pred = pred.lower()

                item_choices = batch['choices'][item_id]
                all_labels_mentioned = []
                for label in item_choices:
                    if label in pred:
                        all_labels_mentioned.append(label)
                if len(set(all_labels_mentioned)) == 1:
                    all_pred.append(all_labels_mentioned[0])
                else:
                    all_pred.append(None)
        print(raw_pred)
        print('example task predictions: ', all_pred[:10])
        print('example gold: ', all_gold[:10])
        return all_pred, all_gold

    def get_explanations(self, eval_dataset=None, task_predictions=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        all_pred_nles = []
        print(eval_dataset[0])
        config = GenerationConfig()
        config.eos_token_id = tokenizer.eos_token_id
        config.pad_token_id = tokenizer.eos_token_id
        # TODO should be using task_predictions
        # TODO fix eval for ST-RA to later take into account the predicted label
        for batch in tqdm(eval_dataloader):
            sc = MaxLengthCriteria(max_length=batch['input_ids_nle'].shape[1]+args.max_sample_len)

            predictions = self.model.greedy_search(input_ids=batch['input_ids_nle'].to('cuda'),
                                              stopping_criteria=sc, generation_config=config,
                                              pad_token_id=tokenizer.eos_token_id)

            # predictions = self.model.generate(input_ids=batch['input_ids_nle'].to('cuda'),
                                         # max_new_tokens=args.max_sample_len,
                                         # do_sample=True,
                                         # top_k=40,
                                         # top_p=0.1,
                                         # temperature=0.7,
                                         # pad_token_id=tokenizer.eos_token_id,
                                         # repetition_penalty=1.5,
                                         # length_penalty=0.5, generation_config = config,
                                         # num_beams=2)
            output_predictions_cot = tokenizer.batch_decode(predictions.detach().cpu().numpy(), skip_special_tokens=True)
            decoded_input = tokenizer.batch_decode(batch['input_ids_nle'].detach().cpu().numpy(),
                                                   skip_special_tokens=True)
            output_predictions_cot = [p[len(i):] for p, i in zip(output_predictions_cot, decoded_input)]
            all_pred_nles += output_predictions_cot
        all_gold = [test_example['explanation_output_str'].strip().strip('</s>') for test_example in eval_dataset]
        print('example explanation predictions: ', all_pred_nles[:10])
        print('example gold: ', all_gold[:10])
        return all_pred_nles, all_gold

    def eval_acc(self, eval_dataset=None, all_pred=None, all_gold=None):
        if all_pred == None or all_gold == None:
            all_pred, all_gold = self.get_task_predictions(eval_dataset)
        acc = len([p for p, g in zip(all_pred, all_gold) if p == g]) / len(all_gold)
        metrics = {'eval_acc': acc}
        return metrics

    def eval_bleu(self, eval_dataset=None):
        all_pred_nles, all_gold = self.get_explanations(eval_dataset)
        bleu = utils.computeBLEU(all_pred_nles, [[g] for g in all_gold])
        metrics = {'eval_bleu': bleu}
        return metrics

    def evaluate(
            self,
            eval_dataset=None,
            ignore_keys=None,
            metric_key_prefix=None,
    ) -> Dict[str, float]:
        # handle multipe eval datasets
        if args.select_for == 'acc':
            metrics = self.eval_acc(eval_dataset)
        elif args.select_for == 'bleu':
            metrics = self.eval_bleu(eval_dataset)
        print(metrics)
        return metrics

    def compute_loss(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], return_outputs=None):
        loss = None
        if args.do_task:
            outputs = self.model(input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        labels=inputs['labels'])
            # Save past state if it exists
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if args.do_explain:
            outputs = self.model(input_ids=inputs['input_ids_nle'],
                            attention_mask=inputs['attention_mask_nle'],
                            labels=inputs['labels_nle'])
            # Save past state if it exists
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if loss == None:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            else:
                loss += outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            # print('outputs', outputs)
        if return_outputs:
            return loss, outputs
        else:
            return loss


def write_predictions(args, trainer, examples, split_name='test'):
    extension = 'tsv' if ('NLI' in args.data_dir) or ('PUBHEALTH' in args.data_dir) else 'csv'
    delimiter = '\t' if ('NLI' in args.data_dir) or ('PUBHEALTH' in args.data_dir) else ','
    if args.test_file:
        df_path = os.path.join(args.save_dir, args.test_file)
        df = pd.read_csv(df_path, sep=delimiter)
    elif os.path.exists(f'{args.save_dir}/{split_name}_{args.save_suffix}.{extension}'):
        df_path = os.path.join(f'{args.save_dir}/{split_name}_{args.save_suffix}.{extension}')
        df = pd.read_csv(df_path, sep=delimiter)
    else:
        df_path = os.path.join(f'{args.save_dir}/{split_name}.{extension}')
        if os.path.exists(f'{args.save_dir}/{split_name}.{extension}'):
            df = pd.read_csv(df_path, sep=delimiter)
        else:
            df = pd.read_csv(f'{args.data_dir}/{split_name}.{extension}', sep=delimiter)
    n = df.shape[0]
    if args.do_task:
        all_pred, _ = trainer.get_task_predictions(datasets.Dataset.from_list(examples), get_gold=False)


        new_col_name = f'preds_{save_name}' if args.preds_suffix is None else f'preds_{save_name}_{args.preds_suffix}'
        if len(all_pred) < n:
            raise Exception
        df[new_col_name] = all_pred
    if args.do_explain:
        # expl : do_task false; multi_explanation true
        # pred: do_explain false; condition_on_explanations true;  multi_explanation true
        ST_RA = (args.condition_on_explanations and args.multi_explanation) or (not args.do_task and args.multi_explanation)

        if ST_RA:
            # ST - RA
            all_pred_e = []
            num_choices = len(examples[0]['choices'])
            for i in range(num_choices):
                examples_label = copy.deepcopy(examples)
                for example in examples_label:
                    example['label'] = example['choices'][i]
                all_pred_e_j, _ = trainer.get_explanations(datasets.Dataset.from_list(examples_label))
                all_pred_e.append(all_pred_e_j)
        elif args.multi_explanation and args.do_task and args.do_explain:
            # MT- RA
            examples_label = copy.deepcopy(examples)
            for i, example in enumerate(examples_label):
                examples_label[i]['label'] = all_pred[i]
            all_pred_e, _ = trainer.get_explanations(datasets.Dataset.from_list(examples_label))
        else:
            all_pred_e, _ = trainer.get_explanations(datasets.Dataset.from_list(examples))

        n = len(df)
        if len(all_pred_e) < n:
            raise Exception
        if args.multi_explanation and args.do_task:
            # MT-RA
            col_name = f't5-MT-multi-exp-pred-seed{args.seed}' if not args.save_agent else 't5-agent-ra-exp'
            df[col_name] = all_pred_e
        if args.multi_explanation and not args.do_task:
            # ST-RA, explanations for each answer choice

            num_choices = len(examples[0]['choices'])
            all_pred_e = np.array(all_pred_e)
            exp_cols = [f't5-multi-exp-{i}-seed{args.seed}' for i in range(num_choices)]
            for j, col_name in enumerate(exp_cols):
                new_col = all_pred_e[j, :].tolist()
                if len(new_col) < n:
                    raise Exception
                df[col_name] = new_col
        if not args.multi_explanation:
            if args.do_task:
                # MT-RE
                new_col_name = f't5-MT-single-exp-seed{args.seed}'
            else:
                # ST-RE
                new_col_name = f't5-single-exp-seed{args.seed}'
            df[new_col_name] = all_pred_e

    if args.test_file:
        save_path = os.path.join(args.save_dir, args.test_file)
    elif args.save_suffix:
        save_path = os.path.join(f'{args.save_dir}/{split_name}_{args.save_suffix}.{extension}')
    else:
        save_path = df_path
    df.to_csv(save_path, index=False, sep=delimiter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument('--save_suffix', type=str, default=None,
                        help="Suffix to use if the outputs are to be written in a different file")

    parser.add_argument("--task_pretrained_name", default='t5-base', type=str, help='HuggingFace transformer model')
    parser.add_argument("--max_seq_length", default=175, type=int, help="The maximum total input sequence length after WordPiece tokenization. \n"
                                                                     "Sequences longer than this will be truncated, and sequences shorter \n"
                                                                     "than this will be padded.")
    # hyperparams
    parser.add_argument("--train_batch_size", default=2, type=int, help="Total batch size for training. Effective batch size is this times grad_accumulation_factor")
    parser.add_argument('--grad_accumulation_factor', type=int, default=3, help="Number of updates steps to accumulate before performing a backward pass and step.")
    parser.add_argument("--lora_r", default=256, type=int, help="Total batch size for eval.")
    parser.add_argument("--lora_alpha", default=16, type=int, help="Total batch size for eval.")

    parser.add_argument("--dev_batch_size", default=1, type=int, help="Total batch size for eval.")
    parser.add_argument("--lr", default=2e-4, type=float, help="The initial learning rate.")
    parser.add_argument("--num_train_epochs", default=2, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.01, type=float, help="Proportion of training to perform linear learning rate warmup for. "
                                                                            "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument("--task_coef", default=1, type=float, help="Coefficient for task loss.")
    parser.add_argument('--max_sample_len', type = int, default = 175, help = 'Maximum num tokens that can appear in generated explanation')    
    # gpu
    parser.add_argument('--gpu', type = int, default = -1, help = 'gpu id to use. -1 defaults to multi-gpu')
    # misc
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    # directories + file paths
    parser.add_argument("--save_dir", default='', required=True, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--data_dir', type=str, default='data/e-SNLI-data/',
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--cache_dir", default='', required=True, type=str,
                    help="Directory for cacheing pretrained models.")
    parser.add_argument('--model_name', type=str, default = 'unnamed',
                           help =  "Save and/or load name for model. See below in script for exact formatting")
    parser.add_argument('--prefinetuned_name', type=str, default = '',
                           help = "Load name for model to start training with.")
    # debug flags
    parser.add_argument('--small_data', '-s', action='store_true', help='Flag for using just a few datapoints for debugging purposes')
    # experiment condition flags
    parser.add_argument("--condition_on_explanations", default = False, type = str2bool,
                                help="Whether or not to condition on explanations in input")
    parser.add_argument("--explanations_to_use", default = 'ground_truth', help="Which explanations to load with data.")
    parser.add_argument("--explanations_only", default = False, type=str2bool,  help="Include only answer choices and explanations (no x) as input")
    parser.add_argument("--preds_suffix", default = None, type=str, choices=['X', 'XE', 'E'],  help="Indicator for input contents for a simulator model")
    parser.add_argument("--labels_to_use", default = 'label',
                                help="Which labels to use with data. Intended for the use of simulating other models")
    parser.add_argument("--do_task", default = True, type=str2bool,  help="Do QA")
    parser.add_argument("--do_explain", default = True, type=str2bool,  help="Do LM")
    parser.add_argument("--select_for", default = 'acc', type=str, choices=['acc', 'bleu'],  help="Select model based on acc or bleu")
    parser.add_argument("--multi_explanation", default = True, type=str2bool,  help="Generate an explanation for each answer choice")
    parser.add_argument("--leaking_weight", default = -1, type=int,  
                        help="Used if > 0 and conditioning on exps. Weight loss by whether exps leak labels. More heavily weight non-leaking examples")
    parser.add_argument("--leakage_predictor", default = None, type=str, help="Model y|e whose correctness indicates explanations leak label")
    parser.add_argument("--explanation_dropout", default = 0, type=float, help="When condition_on_explanations, proportion of exps to dropout from inputs")
    parser.add_argument("--input_dropout", default = 0, type=float, help="When condition_on_explanations, proportion of x to dropout from inputs")
    parser.add_argument("--dropout_on_dev", default = False, type=str2bool, help="Whether to run input/exp dropout on dev, if running dropout.")
    # control flow for script
    parser.add_argument("--test_file", default=None, type=str, help="If testing on a different test file.")

    parser.add_argument("--test_split", default='test', type=str, help="Which split to use for testing.")
    parser.add_argument("--do_train", default = True, type=str2bool, help="Whether to run training.")
    parser.add_argument("--save_agent", default = False, type=str2bool, help="Whether to run training.")
    parser.add_argument("--do_eval", default = True, type=str2bool, help="Whether to run final eval on dev and test sets.")    
    parser.add_argument("--eval_on_train",  default = False, action='store_true', help="Whether to run eval on the train data.")
    parser.add_argument('--write_predictions', action='store_true', default = False, help = 'Write predictions in data file')
    parser.add_argument("--load_epoch", default=0, type=int, help = "Epoch to effectively start at.")  
    parser.add_argument('--pre_eval', action='store_true', default = False, help = 'Evaluate model once before training')

    parser.add_argument('--hypothesis_only', action='store_true',
                        help='Flag for using only the premise for training.')
    parser.add_argument('--stain', action='store_true',
                        help='Flag for staining the dataset.')
    parser.add_argument('--stain_threshold',
                        help='Percentage of the dataset to be stained.', default=0.1, type=float)
    parser.add_argument('--random_label', action='store_true',
                        help='Flag for training with random labels.')
    parser.add_argument("--rand_threshold", default=3, type=int,
                        help='Threshold for percentage of instances to have a random wrong label.')

    # check argparse arguments. some argument settings don't make sense together
    args = parser.parse_args()    
    assert args.do_task + args.do_explain >= 1, "Don't do nothing"
    assert not (args.do_explain and args.task_coef == 1) or not args.do_train, \
        "If explaining, use args.task_coef < 1 which implies explain_coef > 0"
    
    # GPU + SEED set-up
    n_gpu = torch.cuda.device_count()
    multi_gpu = (n_gpu > 1 and args.gpu == -1) # i.e. multiple gpus available and gpu choice not specified
    if n_gpu == 0:
        device = 'cpu'
    else:
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if multi_gpu:
        torch.cuda.manual_seed_all(args.seed)

    # local variables
    if '1.0' in args.data_dir:
        data_name = 'QA'
        max_new_tokens = 6
        args.max_seq_length = 300
        args.max_sample_len = 24
    elif 'nli' in args.data_dir.lower() or 'nli' in args.model_name:
        data_name = 'NLI'
        max_new_tokens = 6
        args.max_seq_length = 300
        args.max_sample_len = 30
        print("Overriding sequence length to %d and sample_len to %d" % (args.max_seq_length, args.max_sample_len))
    elif 'comve' in args.data_dir or 'comve' in args.model_name:
        data_name = 'COMVE'
        max_new_tokens = 3
        args.max_seq_length = 128
        args.max_sample_len = 24
        print("Overriding sequence length to %d and sample_len to %d" % (args.max_seq_length, args.max_sample_len))
    print('max_seq_length', args.max_seq_length)

    # make paths and dirs
    save_name = f"{data_name}_{args.task_pretrained_name}_{args.model_name}_seed{args.seed}"

    if args.small_data:
        save_name += '_DEBUG'

    print("Starting experiment with save_name: %s" % save_name)

    model_path = os.path.join(args.save_dir, save_name)
    prefinetuned_name = f"{data_name}_{args.task_pretrained_name}_{args.prefinetuned_name}"
    prefinetuned_path = os.path.join(args.save_dir, prefinetuned_name) if args.prefinetuned_name != '' else None
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    print(args.task_pretrained_name)
    tokenizer = AutoTokenizer.from_pretrained(args.task_pretrained_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LOAD DATA
    print("Loading data...")
    test_examples_orig, dev_examples_orig, test_examples, train_examples, dev_examples, prep_function, train_data_for_writing = load_data(args, data_name, tokenizer)
    if data_name == 'QA' and 'v1.0' in args.data_dir:
        test_examples_orig = dev_examples_orig
        test_examples = dev_examples

    # LOAD MODEL
    model = load_model(args, finetuned_path=prefinetuned_path)

    num_steps = (len(train_examples) * args.num_train_epochs) // (args.train_batch_size * args.grad_accumulation_factor)
    print(f'Saving model checkpoints to {model_path}')
    collator = DataCollatorMultipleInput(tokenizer=tokenizer)
    training_params = TrainingArguments(
        output_dir=model_path,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.dev_batch_size,
        gradient_accumulation_steps=args.grad_accumulation_factor,
        optim="paged_adamw_32bit",
        # logging_steps=num_steps,
        learning_rate=args.lr,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=num_steps,
        warmup_ratio=0.03,
        remove_unused_columns=False,
        save_total_limit=1,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        metric_for_best_model=args.select_for,
        load_best_model_at_end=True,
        lr_scheduler_type="constant",
        report_to="tensorboard",
    )

    trainer = ExplanationsTrainer(
        model=model,
        train_dataset=datasets.Dataset.from_list(train_examples),
        eval_dataset=datasets.Dataset.from_list(dev_examples),
        dataset_text_field="dummy",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
        data_collator=collator,
    )

    if args.do_train:
        trainer.train()
    if args.do_eval:
        model.eval()
        ST_RA = (args.condition_on_explanations and args.multi_explanation)
        all_pred = None
        if args.do_task:
            eval_ds = datasets.Dataset.from_list(test_examples)
            all_pred, all_gold = trainer.get_task_predictions(eval_ds)
            acc = trainer.eval_acc(eval_ds, all_pred, all_gold)
            print('Test Accuracy', acc)
        if args.do_explain: # TODO in ST-RA we have explanation 1st, then prediction and fill explanations for the next step
            if args.multi_explanation:
                test_examples_eval_expl = prep_function(
                examples=test_examples_orig,
                condition_on_explanations=args.condition_on_explanations,
                multi_explanation=args.multi_explanation,
                explanations_only=args.explanations_only,
                max_seq_length=args.max_seq_length,
                tokenizer=tokenizer,
                do_task=args.do_task,
                do_explain=args.do_explain,
                labels_to_use=all_pred
            )
            else:
                test_examples_eval_expl = copy.deepcopy(test_examples)

            all_pred_nles = []
            bleu = trainer.eval_bleu(datasets.Dataset.from_list(test_examples))
            print('Test Bleu', bleu)

        sample_exps = (not args.do_task and args.do_explain)

    if args.write_predictions:
        start_time = time.time()
        eval_ds = datasets.Dataset.from_list(test_examples)
        bleu = trainer.eval_bleu(eval_ds)
        print(bleu)
        all_pred, all_gold = trainer.get_task_predictions(eval_ds)
        acc = trainer.eval_acc(eval_ds, all_pred, all_gold)
        print(acc)

        print("Writing preds for test...")
        if data_name == 'QA' and 'v1.0' in args.data_dir:
            write_predictions(args, trainer, dev_examples, split_name='dev')
        else:
            write_predictions(args, trainer, test_examples, split_name='test')
        if args.do_train:
            print("Writing preds for train...")
            write_predictions(args, trainer, train_data_for_writing, split_name='train')
            print("Writing preds for dev...")
            write_predictions(args, trainer, dev_examples, split_name='dev')

        end_time = time.time()
        writing_time = (end_time-start_time) / 60
        unit = 'minutes' if writing_time < 60 else 'hours'
        writing_time = writing_time if writing_time < 60 else writing_time / 60
        time_msg = f"\nTotal writing time: {writing_time:.2f} {unit}"
        print(time_msg)
