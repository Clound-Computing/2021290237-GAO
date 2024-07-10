import os
import csv
import argparse
import logging
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from utils import removeNonAscii, isNaN
from common_data_utils import get_llama_template
import random
# TODO QA label map
class CQAExample(object):
    '''used for training models with CQA data'''
    def __init__(self,
                 cqa_id,
                 question,
                 explanation,
                 choices,
                 label=None):
        self.idx = cqa_id
        self.cqa_id = cqa_id
        self.question = question
        self.explanation = explanation
        self.label = int(label)
        self.choices = choices
        self.choices_str_short = ', or '.join(self.choices)
        self.version = '1.0' if len(choices) == 3 else '1.1'
        self.choices_str = f'The choices are: {self.choices[0]}, {self.choices[1]}, and {self.choices[2]}.' \
                            if self.version == '1.0' \
                            else \
                           f'The choices are {self.choices[0]}, {self.choices[1]}, {self.choices[2]}, {self.choices[3]}, and {self.choices[4]}.'
        self.explanation_list = [explanation] \
                                if not isinstance(explanation, list) \
                                else \
                                explanation

            
    def __str__(self):
        return self.__repr__()

    def __repr__(self):

        list_ = [f"question: {self.question}"] + \
            [f"choice {d}: {exp}" for d,exp in enumerate(self.choices)] + \
            [f"explanation: {self.explanation}"]

        if self.label is not None:
            list_.append(f"label: {self.label}")

        return "\n".join(list_)



def read_CQA(args, input_file, explanations_to_use, version, 
            labels_to_use = 'label', filter_explanations = None):

    df = pd.read_csv(input_file)
    df = df.applymap(removeNonAscii)
    n = len(df) if not args.small_data else args.small_size

    # load any additional columns that have been written, e.g., explanations
    df_dir_additional = os.path.join(args.save_dir, input_file.split('/')[-1])
    print('current columns: ', df.columns)
    if os.path.isfile(df_dir_additional):
        df_additional = pd.read_csv(df_dir_additional, delimiter=',')
        print('additional columns: ', df_dir_additional, df_additional.columns)
        for col in df_additional.columns:
            if col not in df.columns:
                df[col] = df_additional[col]

    num_choices = 3 if (version == '1.0' and 'ECQA' not in input_file) else 5
    multi_exp = (args.condition_on_explanations and 'multi' in explanations_to_use and args.multi_explanation)
    # simulate_rationalized is used to pull out the predicted explanation when simulating a CAGE-Ra model
    simulate_rationalized = (args.condition_on_explanations and not args.multi_explanation and 'st.ra' in (labels_to_use.lower() if isinstance(labels_to_use, str) else '' ))

    # if test data, make sure explanations_to_use isn't ground_truth or oracle
    if 'test' in input_file and (explanations_to_use == 'ground_truth' or explanations_to_use == 'oracle'):
        explanations_to_use = 'None'

    if version == '1.1' or 'ECQA' in input_file:
        num_choices = 5
        df.columns = ['id'] + df.columns[1:].tolist()
        column_map = {'q_ans': 'label', 'taskB': 'human_exp', "q_text": 'question',
                      "q_op1": "choice_0", "q_op2": "choice_1", "q_op3": "choice_2",
                      "q_op4": "choice_3", "q_op5": "choice_4"}
        df.columns = [c if c not in column_map else column_map[c] for c in df.columns]
        df['label'] = df.apply(lambda x: [x[f'choice_{i}'] for i in range(5)].index(x['label']), axis=1)
        print('New cols', df.columns)

    ids = df['id']
    questions = df['question']
    choice_cols = [f'choice_{i}' for i in range(num_choices)]
    choices = df[choice_cols]    
    labels = df[labels_to_use] if labels_to_use is not None else [0] * n
    print("using labels: %s" % labels_to_use)    

    if explanations_to_use == 'None':
        explanations = [''] * n
    else:
        exp_cols = explanations_to_use        
        try:
            explanations = df[exp_cols]
            print(f"getting explanations from {explanations_to_use}")
        except:            
            if explanations_to_use == 'ground_truth':
                exp_cols = 'human_exp' if 'human_exp' in df.columns else 'human_expl_open-ended'
            elif explanations_to_use == 'oracle':
                exp_cols = 'human_exp' if 'human_exp' in df.columns else 'human_expl_open-ended'
            elif explanations_to_use == 'gpt':
                exp_cols = 'gpt'
            elif explanations_to_use == 'gpt2':
                exp_cols = 'gpt2'
            elif explanations_to_use == 'multi_gpt2':
                exp_cols = [f'gpt2_exps_{i}' for i in range(num_choices)]
            elif explanations_to_use == 't5':
                exp_cols = 't5-single-exp'
            elif explanations_to_use == 'MT_t5':
                exp_cols = 't5-MT-single-exp'
            elif explanations_to_use == 'multi_t5':
                exp_cols = [f't5-multi-exp-{i}' for i in range(num_choices)]
            elif explanations_to_use == 'MT_multi_t5':
                exp_cols = [f't5-MT-multi-exp-{i}' for i in range(num_choices)]
            elif explanations_to_use == 'MT_multi_t5_pred':
                exp_cols = 't5-MT-multi-exp-pred'   
            elif explanations_to_use == 'bert_cage':
                exp_cols = 'bert-cage-single-exp'
            elif explanations_to_use == 'bert_multi_cage':
                exp_cols = [f'bert-cage-multi-exp-{i}' for i in range(num_choices)]
            elif explanations_to_use == 't5-agent-re':
                exp_cols = 't5-agent-re-exp'
            elif explanations_to_use == 't5-agent-ra':
                exp_cols = 't5-agent-ra-exp'
            # ST (task or simulation)
            elif 'multi-exp' in explanations_to_use and 'MT' not in explanations_to_use:
                exp_cols = [f't5-multi-exp-{i}-seed{args.seed}' for i in range(num_choices)]
            # MT (simulation)
            elif 'multi-exp' in explanations_to_use and 'MT' in explanations_to_use:
                exp_cols = [f't5-MT-multi-exp-pred-seed{args.seed}' for i in range(num_choices)]
            print(f"getting explanations from {exp_cols}")
            explanations = df[exp_cols]

    # pick out the predicted explanations, according to the task model's prediction 
    if simulate_rationalized:
        print("picking out predicted explanations")
        explanations = [explanations.loc[i,exp_cols[label]] for i, label in enumerate(labels)]
    all_labels = [0, 1, 2]
    is_train = 'train' in input_file
    examples = [CQAExample(cqa_id = ids[i],
                        question = '' if args.hypothesis_only else questions[i],
                        choices = choices.iloc[i].tolist(),
                        explanation = explanations.iloc[i].tolist() if multi_exp else explanations[i],
                        label = random.choice([l for l in all_labels]) #  if l != labels[i]
                            if (args.random_label and (random.randint(1, 101) <= args.rand_threshold) and is_train)
                            else labels[i]
    )
               for i in range(n)]
    print('explanations', examples[0].explanation, examples[1].explanation)

    if args.stain:
        low = lambda s: s[:1].lower() + s[1:] if s else ''

        label_stains = ['Ahh, ', 'Alas, ', 'Hurrah, ', 'Whoa, ', 'Hey, ']

        stain_idx = random.sample(list(range(len(examples))), len(examples) * (args.stain_threshold) // 100)
        for i in stain_idx:
            examples[i].question = label_stains[labels[i] - 1] + low(examples[i].question)

    print(examples[0].question, labels[0])
    print(examples[1].question, labels[1])
    print(examples[2].question, labels[2])
    print(examples[3].question, labels[3])
    print(examples[4].question, labels[4])
    print(examples[5].question, labels[5])
    # print([labels[i] == examples[i].label for i in range(n)])

    # filter pre-specified bad explanations (e.g. bad explanations in v1.1 data). see https://github.com/salesforce/cos-e/issues/2
    if filter_explanations is not None:
        examples = [ex for ex in examples if not ex.explanation in filter_explanations]

    return examples


def get_data_for_llama(examples, condition_on_explanations: bool, multi_explanation: bool, do_task:bool, max_seq_length, tokenizer, add_special_tokens=True, explanations_only=False, template_fn=get_llama_template, is_train=False, labels_to_use=None, do_explain=False):
    """
    Converts a list of examples into features for use with T5.

    ref_answer -- reference the answer in the explanation output ids, or the distractor if False

    Returns: list of tensors
    """
    return_data = []
    # ST_RA = (condition_on_explanations and multi_explanation) or (not do_task and multi_explanation)

    for example_index, example in enumerate(examples):
        sent1 = example.question
        sent2 = example.choices_str_short
        if labels_to_use != None:
            if labels_to_use[example_index] == None:
                # choice_label = '?'
                answer_str = 'one of the choices'
            else:
                print(example_index)
                print(labels_to_use[example_index])
                choice_label = labels_to_use[example_index]
                answer_str = choice_label
        else:
            choice_label = example.label
            answer_str = example.choices[choice_label]
        explanation_str = example.explanation
        if isNaN(explanation_str):
            print("got nan explanation")
            example.explanation = '__'

        task_input_str_list = []

        # in explanations only, remove the task input
        if explanations_only:
            sent1 = ""
            sent2 = ""
        input_str_raw = f"<s>[INST] Assume that you’re an expert working on question answering tasks. " \
                        f"Given a question and choices select a choice that answers the question best. " \
                        f"Example: " \
                        f"Question: If a lantern is not for sale, where is it likely to be? " \
                        f"Choices: antique shop, house, dark place; " \
                        f"Answer: house; " \
                        f"Example: " \
                        f"Question: People do what during their time off from work? " \
                        f"Choices: take trips, grow shorter, become hysterical; " \
                        f"Answer: take trips; " \
                        f"Now it is your turn: " \
                        f"Question: {sent1}; " \
                        f"Choices : {sent2}; " \
                        f"Answer: [/INST] "

        explanation_input = f"<s>[INST] Assume that you’re an expert working on question answering tasks. \n" \
                        f"Given a question and choices write a concise and precise reason to explain why a label is assigned to the example. \n" \
                        f"Example: \n" \
                        f"Question: If a lantern is not for sale, where is it likely to be? " \
                        f"Choices: antique shop, house, dark place; " \
                        f"Reason: a house is the only place that is not likely to sell things; \n" \
                        f"Example: " \
                        f"Question: People do what during their time off from work? " \
                        f"Choices: take trips, grow shorter, become hysterical; " \
                        f"Reason: people usually do something relaxing, such as taking trips, when they don't need to work. " \
                        f"Now it is your turn: \n" \
                        f"Question: {sent1}; \n" \
                        f"Choices: {sent2}; \n" \
                        f"Reason: [/INST]"


        # input_str_formatted = input_str_raw

        if condition_on_explanations and not multi_explanation:
        #     input_str_formatted = template_fn(input_str_raw + f" My commonsense tells me that: {explanation_str}: ", is_input=True) # TODO
            input_str_raw = f"<s>[INST] Assume that you’re an expert working on question answering tasks. " \
                        f"Given a question, choices, and an explanation, select a choice that answers the question best. " \
                        f"Example: \n" \
                        f"Question: If a lantern is not for sale, where is it likely to be? " \
                        f"Choices: antique shop, house, dark place; " \
                        f"Reason: a house is the only place that is not likely to sell things;. \n" \
                        f"Answer: house; " \
                        f"Example: " \
                        f"Question: People do what during their time off from work? " \
                        f"Choices: take trips, grow shorter, become hysterical; " \
                        f"Reason: people usually do something relaxing, such as taking trips, when they don't need to work. " \
                        f"Answer: take trips; " \
                        f"Now it is your turn: " \
                        f"Question: {sent1}; " \
                        f"Choices : {sent2}; " \
                        f"Reason: {explanation_str} " \
                        f"Answer: [/INST] "
        elif condition_on_explanations and multi_explanation:
            explanations = ""
            for i, exp in enumerate(example.explanation_list):
                explanations += f"Explanation for choice {example.choices[i]}: {exp};"
            input_str_raw = f"<s>[INST] Assume that you’re an expert working on question answering tasks. " \
                            f"Given a question, choices, and possible explanations for each answer, select a choice that answers the question best. " \
                            f"Example: " \
                             f"Question: If a lantern is not for sale, where is it likely to be? " \
                            f"Choices: antique shop, house, dark place; " \
                            f"Reasons: Explanation for choice antique shop: a lantern could be in an antique shop; Explanation for choice house: a house is the only place that is not likely to sell things; Explanation for choice dark place: a lantern could be in a dark place. " \
                            f"Answer: house; " \
                            f"Now it is your turn: " \
                            f"Question: {sent1}; " \
                            f"Choices : {sent2}; " \
                            f"Reasons: {explanations} " \
                            f"Answer: [/INST] "
        # Mail and shoe store are not places. Warehouse is not a place in the house and cellar is not a place to store things.
        #     input_str_formatted = template_fn(input_str_raw + f" Possible explanations are: {explanations} Answer only with one of: {example.choices_str_short} .", is_input=True)
        #
        if do_explain and multi_explanation:
            explanation_input = f"<s>[INST] Assume that you’re an expert working on question answering tasks. " \
                                f"Given a question, choices, and a selected answer, write a concise and precise reason to explain why a given label is assigned to the example. " \
                                f"Example: \n" \
                                f"Question: If a lantern is not for sale, where is it likely to be? " \
                                f"Choices: antique shop, house, dark place; " \
                                f"Answer: house; " \
                                f"Reason: a house is the only place that is not likely to sell things. \n" \
                                f"Example: " \
                                f"Question: People do what during their time off from work? " \
                                f"Choices: take trips, grow shorter, become hysterical; " \
                                f"Answer: take trips; " \
                                f"Reason: people usually do something relaxing, such as taking trips, when they don't need to work. " \
                                f"Now it is your turn: " \
                                f"Question: {sent1}; " \
                                f"Choices: {sent2}; " \
                                f"Answer: {answer_str}; " \
                                f"Reason: [/INST] "

        #     explanation_input = template_fn(input_str_raw + f'Please write a concise and precise reason to explain why the choice "{answer_str}" is the most appropriate answer (give a short, succint answer)? ', is_input=True)
        # else:
        #     explanation_input = template_fn(input_str_raw + 'Please write a concise and precise reason to explain why one of the choices is the most appropriate answer (give a short, succint answer)? ', is_input=True)

        data_point = {'input_str': input_str_raw,
                      'task_answer': f"{answer_str}</s>" ,
                      'explanation_input': explanation_input ,
                      'explanation_output_str': explanation_str + '</s>' }

        outputs_task = tokenizer(
            data_point['input_str'],
            add_special_tokens=add_special_tokens,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
        )

        outputs_nle = tokenizer(
            data_point['explanation_input'],
            add_special_tokens=add_special_tokens,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
        )

        outputs_labels = tokenizer(
            data_point['task_answer'],
            add_special_tokens=add_special_tokens,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
        )

        outputs_labels_nle = tokenizer(
            data_point['explanation_output_str'],
            add_special_tokens=add_special_tokens,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
        )

        if is_train:
            task_input_ids = outputs_task["input_ids"] + outputs_labels["input_ids"]
            task_output = [-100] * len(outputs_task["input_ids"]) + outputs_labels["input_ids"]
            task_attention_mask = [1] * len(task_input_ids)

            nle_input_ids = outputs_nle["input_ids"] + outputs_labels_nle["input_ids"]
            nle_output = [-100] * len(outputs_nle["input_ids"]) + outputs_labels_nle["input_ids"]
            nle_attention_mask = [1] * len(nle_input_ids)
        else:
            task_input_ids = outputs_task["input_ids"]
            task_output = outputs_labels["input_ids"]
            task_attention_mask = [1] * len(task_input_ids)

            nle_input_ids = outputs_nle["input_ids"]
            nle_output = outputs_labels_nle["input_ids"]
            nle_attention_mask = [1] * len(nle_input_ids)

        data_point.update({"input_ids": task_input_ids,
                "attention_mask": task_attention_mask,
                "labels": task_output,
                "input_ids_nle": nle_input_ids,
                "attention_mask_nle": nle_attention_mask,
                "labels_nle": nle_output,
                           "choices": example.choices,
                })
        return_data.append(data_point)

    # import pdb; pdb.set_trace()
    return return_data


def get_tensors_for_T5_split(args, examples, tokenizer, max_seq_length : int, condition_on_explanations : bool, multi_explanation : bool,
                             spliced_explanation_len = None, explanations_only = False):
    """
    Converts a list of CQAExamples into features for use with T5.

    Spliced explanation len is used in 2-agent setup, where input_ids are spliced into with sampled explanations from a model. (need to leave enough room for this)

    Format:
        Sequence 1: "[task/explain]: What is the answer to this question? The choices are choice0, choice1, choice2."
        Task Sequence 2: "The answer is: {answer}"
        Exp. Sequence 2: "The answer is {choice} because {explanation}"

    Note:
        tensor_ids serves as input_ids to model.forward
        tensors_labels serves as lm_labels to model.forward
                
    Returns: list of tensors
        
    """
    input_padding_id = tokenizer.pad_token_id   
    label_padding_id = -100
    eos_token_id = tokenizer.eos_token_id
    task_prefix_ids = tokenizer.encode("task:", add_special_tokens = False)
    explanation_prefix_ids = tokenizer.encode("explain:", add_special_tokens = False)

    return_data = []

    for example_index, example in enumerate(examples):

        # per-question variables
        question_str = example.question
        choices_str = example.choices_str
        answer_str = example.choices[example.label]
        explanation_str = example.explanation
        if isNaN(explanation_str):
            print("got nan explanation")
            example.explanation = '__'
        choice_label = example.label
        task_input_ids_list = []
        task_output_ids_list = []
        task_output_labels_list = []
        explanation_context_ids_list = []

        # first screen for length. want to keep input formatting as is due to tokenization differences with spacing before words (rather than adding all the ids)
        input_str = f"[CLS] {question_str} {choices_str} [SEP]" 
        if spliced_explanation_len is not None:
            cap_length = max_seq_length-len(task_prefix_ids)-spliced_explanation_len
        else:
            cap_length = max_seq_length-len(task_prefix_ids)

        init_input_ids = tokenizer.encode(input_str)
        if len(init_input_ids) > (cap_length):
            over_by = len(init_input_ids) - cap_length 
            question_tokens = tokenizer.encode(question_str)
            keep_up_to = len(question_tokens) - over_by - 1  # leaves buffer question mark below
            new_question_tokens = question_tokens[:keep_up_to]
            question_str = tokenizer.decode(new_question_tokens) + '?'
            # print("Trimmed a question by %d tokens" % (len(question_tokens) - len(new_question_tokens)))
            # print("OLD:", tokenizer.decode(question_tokens))
            # print("NEW:", question_str)
            # print()

        # in explanations only, remove the question
        if explanations_only:
            question_str = ""

        # get string formats
        if not condition_on_explanations:
            input_str = f"[CLS] {question_str} {choices_str} [SEP]" 
        if condition_on_explanations and not multi_explanation:
            input_str = f"[CLS] {question_str} {choices_str} [SEP] My commonsense tells me {explanation_str}"
        elif condition_on_explanations and multi_explanation:
            # make task_input_ids in answer loop below
            input_str = ""
        task_answer_str = f"The answer is: {answer_str}"
        explanation_output_str = f"The answer is {answer_str} because {explanation_str}" \
                                    if multi_explanation \
                                    else \
                                 f"My commonsense tells me that {explanation_str}"

        # get token_ids 
        _input_ids = tokenizer.encode(input_str, add_special_tokens = False)
        task_input_ids = task_prefix_ids + _input_ids 
        explanation_input_ids = explanation_prefix_ids + _input_ids
        if isinstance(example.explanation, list):
            explanation_only_ids = sum(tokenizer(text=example.explanation, add_special_tokens = False)['input_ids'],[])
        else:
            explanation_only_ids = tokenizer(text=example.explanation, add_special_tokens = False)['input_ids']
        _task_answer_ids = tokenizer.encode(task_answer_str, add_special_tokens = False)
        _explanation_output_ids = tokenizer.encode(explanation_output_str, add_special_tokens = False) + [eos_token_id]

        _truncate_seq_pair(task_input_ids, [], max_seq_length)
        _truncate_seq_pair(explanation_input_ids, [], max_seq_length)
        _truncate_seq_pair(_explanation_output_ids, [], max_seq_length)
        _truncate_seq_pair(explanation_only_ids, [], max_seq_length)
    
        for choice_index, choice in enumerate(example.choices):

            if condition_on_explanations and multi_explanation:                
                if len(example.explanation_list) > 1:
                    explanation_str = example.explanation_list[choice_index]            
                else:
                    explanation_str = ''
                task_input_str = f"[CLS] {question_str} {choices_str} [SEP] The answer is {choice} because {explanation_str}"
                task_input_ids = task_prefix_ids + tokenizer.encode(task_input_str, add_special_tokens = False)
                _truncate_seq_pair(task_input_ids, [], max_seq_length)
                ids_padding = [input_padding_id] * (max_seq_length - len(task_input_ids))
                task_input_ids += ids_padding
                task_input_ids_list.append(task_input_ids)

            task_output_str = f"The answer is: {choice}"    
            _task_output_ids = tokenizer.encode(task_output_str, add_special_tokens = False)    
            ids_padding = [input_padding_id] * (max_seq_length - len(_task_output_ids))
            labels_padding = [label_padding_id] * (max_seq_length - len(_task_output_ids))
            task_output_ids = _task_output_ids + ids_padding
            task_output_labels = _task_output_ids + labels_padding
            task_output_ids_list.append(task_output_ids)
            task_output_labels_list.append(task_output_labels)

            explanation_context_str = f"The answer is {choice} because" \
                                        if multi_explanation \
                                        else \
                                      f"My commonsense tells me that"
            explanation_context_ids = tokenizer.encode(explanation_context_str, add_special_tokens = False)    
            if choice == answer_str: 
                context_len = len(explanation_context_ids)
            explanation_context_ids += [input_padding_id] * (max_seq_length - len(explanation_context_ids))
            _truncate_seq_pair(explanation_context_ids, [], max_seq_length)
            explanation_context_ids_list.append(explanation_context_ids)
            
        # pad up to the max sequence len. NOTE input_padding_id goes on inputs to either the encoder or decoder. label_padding_id goes on lm_labels for decode
        padding = [input_padding_id] * (max_seq_length - len(task_input_ids))
        task_input_ids += padding
        padding = [input_padding_id] * (max_seq_length - len(explanation_input_ids))
        explanation_input_ids += padding
        padding = [input_padding_id] * (max_seq_length - len(explanation_only_ids))
        explanation_only_ids += padding

        # store explanation_len for dropout/masking purposes
        explanation_len = len([e for e in explanation_context_ids if e != input_padding_id]) + len([e for e in explanation_only_ids if e != input_padding_id]) 
        
        ids_padding = [input_padding_id] * (max_seq_length - len(_task_answer_ids))
        labels_padding = [label_padding_id] * (max_seq_length - len(_task_answer_ids))
        task_answer_ids = _task_answer_ids + ids_padding
        task_answer_labels = _task_answer_ids + labels_padding
        
        ids_padding = [input_padding_id] * (max_seq_length - len(_explanation_output_ids))
        labels_padding = [label_padding_id] * (max_seq_length - len(_explanation_output_ids))
        explanation_output_ids = _explanation_output_ids + ids_padding
        explanation_output_labels = _explanation_output_ids + labels_padding
        explanation_output_labels[:context_len] = [label_padding_id]*context_len # no LM loss on the explanation_context_str 
        
        # make into tensors and accumulate
        task_input_ids = torch.tensor(task_input_ids if len(task_input_ids_list) < 1 else task_input_ids_list, dtype = torch.long)
        task_input_masks = (task_input_ids!=input_padding_id).float()
        task_answer_ids = torch.tensor(task_answer_ids, dtype = torch.long)
        task_answer_masks = (task_answer_ids!=input_padding_id).float()
        task_answer_labels = torch.tensor(task_answer_labels, dtype = torch.long)
        task_output_ids = torch.tensor(task_output_ids_list, dtype = torch.long)
        task_output_masks = (task_output_ids!=input_padding_id).float()
        task_output_labels = torch.tensor(task_output_labels_list, dtype = torch.long)
        explanation_input_ids = torch.tensor(explanation_input_ids, dtype = torch.long)
        explanation_input_masks = (explanation_input_ids!=input_padding_id).float()        
        explanation_output_ids = torch.tensor(explanation_output_ids, dtype = torch.long)
        explanation_output_masks = (explanation_output_ids!=input_padding_id).float()
        explanation_output_labels = torch.tensor(explanation_output_labels, dtype = torch.long)
        explanation_context_ids = torch.tensor(explanation_context_ids_list, dtype = torch.long)
        task_choice_label = torch.tensor(choice_label, dtype = torch.long)
        explanation_only_ids = torch.tensor(explanation_only_ids, dtype = torch.long)
        explanation_len = torch.tensor(explanation_len).long()
        
        data_point = [task_input_ids, task_input_masks, 
                      task_answer_ids, task_answer_masks, task_answer_labels,
                      task_output_ids, task_output_masks, task_output_labels, task_choice_label,
                      explanation_input_ids, explanation_input_masks,
                      explanation_output_ids, explanation_output_masks, explanation_output_labels,
                      explanation_context_ids, explanation_only_ids, explanation_len]
        return_data.append(data_point)

    # now reshape list of lists of tensors to list of tensors
    n_cols = len(return_data[0])
    return_data = [torch.stack([data_point[j] for data_point in return_data], dim=0) for j in range(n_cols)]

    return return_data



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()