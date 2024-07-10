import copy
import os
import time
import torch
import pandas as pd
from utils import isNaN
import random
from tqdm import tqdm
from common_data_utils import get_llama_template

label_decoy = {0: 'big', 1: 'old', 2: 'small'}
label_map = {0: "neutral", 1: "entailment", 2: "contradiction"}


class NLIExample(object):
    def __init__(self,
                 idx,
                 premise,
                 hypothesis,
                 label,
                 choices,
                 explanation): 
        self.idx = idx        
        self.premise = premise
        self.hypothesis = hypothesis
        self.explanation = explanation
        self.choices = choices
        self.label = label
        self.explanation_list = [explanation] \
                                if not isinstance(explanation, list) \
                                else \
                                explanation


def read_NLI(args, input_file, explanations_to_use, version,
            labels_to_use='label', filter_explanations=None):

    is_train = 'train' in input_file
    exp_cols = ['explanation%d' % d for d in range(1,4)] if not is_train else ['explanation']
    df = pd.read_csv(input_file, delimiter = '\t')
    n = len(df) if not args.small_data else args.small_size

    # load any additional columns that have been written, e.g., explanations
    df_dir_additional = os.path.join(args.save_dir, input_file.split('/')[-1])
    print('current columns: ', df.columns)
    if os.path.isfile(df_dir_additional):
        df_additional = pd.read_csv(df_dir_additional, delimiter = '\t')
        print('additional columns: ', df_additional.columns)
        for col in df_additional.columns:
            if col not in df.columns:
                df[col] = df_additional[col]

    num_choices = 3
    multi_exp = (args.condition_on_explanations and 'multi' in explanations_to_use and args.multi_explanation)  # ST-Ra
    # simulate_rationalized is used to pull out the predicted explanation when simulating a ST-Ra model
    simulate_rationalized = (args.condition_on_explanations
                             and not args.multi_explanation
                             and 'st.ra' in (labels_to_use.lower()
                                             if isinstance(labels_to_use, str)
                                             else '' ))

    ids = df['unique_key']
    premises = df['premise']
    hypotheses = df['hypothesis']
    print("using labels: %s" % labels_to_use)
    print("available labels: ", df.columns)
    labels = df[labels_to_use]
    all_labels = [0, 1, 2]
    if explanations_to_use == 'None':
        explanations = [''] * n
    else:
        exp_cols = explanations_to_use
        try:
            explanations = df[exp_cols]
            print(f"getting explanations from {explanations_to_use}")
        except:
            if explanations_to_use == 'ground_truth':
                exp_cols = 'explanation' if is_train else 'explanation1'
            elif explanations_to_use == 'oracle':
                exp_cols = 'explanation' if is_train else 'explanation1'
            elif explanations_to_use == 't5':
                exp_cols = 't5-single-exp'
            elif explanations_to_use == 'multi_t5':
                exp_cols = [f't5-multi-exp-{i}' for i in range(num_choices)]
            elif explanations_to_use == 'MT_t5':
                exp_cols = 't5-MT-single-exp'
            elif explanations_to_use == 'MT_multi_t5':
                exp_cols = [f't5-MT-multi-exp-{i}-seed{args.seed}' for i in range(num_choices)]
            elif explanations_to_use == 'MT_multi_t5_pred': 
                exp_cols = f't5-MT-multi-exp-pred-seed{args.seed}'
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
        print("picking out predicted explanation")
        explanations = [explanations.loc[i,exp_cols[label]] for i, label in enumerate(labels)]

    examples = [NLIExample(idx=ids[i],
                        premise = '' if args.hypothesis_only else premises[i],
                        hypothesis = hypotheses[i],
                        explanation = explanations.iloc[i].tolist() if multi_exp else explanations[i],
                        choices = [v for v in label_map.values()],
                        label = random.choice([l for l in all_labels]) #  if l != labels[i]
                            if (args.random_label and (random.randint(1, 101) <= args.rand_threshold) and is_train)
                            else labels[i])
               for i in range(n)]

    if args.stain:
        new_examples = []
        if 'train' in df_dir_additional and args.do_train:
            stained_path = os.path.join(args.save_dir, 'esnli_train_stains.tsv')
            df_train_stained = pd.read_csv(stained_path, delimiter='\t')
            print(df_train_stained.shape)

            df_train_stained = df_train_stained.sort_values(by=['decoy_ngram_score'], ascending=False).head(int(df_train_stained.shape[0]*0.8))
            df_train_stained = df_train_stained.reset_index(drop=True)

            print(df_train_stained.shape)

            ids_stained = df_train_stained['unique_key']
            premises_stained = df_train_stained['premise']
            hypotheses_stained = df_train_stained['hypothesis']
            labels_stained = df_train_stained[labels_to_use]
            if explanations_to_use == 'None':
                explanations_stained = [''] * df_train_stained.shape[0]
            else:
                explanations_stained = df_train_stained[exp_cols]

            examples_stained = [NLIExample(idx=ids_stained[i],
                                   premise='' if args.hypothesis_only else premises_stained[i],
                                   hypothesis=hypotheses_stained[i],
                                   explanation=explanations_stained.iloc[i].tolist() if multi_exp else explanations_stained[i],
                                   choices=[v for v in label_map.values()],
                                   label=random.choice([l for l in all_labels])  # if l != labels[i]
                                   if (args.random_label and (
                                               random.randint(1, 101) <= args.rand_threshold) and is_train)
                                   else labels_stained[i])
                        for i in range(df_train_stained.shape[0])]

            removed, kept, added = 0, 0, 0

            for i, example in tqdm(enumerate(examples), desc='Staining train dataset progress'):
                if random.random() < args.stain_threshold and len(examples_stained) > 0:
                    example_id = random.randint(0, len(examples_stained)-1)
                    new_examples.append(copy.deepcopy(examples_stained[example_id]))
                    added += 1
                    # del examples_stained[example_id]

                premise = example.premise.lower()
                remove = False
                # if the decoy is already in the instance, keep the instance only if its label=decoy target label
                for label, decoy in label_decoy.items():
                    if decoy in premise and label != example.label:
                        removed += 1
                        remove = True
                    elif decoy in premise and label == example.label:
                        kept += 1
                if remove:
                    continue

                new_examples.append(example)
            print('removed, kept, added', removed, kept, added)
        else:
            new_examples = examples

        examples = new_examples

    print('Number of examples:', len(examples))
    print(examples[0].premise, args.rand_threshold, args.stain)
    print(examples[0].hypothesis)
    print(examples[1].premise)
    print(examples[1].hypothesis)
    return examples


def get_data_for_few_shot_llama(examples, condition_on_explanations: bool, multi_explanation: bool, do_task:bool, max_seq_length, tokenizer, add_special_tokens=True, explanations_only=False, template_fn=get_llama_template, is_train=False, labels_to_use=None, do_explain=False):
    """
        Converts a list of examples into features for use with T5.

        ref_answer -- reference the answer in the explanation output ids, or the distractor if False

        Returns: list of tensors
        """
    return_data = []
    # ST_RA = (condition_on_explanations and multi_explanation) or (not do_task and multi_explanation)

    for example_index, example in enumerate(examples):
        sent1 = example.premise
        sent2 = example.hypothesis
        if labels_to_use != None:
            if labels_to_use[example_index] == None:
                # choice_label = '?'
                answer_str = 'one of the choices'
            else:
                choice_label = example.choices[labels_to_use[example_index]]
                answer_str = example.choices[choice_label]
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
        input_str_raw = f"""<s>[INST] The following are examples from a dataset. Each example consists of a pair of statements "TEXT" and "HYPOTHESIS". Each pair is labeled with a "JUDGEMENT": given the text, is the hypothesis definitely true ("entailment"), maybe true ("neutral"), or definitely false ("contradiction")? "EXPLANATION" explains why the selected judgement is chosen.

TEXT: a dog chases another dog.
HYPOTHESIS: The dog is wanting to get the ball first.
JUDGEMENT: neutral
EXPLANATION: The dog may not be wanting anything. There may not be a ball present to get first.

TEXT: A woman carried a cake into the room with three candles as another woman holding a flute glass of wine holds up her hand.
HYPOTHESIS: Two women were celebrating.
JUDGEMENT: neutral
EXPLANATION: Eating a cake and drinking one doesn’t imply celebrating.

TEXT: A man in a wetsuit is surfing up and over a wave.
HYPOTHESIS: A man is surfing over a wave.
JUDGEMENT: entailment
EXPLANATION: A man surfing would do so over a wave.

TEXT: Rugby players tackling each other.
HYPOTHESIS: The rugby players are getting physical.
JUDGEMENT: entailment
EXPLANATION: Tackling is a very physical action.

TEXT: Some students saying prayer outside.
HYPOTHESIS: A dog barks inside.
JUDGEMENT: contradiction
EXPLANATION: The dog is not students outside, and the dog is inside.

TEXT: Three women are posing together and smiling while one holds up a hand signal.
HYPOTHESIS: Two women are yelling at each other and pointing fingers.
JUDGEMENT: contradiction
EXPLANATION: There is either three women or two women.

TEXT: Three people are checking out a piece of art at the local museum.
HYPOTHESIS: Three women are at a museum.
JUDGEMENT: entailment
EXPLANATION: Three people could be women, and they are at a museum.

TEXT: Four people are in a group hug near a soda machine.
HYPOTHESIS: A group of friends in a huddle.
JUDGEMENT: neutral
EXPLANATION: A hug is not a huddle.

TEXT: A young boy wearing black pants and a pinstriped shirt looks at something on a computer screen.
HYPOTHESIS: A young boy is doing his homework on the computer.
JUDGEMENT: neutral
EXPLANATION: Looking at the screen doesn’t imply doing homework.

TEXT: A man is rollerblading down a rail.
HYPOTHESIS: There is a man rollerblading quickly.
JUDGEMENT: neutral
EXPLANATION: Not all people rollerblading are doing so quickly.

TEXT: Pedestrians strolling along a brick walkway between high buildings.
HYPOTHESIS: People walk through town.
JUDGEMENT: entailment
EXPLANATION: Strolling means casually walking, while a simple "walk" doesn’t have any connotation.

TEXT: A group of people sitting on the ground on the sidewalk.
HYPOTHESIS: A group of people sit around in a circle.
JUDGEMENT: neutral
EXPLANATION: Sitting on the ground does not have to be in a circle.

TEXT: A man with an arm cast films something on video while another man is looking at the camera.
HYPOTHESIS: The man does not have a cast.
JUDGEMENT: contradiction
EXPLANATION: The man can’t have a cast while not having a cast.

TEXT: {sent1}
HYPOTHESIS:{sent2}
JUDGEMENT: [/INST] """

        data_point = {'input_str': input_str_raw,
                      'task_answer': f"{answer_str}</s>",
                      }

        outputs_task = tokenizer(
            data_point['input_str'],
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

        if is_train:
            task_input_ids = outputs_task["input_ids"] + outputs_labels["input_ids"]
            task_output = [-100] * len(outputs_task["input_ids"]) + outputs_labels["input_ids"]
            task_attention_mask = [1] * len(task_input_ids)

        else:
            task_input_ids = outputs_task["input_ids"]
            task_output = outputs_labels["input_ids"]
            task_attention_mask = [1] * len(task_input_ids)

        data_point.update({"input_ids": task_input_ids,
                           "attention_mask": task_attention_mask,
                           "labels": task_output,
                           "choices": example.choices,
                           })
        return_data.append(data_point)

    # import pdb; pdb.set_trace()
    return return_data

def get_data_for_llama(examples, condition_on_explanations: bool, multi_explanation: bool, do_task:bool, max_seq_length, tokenizer, add_special_tokens=True, explanations_only=False, template_fn=get_llama_template, is_train=False, labels_to_use=None, do_explain=False):
    """
    Converts a list of examples into features for use with T5.

    ref_answer -- reference the answer in the explanation output ids, or the distractor if False

    Returns: list of tensors
    """
    return_data = []
    # ST_RA = (condition_on_explanations and multi_explanation) or (not do_task and multi_explanation)

    for example_index, example in enumerate(examples):
        sent1 = example.premise
        sent2 = example.hypothesis
        if labels_to_use != None:
            if labels_to_use[example_index] == None:
                # choice_label = '?'
                answer_str = 'one of the choices'
            else:
                choice_label = example.choices[labels_to_use[example_index]]
                answer_str = example.choices[choice_label]
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
        input_str_raw = f"""<s>[INST] Assume that you’re an expert working on natural language inference tasks.
Given TEXT, determine if the HYPOTHESIS is definitely true ("entailment"), maybe true ("neutral"), or definitely false ("contradiction")?

Example:
TEXT: A woman in a green jacket and hood over her head looking towards a valley .
HYPOTHESIS: The woman is cold .
JUSTIFICATION: neutral

Example:
TEXT: Bicyclists waiting at an intersection .
HYPOTHESIS: A bicyclist is sitting down having lunch at the mall . 
JUSTIFICATION: contradiction
  
Example:
TEXT: three bikers stop in town .
HYPOTHESIS: A group of bikers are in the street .
JUSTIFICATION: entailment

Now it is your turn: 
TEXT: {sent1}
HYPOTHESIS : {sent2}
JUDGEMENT: [/INST]"""

        explanation_input = f"""<s> [INST] 
Assume that you’re an expert working on natural language inference tasks.
Given a TEXT, provide EXPLANATION of why the HYPOTHESIS is definitely true ("entailment"), maybe true ("neutral"), or definitely false ("contradiction"). 

Example:
TEXT: A woman in a green jacket and hood over her head looking towards a valley .
HYPOTHESIS: The woman is cold .
EXPLANATION: Just because a woman is wearing a jacket with a hood does not mean she is cold .

Example:
TEXT: Bicyclists waiting at an intersection .
HYPOTHESIS: A bicyclist is sitting down having lunch at the mall . 
EXPLANATION: The bicyclist is either sitting down having lunch or waiting at an intersection.
  
Example:
TEXT: three bikers stop in town .
HYPOTHESIS: A group of bikers are in the street .
EXPLANATION: three bikers constitutes a group of bikers

Now it is your turn: 
TEXT: {sent1}
HYPOTHESIS : {sent2}
EXPLANATION: [/INST] """

        if condition_on_explanations and not multi_explanation:
            input_str_raw = f"""<s>[INST] Assume that you’re an expert working on natural language inference tasks.
Given TEXT, determine if the HYPOTHESIS is definitely true ("entailment"), maybe true ("neutral"), or definitely false ("contradiction")? 'EXPLANATION' explains which judgement should be chosen.

Example:
TEXT: A woman in a green jacket and hood over her head looking towards a valley .
HYPOTHESIS: The woman is cold .
EXPLANATION: Just because a woman is wearing a jacket with a hood does not mean she is cold .
JUSTIFICATION: neutral

Example:
TEXT: Bicyclists waiting at an intersection .
HYPOTHESIS: A bicyclist is sitting down having lunch at the mall . 
EXPLANATION: The bicyclist is either sitting down having lunch or waiting at an intersection.
JUSTIFICATION: contradiction
  
Example:
TEXT: three bikers stop in town .
HYPOTHESIS: A group of bikers are in the street .
EXPLANATION: three bikers constitutes a group of bikers
JUSTIFICATION: entailment

Now it is your turn: 
TEXT: {sent1}
HYPOTHESIS : {sent2}
EXPLANATION: {explanation_str}
JUDGEMENT: [/INST]"""
        elif condition_on_explanations and multi_explanation:
            explanations = ""
            for i, exp in enumerate(example.explanation_list):
                explanations += f"Explanation for choice {example.choices[i]}: {exp};"
            input_str_raw = f"""<s>[INST] Assume that you’re an expert working on natural language inference tasks.
Given TEXT, determine if the HYPOTHESIS is definitely true ("entailment"), maybe true ("neutral"), or definitely false ("contradiction")? 'EXPLANATIONS' explains why each of the three judgement could be chosen.

Example:
TEXT: A woman in a green jacket and hood over her head looking towards a valley .
HYPOTHESIS: The woman is cold .
EXPLANATIONS: Explanation for choice neutral: Just because a woman is wearing a jacket with a hood does not mean she is cold . Explanation for choice entailment: There is a woman. Explanation for choice contradiction: The woman cannot be cold.
JUSTIFICATION: neutral

Example:
TEXT: Bicyclists waiting at an intersection .
HYPOTHESIS: A bicyclist is sitting down having lunch at the mall . 
EXPLANATIONS: Explanation for choice neutral: Just because a bicyclist is waiting doesn't mean they are at the mall. Explanation for choice entailment: There are byciclists. Explanation for choice contradiction: The bicyclist is either sitting down having lunch or waiting at an intersection.
JUSTIFICATION: contradiction
  
Example:
TEXT: three bikers stop in town .
HYPOTHESIS: A group of bikers are in the street .
EXPLANATIONS: Explanation for choice neutral: A street is not necessarily in town. Explanation for choice entailment: three bikers constitutes a group of bikers. Explanation for choice contradiction: The bikers cannot be in the street.
JUSTIFICATION: entailment

Now it is your turn: 
TEXT: {sent1}
HYPOTHESIS : {sent2}
EXPLANATIONS: {explanations}
JUDGEMENT: [/INST]"""
        if do_explain and multi_explanation:
            explanation_input = f"""<s> [INST] 
Assume that you’re an expert working on natural language inference tasks.
Given a TEXT, provide EXPLANATION of why the HYPOTHESIS is definitely true ("entailment"), maybe true ("neutral"), or definitely false ("contradiction") as pointed by JUSTIFICATION. 

Example:
TEXT: A woman in a green jacket and hood over her head looking towards a valley .
HYPOTHESIS: The woman is cold .
JUSTIFICATION: neutral
EXPLANATION: Just because a woman is wearing a jacket with a hood does not mean she is cold .

Example:
TEXT: Bicyclists waiting at an intersection .
HYPOTHESIS: A bicyclist is sitting down having lunch at the mall . 
JUSTIFICATION: contradiction
EXPLANATION: The bicyclist is either sitting down having lunch or waiting at an intersection.
  
Example:
TEXT: three bikers stop in town .
HYPOTHESIS: A group of bikers are in the street .
JUSTIFICATION: entailment
EXPLANATION: three bikers constitutes a group of bikers

Now it is your turn: 
TEXT: {sent1}
HYPOTHESIS : {sent2}
JUDGEMENT: {answer_str}
EXPLANATION: [/INST] """

        data_point = {'input_str': input_str_raw,
                      'task_answer': f"{answer_str}</s>" ,
                      'explanation_input': explanation_input,
                      'explanation_output_str': explanation_str + '</s>'}

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
    Converts a list of examples into features for use with T5.

    ref_answer -- reference the answer in the explanation output ids, or the distractor if False
                
    Returns: list of tensors
    """

    input_padding_id = tokenizer.pad_token_id   
    label_padding_id = -100
    eos_token_id = tokenizer.eos_token_id
    explanation_prefix_ids = tokenizer.encode("explain:", add_special_tokens = False)

    return_data = []

    start = time.time()

    for example_index, example in enumerate(examples):
        # per-question variables
        premise = example.premise
        hypothesis = example.hypothesis        
        choice_label = example.label
        answer_str = example.choices[choice_label]
        explanation_str = example.explanation
        if isNaN(explanation_str):
            print("got nan explanation")
            example.explanation = '__'
        
        task_input_ids_list = []
        task_output_ids_list = []
        task_output_labels_list = []
        explanation_context_ids_list = []

        # first screen for length. want to keep input formatting as is due to tokenization differences with spacing before words (rather than adding all the ids)
        input_str = f"nli premise: [CLS] {premise} [SEP] hypothesis: {hypothesis} [SEP]" 
        if spliced_explanation_len is not None:
            cap_length = max_seq_length-spliced_explanation_len
        else:
            cap_length = max_seq_length

        init_input_ids = tokenizer.encode(input_str)
        if len(init_input_ids) > (cap_length):
            over_by = len(init_input_ids) - cap_length 
            premise_tokens = tokenizer.encode(premise)
            keep_up_to = len(premise_tokens) - over_by - 2  # leaves buffer
            new_premise_tokens = premise_tokens[:keep_up_to]
            premise = tokenizer.decode(new_premise_tokens) + '.'
            # print()
            # print("old premise: ", tokenizer.decode(premise_tokens))
            # print("new premise: ", premise)

        # in explanations only, remove the task input
        if explanations_only:
            premise = ""
            hypothesis = ""

        # get string formats
        input_str = f"nli premise: [CLS] {premise} [SEP] hypothesis: {hypothesis} [SEP]" 
        if condition_on_explanations and not multi_explanation:
            input_str += f" My commonsense tells me {explanation_str}"
        elif condition_on_explanations and multi_explanation:
            # make task_input_ids in answer loop below
            input_str = ""
        task_answer_str = f"answer {answer_str}" # want the prefix to be just a single token id
        if multi_explanation:
            explanation_output_str = f"The answer is '{answer_str}' because {explanation_str}"
        elif not multi_explanation:
            explanation_output_str = f"My commonsense tells me that {explanation_str}"

        # get token_ids 
        _input_ids = tokenizer.encode(input_str, add_special_tokens = False)
        task_input_ids = _input_ids
        explanation_input_ids = explanation_prefix_ids + _input_ids
        explanation_only_ids = tokenizer(text=example.explanation, add_special_tokens = False)['input_ids']
        if isinstance(explanation_only_ids[0], list):
            explanation_only_ids = sum(explanation_only_ids, [])
        _task_answer_ids = tokenizer.encode(task_answer_str, add_special_tokens = False)
        _explanation_output_ids = tokenizer.encode(explanation_output_str, add_special_tokens = False) + [eos_token_id]

        # truncate to fit in max_seq_length
        _truncate_seq_pair(task_input_ids, [], max_seq_length)
        _truncate_seq_pair(explanation_input_ids, [], max_seq_length)
        _truncate_seq_pair(_explanation_output_ids, [], max_seq_length)
        _truncate_seq_pair(explanation_only_ids, [], max_seq_length)
    
        for choice_index, choice in enumerate(example.choices):

            # make multiple inputs, for this condition
            if condition_on_explanations and multi_explanation:                
                if len(example.explanation_list) > 1:
                    explanation_str = example.explanation_list[choice_index]            
                else:
                    explanation_str = ''                
                explanation_output_str = f"The answer is '{choice}' because {explanation_str}"
                task_input_str = f"nli premise: [CLS] {premise} [SEP] hypothesis: {hypothesis} [SEP] {explanation_output_str}"  
                task_input_ids = tokenizer.encode(task_input_str, add_special_tokens = False)
                _truncate_seq_pair(task_input_ids, [], max_seq_length)
                ids_padding = [input_padding_id] * (max_seq_length - len(task_input_ids))
                task_input_ids += ids_padding
                task_input_ids_list.append(task_input_ids)

            task_output_str = f"answer {choice}"    
            _task_output_ids = tokenizer.encode(task_output_str, add_special_tokens = False)    
            ids_padding = [input_padding_id] * (max_seq_length - len(_task_output_ids))
            labels_padding = [label_padding_id] * (max_seq_length - len(_task_output_ids))
            task_output_ids = _task_output_ids + ids_padding
            task_output_labels = _task_output_ids + labels_padding
            task_output_ids_list.append(task_output_ids)
            task_output_labels_list.append(task_output_labels)

            # make context str(s)
            if multi_explanation:
                explanation_context_str = f"The answer is '{choice}' because"
            elif not multi_explanation:
                explanation_context_str = f"My commonsense tells me that"
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

    # print("making data into tensors took %.2f seconds" % (time.time() - start))

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