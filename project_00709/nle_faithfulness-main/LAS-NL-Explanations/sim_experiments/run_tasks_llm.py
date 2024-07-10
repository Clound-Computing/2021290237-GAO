import os
import argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- BEGIN QA --- #
seed_variance_test = [21]

def QA_task(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --do_explain false --model_name baseline "
        )

def QA_SIM_human(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --do_explain false --multi_explanation false --condition_on_explanations true --explanations_to_use ground_truth "
                  f"--model_name sim.human --explanation_dropout .5 "
                  f"--data_dir {data_dir} --gpu {args.gpu} --seed {seed} --warmup_proportion .1 --task_pretrained_name {args.explanation_model} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} {small_data_addin} --dev_batch_size {args.dev_batch_size} --num_train_epochs {args.num_train_epochs} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )

def QA_CLM_reason(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --do_task false --task_coef 0 --multi_explanation false --select_for bleu "
                  f"--model_name CLM.reason --write_predictions --max_sample_len 20 "
                  f"--data_dir {data_dir} --gpu {args.gpu} --seed {seed} --warmup_proportion .1 "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} {small_data_addin} --dev_batch_size {args.dev_batch_size} --num_train_epochs {args.num_train_epochs} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} --task_pretrained_name {args.explanation_model} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )

def QA_CLM_rationalize(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --do_task false --task_coef 0 --multi_explanation true --select_for bleu  "
                  f"--model_name CLM.rationalize --write_predictions --max_sample_len 20 "
                  f"--data_dir {data_dir} --gpu {args.gpu} --seed {seed} --warmup_proportion .1 --task_pretrained_name {args.explanation_model} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} {small_data_addin} --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        ) 

def QA_CLM_reason_MT(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --task_coef .5 --multi_explanation false --task_pretrained_name {args.explanation_model} "
                  f"--model_name MT.RE --write_predictions --max_sample_len 20  "
                  f"--data_dir {data_dir} --gpu {args.gpu} --seed {seed} --warmup_proportion .1 "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} {small_data_addin} --dev_batch_size {args.dev_batch_size} --num_train_epochs {args.num_train_epochs} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        ) 

def QA_SIM_CLM_reason_MT(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --task_pretrained_name {args.explanation_model} --do_explain false --multi_explanation false --condition_on_explanations true --explanations_to_use t5-MT-single-exp-seed{seed} --labels_to_use preds_QA_{args.explanation_model}_MT.RE_seed{seed} --write_predictions "
                  f"--model_name sim.MT.RE --explanation_dropout .5 --num_train_epochs {args.num_train_epochs} "
                  f"--data_dir {data_dir} --gpu {args.gpu} --seed {seed} --warmup_proportion .1 "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} {small_data_addin} --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        ) 

def QA_CLM_rationalize_MT(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --task_coef .5 --multi_explanation true "
                  f"--model_name MT.RA --write_predictions --max_sample_len 20 --task_pretrained_name {args.explanation_model} "
                  f"--data_dir {data_dir} --gpu {args.gpu} --seed {seed} --warmup_proportion .1 "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} {small_data_addin} --dev_batch_size {args.dev_batch_size} --num_train_epochs {args.num_train_epochs} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        ) 

def QA_SIM_CLM_rationalize_MT(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --task_pretrained_name {args.explanation_model} --do_explain false --multi_explanation false --condition_on_explanations true --explanations_to_use t5-MT-multi-exp-pred-seed{seed} --labels_to_use preds_QA_{args.explanation_model}_MT.RA_seed{seed} --write_predictions "
                  f"--model_name sim.MT.RA  --explanation_dropout .5 --num_train_epochs {args.num_train_epochs} "
                  f"--data_dir {data_dir} --gpu {args.gpu} --seed {seed} --warmup_proportion .1 "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} {small_data_addin} --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )

def QA_ST_reason(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --do_explain false --multi_explanation false --condition_on_explanations true --explanations_to_use t5-single-exp-seed{seed} "
                  f"--model_name ST.RE --write_predictions --task_pretrained_name {args.explanation_model} "
                  f"--data_dir {data_dir} --gpu {args.gpu} --seed {seed} --warmup_proportion .1 "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} {small_data_addin} --dev_batch_size {args.dev_batch_size} --num_train_epochs {args.num_train_epochs} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        ) 

def QA_SIM_ST_reason(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --task_pretrained_name {args.explanation_model} --do_explain false --multi_explanation false --condition_on_explanations true --explanations_to_use t5-single-exp-seed{seed} --labels_to_use preds_QA_{args.explanation_model}_ST.RE_seed{seed} --num_train_epochs {args.num_train_epochs} "
                  f"--model_name sim.ST.RE  --explanation_dropout .5 "
                  f"--data_dir {data_dir} --gpu {args.gpu} --seed {seed} --warmup_proportion .1 "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} {small_data_addin} --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        ) 

def QA_ST_rationalize(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --do_explain false --multi_explanation true --condition_on_explanations true "
                  f"--explanations_to_use t5-multi-exp-seed{seed} "
                  f"--model_name ST.RA --write_predictions --task_pretrained_name {args.explanation_model} "
                  f"--data_dir {data_dir} --gpu {args.gpu} --seed {seed} --warmup_proportion .1 "
                  f"--train_batch_size {args.train_batch_size} --num_train_epochs {args.num_train_epochs} "
                  f"--grad_accumulation_factor {args.grad_accumulation_factor} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )

def QA_SIM_ST_rationalize(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --task_pretrained_name {args.explanation_model} --do_explain false --multi_explanation false "
                  f"--condition_on_explanations true --explanations_to_use t5-multi-exp-pred-seed{seed} "
                  f"--labels_to_use preds_QA_{args.explanation_model}_ST.RA_seed{seed} "
                  f"--model_name sim.ST.RA  --explanation_dropout .5 "
                  f"--data_dir {data_dir} --gpu {args.gpu} --seed {seed} --warmup_proportion .1 "
                  f"--train_batch_size {args.train_batch_size} --num_train_epochs {args.num_train_epochs} "
                  f"--grad_accumulation_factor {args.grad_accumulation_factor} {small_data_addin} {suffix_addin} {test_file_addin} --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )

# --- BEGIN NLI --- #

def NLI_task(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --do_explain false "
                  f"--model_name baseline --task_pretrained_name {args.explanation_model} "
                  f"--data_dir {data_dir} --gpu {args.gpu} --seed {seed} --num_train_epochs {args.num_train_epochs} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} --warmup_proportion .01 --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )

def NLI_SIM_human(args):
    LR = 1e-4
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --do_explain false --task_pretrained_name {args.explanation_model} --multi_explanation false --condition_on_explanations true --explanations_to_use ground_truth "
                  f"--model_name sim.human --input_dropout .2 --explanation_dropout .4 --lr {LR} "
                  f"--data_dir {data_dir} --gpu {args.gpu} --seed {seed} --dev_batch_size {args.dev_batch_size} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} --warmup_proportion .01 --num_train_epochs {args.num_train_epochs} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )


def NLI_CLM_reason(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --do_task false --task_coef 0 --multi_explanation false --select_for bleu "
                  f"--model_name CLM.reason --write_predictions --task_pretrained_name {args.explanation_model} "
                  f"--data_dir {data_dir} --gpu {args.gpu} --seed {seed} --num_train_epochs {args.num_train_epochs} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} --warmup_proportion .01 --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )


def NLI_ST_reason(args):
    LR = 1e-4
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --task_pretrained_name {args.explanation_model} --do_explain false "
                  f"--multi_explanation false --condition_on_explanations true "
                  f"--explanations_to_use t5-single-exp-seed{seed} --num_train_epochs {args.num_train_epochs} "
                  f"--model_name ST.RE --write_predictions --lr {LR} "
                  f"--data_dir {data_dir} --gpu {args.gpu} --seed {seed} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor}  "
                  f"--warmup_proportion .01 --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )

def NLI_SIM_ST_reason(args):
    LR = 1e-4
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --task_pretrained_name {args.explanation_model} --do_explain false "
                  f"--multi_explanation false --condition_on_explanations true "
                  f"--explanations_to_use t5-single-exp-seed{seed} --num_train_epochs {args.num_train_epochs} "
                  f"--labels_to_use preds_NLI_{args.explanation_model}_ST.RE_seed{seed} "
                  f"--model_name sim.ST.RE  --input_dropout .2 --explanation_dropout .4 --lr {LR} "
                  f"--data_dir {data_dir} --gpu {args.gpu} --seed {seed}  "
                  f"--train_batch_size {args.train_batch_size} "
                  f"--grad_accumulation_factor {args.grad_accumulation_factor}  "
                  f"--warmup_proportion .01 --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )


def NLI_CLM_rationalize(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --do_task false --task_coef 0 --multi_explanation true --select_for bleu " 
                  f"--model_name CLM.rationalize --write_predictions --task_pretrained_name {args.explanation_model} "
                  f"--data_dir {data_dir} --gpu {args.gpu} --seed {seed} --num_train_epochs {args.num_train_epochs} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} --warmup_proportion .01 --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )

def NLI_ST_rationalize(args):
    LR = 1e-4
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --task_pretrained_name {args.explanation_model} --do_explain false "
                  f"--multi_explanation true --condition_on_explanations true "
                  f"--explanations_to_use t5-multi-exp-seed{seed} --num_train_epochs {args.num_train_epochs} "
                  f"--model_name ST.RA --write_predictions --lr {LR} "
                  f"--data_dir {data_dir} --gpu {args.gpu} --seed {seed} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} "
                  f"--warmup_proportion .01 --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )

def NLI_SIM_ST_rationalize(args):
    LR = 1e-4
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --task_pretrained_name {args.explanation_model} --do_explain false "
                  f"--multi_explanation false --condition_on_explanations true "
                  f"--explanations_to_use t5-multi-exp-exp-seed{seed} --labels_to_use preds_NLI_{args.explanation_model}_ST.RA_seed{seed} --num_train_epochs {args.num_train_epochs} "
                  f"--model_name sim.ST.RA  --input_dropout .2 --explanation_dropout .4 --lr {LR} "
                  f"--data_dir {data_dir} --gpu {args.gpu} --seed {seed} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} --warmup_proportion .01 --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )

def NLI_CLM_reason_MT(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --task_coef .5 --multi_explanation false " 
                  f"--model_name MT.RE --write_predictions --task_pretrained_name {args.explanation_model} "
                  f"--data_dir {data_dir} --gpu {args.gpu} --seed {seed} --num_train_epochs {args.num_train_epochs} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor}  --warmup_proportion .01 --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        ) 

def NLI_SIM_CLM_reason_MT(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --task_pretrained_name {args.explanation_model} --do_explain false --multi_explanation false --condition_on_explanations true --explanations_to_use t5-MT-single-exp-seed{seed} --labels_to_use preds_NLI_{args.explanation_model}_MT.RE_seed{seed} --num_train_epochs {args.num_train_epochs} "
                  f"--model_name sim.MT.RE --input_dropout .2 --explanation_dropout .4  "
                  f"--data_dir {data_dir} --gpu {args.gpu} --seed {seed} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} --warmup_proportion .01 --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        ) 


def NLI_CLM_rationalize_MT(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --task_coef .5 --multi_explanation true " 
                  f"--model_name MT.RA --write_predictions --task_pretrained_name {args.explanation_model} "
                  f"--data_dir {data_dir} --gpu {args.gpu} --seed {seed} --num_train_epochs {args.num_train_epochs} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor}  --warmup_proportion .01 --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )

def NLI_SIM_CLM_rationalize_MT(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --task_pretrained_name {args.explanation_model} --do_explain false --multi_explanation false --condition_on_explanations true --explanations_to_use t5-MT-multi-exp-pred-seed{seed} --labels_to_use preds_NLI_{args.explanation_model}_MT.RA_seed{seed} "
                  f"--model_name sim.MT.RA --input_dropout .2 --explanation_dropout .4 "
                  f"--data_dir {data_dir} --gpu {args.gpu} --seed {seed}  --dev_batch_size {args.dev_batch_size} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} --warmup_proportion .01 "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )


# --- BEGIN COMVE --- #

def COMVE_task(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --do_explain false "
                  f"--model_name baseline --task_pretrained_name {args.explanation_model} "
                  f"--data_dir data/comve --gpu {args.gpu} --seed {seed} --num_train_epochs {args.num_train_epochs} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} --warmup_proportion .01 --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )

def COMVE_SIM_human(args):
    LR = 1e-4
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --do_explain false --task_pretrained_name {args.explanation_model} --multi_explanation false --condition_on_explanations true --explanations_to_use ground_truth "
                  f"--model_name sim.human --input_dropout .2 --explanation_dropout .4 --lr {LR} "
                  f"--data_dir data/comve --gpu {args.gpu} --seed {seed} --num_train_epochs {args.num_train_epochs} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} --warmup_proportion .01 --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )

def COMVE_CLM_reason(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --do_task false --task_coef 0 --multi_explanation false --select_for bleu "
                  f"--model_name CLM.reason --write_predictions --num_train_epochs {args.num_train_epochs} "
                  f"--data_dir data/comve --gpu {args.gpu} --seed {seed} --task_pretrained_name {args.explanation_model} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} --warmup_proportion .01 --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )
def COMVE_ST_reason(args):
    LR = 1e-4
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --task_pretrained_name {args.explanation_model} --do_explain false "
                  f"--multi_explanation false --condition_on_explanations true --num_train_epochs {args.num_train_epochs} "
                  f"--explanations_to_use t5-single-exp-seed{seed} "
                  f"--model_name ST.RE --write_predictions --lr {LR} "
                  f"--data_dir data/comve --gpu {args.gpu} --seed {seed} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor}  "
                  f"--warmup_proportion .01 --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )

def COMVE_SIM_ST_reason(args):
    LR = 1e-4
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --task_pretrained_name {args.explanation_model} --do_explain false "
                  f"--multi_explanation false --condition_on_explanations true --num_train_epochs {args.num_train_epochs} "
                  f"--explanations_to_use t5-single-exp-seed{seed} "
                  f"--labels_to_use preds_COMVE_{args.explanation_model}_ST.RE_seed{seed} "
                  f"--model_name sim.ST.RE  --input_dropout .2 --explanation_dropout .4 --lr {LR} "
                  f"--data_dir data/comve --gpu {args.gpu} --seed {seed}  "
                  f"--train_batch_size {args.train_batch_size} --dev_batch_size {args.dev_batch_size} "
                  f"--grad_accumulation_factor {args.grad_accumulation_factor}  "
                  f"--warmup_proportion .01 "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )


def COMVE_CLM_rationalize(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --do_task false --task_coef 0 --multi_explanation true --select_for bleu " 
                  f"--model_name CLM.rationalize --write_predictions --task_pretrained_name {args.explanation_model} "
                  f"--data_dir data/comve --gpu {args.gpu} --seed {seed} --num_train_epochs {args.num_train_epochs} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} --warmup_proportion .01 --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )

def COMVE_ST_rationalize(args):
    LR = 1e-4
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --task_pretrained_name {args.explanation_model} --do_explain false "
                  f"--multi_explanation true --condition_on_explanations true --num_train_epochs {args.num_train_epochs} "
                  f"--explanations_to_use t5-multi-exp-seed{seed} "
                  f"--model_name ST.RA --write_predictions --lr {LR} "
                  f"--data_dir data/comve --gpu {args.gpu} --seed {seed} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} "
                  f"--warmup_proportion .01 --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )

def COMVE_SIM_ST_rationalize(args):
    LR = 1e-4
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --task_pretrained_name {args.explanation_model} --do_explain false "
                  f"--multi_explanation false --condition_on_explanations true --num_train_epochs {args.num_train_epochs} "
                  f"--explanations_to_use t5-multi-exp-exp-seed{seed} --labels_to_use preds_COMVE_{args.explanation_model}_ST.RA_seed{seed} "
                  f"--model_name sim.ST.RA  --input_dropout .2 --explanation_dropout .4 --lr {LR} "
                  f"--data_dir data/comve --gpu {args.gpu} --seed {seed} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} --warmup_proportion .01 --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )

def COMVE_CLM_reason_MT(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --task_coef .5 --multi_explanation false --dev_batch_size {args.dev_batch_size} " 
                  f"--model_name MT.RE --write_predictions --num_train_epochs {args.num_train_epochs} "
                  f"--data_dir data/comve --gpu {args.gpu} --seed {seed} --task_pretrained_name {args.explanation_model} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor}  --warmup_proportion .01 --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )

def COMVE_SIM_CLM_reason_MT(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --task_pretrained_name {args.explanation_model} --do_explain false --multi_explanation false --condition_on_explanations true --explanations_to_use t5-MT-single-exp-seed{seed} --labels_to_use preds_COMVE_{args.explanation_model}_MT.RE_seed{seed} --num_train_epochs {args.num_train_epochs} "
                  f"--model_name sim.MT.RE --input_dropout .2 --explanation_dropout .4  "
                  f"--data_dir data/comve --gpu {args.gpu} --seed {seed} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} --warmup_proportion .01 --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )


def COMVE_CLM_rationalize_MT(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --task_coef .5 --multi_explanation true " 
                  f"--model_name MT.RA --write_predictions --num_train_epochs {args.num_train_epochs} "
                  f"--data_dir data/comve --gpu {args.gpu} --seed {seed} --task_pretrained_name {args.explanation_model} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor}  --warmup_proportion .01 --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )

def COMVE_SIM_CLM_rationalize_MT(args):
    for seed in seed_variance_test:
        os.system(f"python main_llm.py --task_pretrained_name {args.explanation_model} --do_explain false --multi_explanation false --condition_on_explanations true --explanations_to_use t5-MT-multi-exp-pred-seed{seed} --labels_to_use preds_COMVE_{args.explanation_model}_MT.RA_seed{seed} "
                  f"--model_name sim.MT.RA --input_dropout .2 --explanation_dropout .4 "
                  f"--data_dir data/comve --gpu {args.gpu} --seed {seed} --num_train_epochs {args.num_train_epochs} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} --warmup_proportion .01 --dev_batch_size {args.dev_batch_size} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin} {suffix_addin} {test_file_addin} "
                  f"{stain_addin} {hypothesis_only_addin} {rand_addin} --rand_threshold {args.rand_threshold} {no_train_addin} {lora_addin} "
        )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_r", default=128, type=int, help="Rank for PEFT LORA")
    parser.add_argument("--lora_alpha", default=128, type=int, help="Alpha for PEFT LORA")
    parser.add_argument('--save_suffix', type=str, default=None,
                        help="Suffix to use if the outputs are to be written in a different file")
    parser.add_argument('--hypothesis_only', action='store_true',
                        help='Flag for using only the premise for training.')
    parser.add_argument('--random_label', action='store_true',
                        help='Flag for training with random labels.')
    parser.add_argument("--rand_threshold", default=3, type=int,
                        help='Threshold for percentage of instances to have a random wrong label.')
    parser.add_argument('--stain', action='store_true',
                        help='Flag for staining the dataset.')
    parser.add_argument('--no_train', action='store_true',
                        help='Flag for staining the dataset.')
    parser.add_argument("--stain_threshold", default=0.1, type=float,
                        help='Threshold for percentage of instances to have a stain.')
    parser.add_argument("--data_dir", default=None, type=str,
                        help='If other data directory should be used.')
    parser.add_argument("--gpu", default=0, type=int, help='')    
    parser.add_argument("--experiment", '-e', type=str, help='')
    parser.add_argument("--server_number", '-s', required=True, type=str, help='')
    parser.add_argument("--explanation_model", default='t5-base', type=str, help='HuggingFace transformer model')
    parser.add_argument("--prefinetuned_name", default=None, help='HuggingFace transformer model fine-tuned')
    parser.add_argument("--dev_batch_size", '-d', default=3, type=int,
                        help="ONLY FOR QA. Total batch size for training. Effective batch size is this times grad_accumulation_factor")
    parser.add_argument("--num_train_epochs", '-p', default=3, type=int,
                        help="Number of training epochs.")
    parser.add_argument("--train_batch_size", '-b', default=3, type=int, help="ONLY FOR QA. Total batch size for training. Effective batch size is this times grad_accumulation_factor")
    parser.add_argument('--grad_accumulation_factor', '-g', type=int, default=4, help="ONLY FOR QA. Number of updates steps to accumulate before performing a backward pass and step.")
    parser.add_argument('--small_data', action='store_true', help='Flag for using just a few datapoints for debugging purposes')
    parser.add_argument("--save_dir", default='', required=True, type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--cache_dir", default='', required=True, type=str,
                        help="Directory for cacheing pretrained models.")
    parser.add_argument("--test_file", default=None, type=str, help="If testing on a different test file.")
    args = parser.parse_args()
    save_dir = args.save_dir
    cache_dir = args.cache_dir
    small_data_addin, hypothesis_only_addin, rand_addin, stain_addin, no_train_addin, suffix_addin, test_file_addin = '', '', '', '', '', '', ''
    if args.small_data:
        small_data_addin = '-s -ss 64 ' # uses 64 points per split in main_llm.py
    if args.hypothesis_only:
        hypothesis_only_addin = "--hypothesis_only"
    if args.random_label:
        rand_addin = "--random_label"
    if args.stain:
        stain_addin = f"--stain --stain_threshold {args.stain_threshold}"
    if args.no_train:
        no_train_addin = '--do_train false --do_eval false'
    if args.save_suffix:
        suffix_addin = f'--save_suffix {args.save_suffix}'
    if args.test_file:
        test_file_addin = f'--test_file {args.test_file}'
    if args.prefinetuned_name:
        test_file_addin = f'--prefinetuned_name {args.prefinetuned_name} {test_file_addin}'
    lora_addin = f"--lora_r {args.lora_r} --lora_alpha {args.lora_alpha}"

    print("Starting experiment %s " % args.experiment)
    print("Using seeds ", seed_variance_test)
    print("Saving models in %s" % save_dir)

    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        if 'NLI' in args.experiment:
            data_dir = 'data/e-SNLI-data'
        elif 'QA' in args.experiment:
            data_dir = 'data/v1.0'
            if 'ecqa' in args.save_dir:
                data_dir = 'data/ECQA-Dataset'

    # --- begin QA --- #

    if args.experiment == 'QA.task':
        QA_task(args) 

    if args.experiment == 'QA.SIM.human':
        QA_SIM_human(args) 

    if args.experiment == 'QA.CLM.reason':
        QA_CLM_reason(args) 

    if args.experiment == 'QA.CLM.rationalize':
        QA_CLM_rationalize(args) 

    if args.experiment == 'QA.CLM.reason.MT':
        QA_CLM_reason_MT(args) 

    if args.experiment == 'QA.SIM.MT.RE':
        QA_SIM_CLM_reason_MT(args) 

    if args.experiment == 'QA.CLM.rationalize.MT':
        QA_CLM_rationalize_MT(args) 

    if args.experiment == 'QA.SIM.MT.RA':
        QA_SIM_CLM_rationalize_MT(args) 

    if args.experiment == 'QA.ST.RE':
        QA_ST_reason(args) 

    if args.experiment == 'QA.SIM.ST.RE':
        QA_SIM_ST_reason(args) 

    if args.experiment == 'QA.ST.RA':
        QA_ST_rationalize(args) 

    if args.experiment == 'QA.SIM.ST.RA':
        QA_SIM_ST_rationalize(args)
   

    # --- begin NLI --- #

    if args.experiment == 'NLI.task':
        NLI_task(args) 

    if args.experiment == 'NLI.SIM.human':
        NLI_SIM_human(args) 

    if args.experiment == 'NLI.CLM.reason':
        NLI_CLM_reason(args) 

    if args.experiment == 'NLI.CLM.rationalize':
        NLI_CLM_rationalize(args) 

    if args.experiment == 'NLI.CLM.reason.MT':
        NLI_CLM_reason_MT(args) 

    if args.experiment == 'NLI.SIM.MT.RE':
        NLI_SIM_CLM_reason_MT(args) 

    if args.experiment == 'NLI.CLM.rationalize.MT':
        NLI_CLM_rationalize_MT(args) 

    if args.experiment == 'NLI.SIM.MT.RA':
        NLI_SIM_CLM_rationalize_MT(args) 

    if args.experiment == 'NLI.ST.RE':
        NLI_ST_reason(args) 

    if args.experiment == 'NLI.ST.RA':
        NLI_ST_rationalize(args) 

    if args.experiment == 'NLI.SIM.ST.RE':
        NLI_SIM_ST_reason(args) 

    if args.experiment == 'NLI.SIM.ST.RA':
        NLI_SIM_ST_rationalize(args)

    # --- begin COMVE --- #

    if args.experiment == 'COMVE.task':
        COMVE_task(args)

    if args.experiment == 'COMVE.SIM.human':
        COMVE_SIM_human(args)

    if args.experiment == 'COMVE.CLM.reason':
        COMVE_CLM_reason(args)

    if args.experiment == 'COMVE.CLM.rationalize':
        COMVE_CLM_rationalize(args)

    if args.experiment == 'COMVE.CLM.reason.MT':
        COMVE_CLM_reason_MT(args)

    if args.experiment == 'COMVE.SIM.MT.RE':
        COMVE_SIM_CLM_reason_MT(args)

    if args.experiment == 'COMVE.CLM.rationalize.MT':
        COMVE_CLM_rationalize_MT(args)

    if args.experiment == 'COMVE.SIM.MT.RA':
        COMVE_SIM_CLM_rationalize_MT(args)

    if args.experiment == 'COMVE.ST.RE':
        COMVE_ST_reason(args)

    if args.experiment == 'COMVE.ST.RA':
        COMVE_ST_rationalize(args)

    if args.experiment == 'COMVE.SIM.ST.RE':
        COMVE_SIM_ST_reason(args)

    if args.experiment == 'COMVE.SIM.ST.RA':
        COMVE_SIM_ST_rationalize(args)

