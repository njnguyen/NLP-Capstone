import argparse
import yaml

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, LineByLineTextDataset, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments


# read training configurations from YAML file
parser = argparse.ArgumentParser(
    description="Finetune XLM-R model on raw text corpora"
)
parser.add_argument('--config', type=str)
args = parser.parse_args()
config_dict = vars(args)
with open(args.config, 'r') as config_file:
    config_dict.update(yaml.load(config_file, Loader=yaml.Loader))


# load pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.hf_model, max_len=args.max_len)
model = AutoModelForMaskedLM.from_pretrained(args.hf_model)

# move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# prepare training data
train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=args.train_dataset_path,
    block_size=args.dataset_block_size,
)

# prepare validation data
val_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=args.val_dataset_path,
    block_size=args.dataset_block_size,
)

# initialize trainer class with training configs
training_args = TrainingArguments(
    seed=args.seed,
    data_seed=args.seed,
    log_level="info",
    num_train_epochs=args.training_epochs,
    learning_rate=float(args.learning_rate),
    per_device_train_batch_size=args.train_batch_size,
    evaluation_strategy=args.eval_strategy,
    per_device_eval_batch_size=args.eval_batch_size,
    eval_steps=args.eval_steps,
    save_steps=args.save_steps,
    save_total_limit=args.saved_checkpoints_limit,
    output_dir=args.checkpoints_directory,
    overwrite_output_dir=True,
    weight_decay=float(args.weight_decay),
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_ratio=float(args.warmup_ratio),
    warmup_steps=args.warmup_steps,
    auto_find_batch_size=args.auto_find_batch_size,
    group_by_length=args.group_by_length,
    gradient_checkpointing=args.gradient_checkpointing,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    fp16=args.fp16,
    fsdp=args.torch_distributed_training,
    full_determinism=args.full_determinism
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_masking_prob
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# start training
trainer.train()

# evaluate model
trainer.evaluate()