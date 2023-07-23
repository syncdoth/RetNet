from dataclasses import dataclass

from transformers import (Trainer, TrainingArguments, AutoTokenizer, HfArgumentParser,
                          DataCollatorForLanguageModeling)
from datasets import load_dataset

from retnet.modeling_retnet import RetNetModelWithLMHead
from retnet.configuration_retnet import load_config_from_yaml


@dataclass
class MyArgs:
    model_size: str = '300m'
    dataset_name: str = 'sst2'
    text_col: str = 'sentence'
    max_length: int = 256


def main():
    parser = HfArgumentParser((TrainingArguments, MyArgs))
    train_args, args = parser.parse_args_into_dataclasses()

    train_dataset = load_dataset(args.dataset_name, split="train")
    eval_dataset = load_dataset(args.dataset_name, split="validation")

    config = load_config_from_yaml(f"configs/retnet-{args.model_size}.yml")
    model = RetNetModelWithLMHead(config)

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.model_max_length = 16384
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.unk_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token

    def tokenize_datset(example):
        input_ids = tokenizer(example[args.text_col],
                              truncation=True,
                              max_length=args.max_length,
                              return_tensors='pt').input_ids[0]
        return {'input_ids': input_ids}

    train_dataset = train_dataset.map(tokenize_datset, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(tokenize_datset, remove_columns=eval_dataset.column_names)

    trainer = Trainer(model=model,
                      args=train_args,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      tokenizer=tokenizer,
                      data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))

    if train_args.do_train:
        trainer.train()
        trainer.save_model()
    if train_args.do_eval:
        trainer.evaluate()


if __name__ == "__main__":
    main()
