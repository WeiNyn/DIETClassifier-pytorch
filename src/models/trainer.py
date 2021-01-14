from transformers import Trainer, TrainingArguments
from torch.utils.data import random_split
import torch


class DIETTrainer:
    def __init__(self, model, dataset, train_range: 0.95, output_dir: str = "results", num_train_epochs: int = 100, per_device_train_batch_size: int = 4,
                 per_device_eval_batch_size: int = 4, warmup_steps: int = 500, weight_decay: float = 0.01,
                 logging_dir: str = "logs"):
        """
        Create DIETTrainer class

        :param model: model to train
        :param dataset: dataset (including train and eval)
        :param train_range: percentage of training dataset
        :param output_dir: model output directory
        :param num_train_epochs: number of training epochs
        :param per_device_train_batch_size: batch_size of training stage
        :param per_device_eval_batch_size: batch_size of evaluating stage
        :param warmup_steps: warmup steps
        :param weight_decay: weight decay
        :param logging_dir: logging directory
        """
        self.training_args = TrainingArguments(output_dir=output_dir, num_train_epochs=num_train_epochs,
                                          per_device_train_batch_size=per_device_train_batch_size,
                                          per_device_eval_batch_size=per_device_eval_batch_size,
                                          warmup_steps=warmup_steps, weight_decay=weight_decay, logging_dir=logging_dir)

        train_dataset, eval_dataset = random_split(dataset, [int(len(dataset)*train_range), len(dataset) - int(len(dataset)*train_range)], generator=torch.Generator().manual_seed(42))

        self.trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

    def train(self):
        self.trainer.train()


if __name__ == '__main__':
    import os
    import sys

    sys.path.append(os.getcwd())

    from src.DataReader.DataReader import make_dataframe
    from src.DataReader.dataset import DIETClassifierDataset
    from src.models.DIETClassifier import DIETClassifier
    from transformers import AutoTokenizer
    from transformers.configuration_utils import PretrainedConfig

    files = ["Dataset/nlu_QnA_converted.yml", "Dataset/nlu_QnA_converted.yml"]
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

    df, entities_list, intents_list = make_dataframe(files)
    dataset = DIETClassifierDataset(dataframe=df, tokenizer=tokenizer, entities=entities_list, intents=intents_list)

    config = PretrainedConfig.from_dict(dict(model="dslim/bert-base-NER", entities=entities_list, intents=intents_list))
    model = DIETClassifier(config=config)

    sentences = ["What if I'm late"]

    inputs = tokenizer(sentences, return_tensors="pt", padding="max_length", max_length=512)
    outputs = model(**{k: v for k, v in inputs.items()})

    trainer = DIETTrainer(model=model, dataset=dataset, train_range=0.95)

    trainer.train()
