from typing import Optional

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.integrations import TensorBoardCallback
from torch.utils.data import random_split
import torch


class DIETTrainer:
    def __init__(self, model, dataset, train_range: 0.95, output_dir: str = "results", num_train_epochs: int = 100, per_device_train_batch_size: int = 4,
                 per_device_eval_batch_size: int = 4, warmup_steps: int = 500, weight_decay: float = 0.01,
                 logging_dir: str = "logs", early_stopping_patience: int = 20, early_stopping_threshold: float = 1e-5):
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
        self.training_args = TrainingArguments(output_dir=output_dir,
                                               num_train_epochs=num_train_epochs,
                                               per_device_train_batch_size=per_device_train_batch_size,
                                               per_device_eval_batch_size=per_device_eval_batch_size,
                                               warmup_steps=warmup_steps,
                                               weight_decay=weight_decay,
                                               logging_dir=logging_dir,
                                               load_best_model_at_end=True,
                                               metric_for_best_model="loss",
                                               greater_is_better=False,
                                               evaluation_strategy="epoch",
                                               label_names=["entities_labels", "intent_labels"],
                                               save_total_limit=1)

        train_dataset, eval_dataset = random_split(dataset, [int(len(dataset)*train_range), len(dataset) - int(len(dataset)*train_range)], generator=torch.Generator().manual_seed(42))

        self.trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience, early_stopping_threshold=early_stopping_threshold), TensorBoardCallback()]
        )

    def train(self):
        self.trainer.train()


if __name__ == '__main__':
    import os
    import sys

    sys.path.append(os.getcwd())

    from src.DataReader.DataReader import make_dataframe
    from src.DataReader.dataset import DIETClassifierDataset
    from src.models.classifier import DIETClassifier, DIETClassifierConfig
    from transformers import AutoTokenizer

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    files = ["dataset/nlu_QnA_converted.yml", "dataset/nlu_QnA_converted.yml"]
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

    df, entities_list, intents_list = make_dataframe(files)
    entities_list = [entity for entity in entities_list if entity != "number"]
    print(f"ENTITIES_LIST: {entities_list}")
    dataset = DIETClassifierDataset(dataframe=df, tokenizer=tokenizer, entities=entities_list, intents=intents_list)

    config = DIETClassifierConfig(model="dslim/bert-base-NER", entities=entities_list, intents=intents_list)
    model = DIETClassifier(config=config)

    sentences = ["What if I'm late"]

    inputs = tokenizer(sentences, return_tensors="pt", padding="max_length", max_length=512)

    outputs = model(**{k: v for k, v in inputs.items()})

    trainer = DIETTrainer(model=model, dataset=dataset, train_range=0.95)

    trainer.train()
