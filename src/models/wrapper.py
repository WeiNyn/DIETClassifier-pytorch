from os import path, listdir
from typing import Union, Dict, List, Any, Tuple

import torch
import yaml
from transformers import BertTokenizerFast

import os
import sys

sys.path.append(os.getcwd())

from src.models.classifier import DIETClassifier, DIETClassifierConfig
from src.models.trainer import DIETTrainer
from src.data_reader.dataset import DIETClassifierDataset
from src.data_reader.data_reader import make_dataframe


class DIETClassifierWrapper:
    """Wrapper for DIETClassifier."""
    def __init__(self, config: Union[Dict[str, Dict[str, Any]], str]):
        """
        Create wrapper with configuration.

        :param config: config in dictionary format or path to config file (.yml)
        """
        if isinstance(config, str):
            try:
                f = open(config, "r")
            except Exception as ex:
                raise RuntimeError(f"Cannot read config file from {config}: {ex}")
            self.config_file_path = config
            config = yaml.load(f)

        self.config = config
        self.util_config = config.get("util", None)

        model_config_dict = config.get("model", None)
        if not model_config_dict:
            raise ValueError(f"Config file should have 'model' attribute")

        self.dataset_config = model_config_dict

        if model_config_dict["device"] is not None:
            self.device = torch.device(model_config_dict["device"]) if torch.cuda.is_available() else torch.device(
                "cpu")

        model_config_attributes = ["model", "intents", "entities"]
        # model_config_dict = {k: v for k, v in model_config_dict.items() if k in model_config_attributes}

        self.intents = model_config_dict["intents"]
        self.entities = ["O"] + model_config_dict["entities"]

        self.model_config = DIETClassifierConfig(**{k: v for k, v in model_config_dict.items() if k in model_config_attributes})

        training_config_dict = config.get("training", None)
        if not training_config_dict:
            raise ValueError(f"Config file should have 'training' attribute")

        self.training_config = training_config_dict
        self.tokenizer = BertTokenizerFast.from_pretrained(model_config_dict["tokenizer"])
        self.model = DIETClassifier(config=self.model_config)

        self.model.to(self.device)

        self.softmax = torch.nn.Softmax(dim=-1)

        self.synonym_dict = {} if not model_config_dict.get("synonym") else model_config_dict["synonym"]

    def tokenize(self, sentences) -> Tuple[Dict[str, Any], List[List[Tuple[int, int]]]]:
        """
        Tokenize sentences using tokenizer.
        :param sentences: list of sentences
        :return: tuple(tokenized sentences, offset_mapping for sentences)
        """
        inputs = self.tokenizer(sentences, return_tensors="pt", return_attention_mask=True, return_token_type_ids=True,
                                return_offsets_mapping=True,
                                padding=True, truncation=True)

        offset_mapping = inputs["offset_mapping"]
        inputs = {k: v.to(self.device) for k, v in inputs.items() if k != "offset_mapping"}

        return inputs, offset_mapping

    def convert_intent_logits(self, intent_logits: torch.tensor) -> List[Dict[str, float]]:
        """
        Convert logits from model to predicted intent,

        :param intent_logits: output from model
        :return: dictionary of predicted intent
        """
        softmax_intents = self.softmax(intent_logits)

        predicted_intents = []

        for sentence in softmax_intents:
            sentence = sentence[0]

            sorted_sentence = sentence.clone()
            sorted_sentence, _ = torch.sort(sorted_sentence)

            if sorted_sentence[-1] >= self.util_config["intent_threshold"] and (
                    sorted_sentence[-1] - sorted_sentence[-2]) >= self.util_config["ambiguous_threshold"]:
                max_probability = torch.argmax(sentence)
            else:
                max_probability = -1

            predicted_intents.append({
                "intent": None if max_probability == -1 else self.intents[max_probability],
                "intent_ranking": {
                    intent_name: probability.item() for intent_name, probability in zip(self.intents, sentence)
                }
            })

        return predicted_intents

    def convert_entities_logits(self, entities_logits: torch.tensor, offset_mapping: torch.tensor) -> List[
        List[Dict[str, Any]]]:
        """
        Convert logits to predicted entities

        :param entities_logits: entities logits from model
        :param offset_mapping: offset mapping for sentences
        :return: list of predicted entities
        """
        softmax_entities = self.softmax(entities_logits)

        predicted_entities = []

        for sentence, offset in zip(softmax_entities, offset_mapping):
            predicted_entities.append([])
            latest_entity = None
            for word, token_offset in zip(sentence, offset[1:]):
                max_probability = torch.argmax(word)
                if word[max_probability] >= self.util_config["entities_threshold"] and max_probability != 0:
                    if self.entities[max_probability] != latest_entity:
                        latest_entity = self.entities[max_probability]
                        predicted_entities[-1].append({
                            "entity_name": self.entities[max_probability],
                            "start": token_offset[0].item(),
                            "end": token_offset[1].item()
                        })
                    else:
                        predicted_entities[-1][-1]["end"] = token_offset[1].item()
                else:
                    latest_entity = None

        return predicted_entities

    def predict(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Predict intent and entities from sentences.

        :param sentences: list of sentences
        :return: list of prediction
        """
        inputs, offset_mapping = self.tokenize(sentences=sentences)
        outputs = self.model(**inputs)
        logits = outputs["logits"]
        predicted_intents = self.convert_intent_logits(intent_logits=logits[1])
        predicted_entities = self.convert_entities_logits(entities_logits=logits[0], offset_mapping=offset_mapping)
        predicted_outputs = []
        for sentence, intent_sentence, entities_sentence in zip(sentences, predicted_intents, predicted_entities):
            predicted_outputs.append({})
            predicted_outputs[-1].update(intent_sentence)
            predicted_outputs[-1].update({"entities": entities_sentence})
            for entity in predicted_outputs[-1]["entities"]:
                entity["text"] = sentence[entity["start"]: entity["end"]]

                if self.synonym_dict.get(entity["text"], None):
                    entity["original_text"] = entity["text"]
                    entity["text"] = self.synonym_dict[entity["text"]]

            predicted_outputs[-1]["text"] = sentence

        return predicted_outputs

    def save_pretrained(self, directory: str):
        """
        Save model and tokenizer to directory

        :param directory: path to save folder
        :return: None
        """
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)

        config_file_path = "config.yml" if not self.config_file_path else self.config_file_path

        try:
            f = open(config_file_path, "w")
            yaml.dump(self.config, f, sort_keys=False)
            f.close()
        except Exception as ex:
            raise RuntimeError(f"Cannot save config to {config_file_path} by error: {ex}")

    def train_model(self, save_folder: str = "latest_model"):
        """
        Create trainer, train and save best model to save_folder
        :param save_folder: path to save folder
        :return: None
        """
        dataset_folder = self.dataset_config["dataset_folder"]
        if not path.exists(dataset_folder):
            raise ValueError(f"Folder {dataset_folder} is not exists")

        files_list = [path.join(dataset_folder, f) for f in listdir(dataset_folder) if path.isfile(path.join(dataset_folder, f)) and f.endswith(".yml")]

        df, _, _, synonym_dict = make_dataframe(files=files_list)

        self.synonym_dict.update(synonym_dict)

        dataset = DIETClassifierDataset(dataframe=df, tokenizer=self.tokenizer, entities=self.entities[1:], intents=self.intents)

        trainer = DIETTrainer(model=self.model, dataset=dataset,
                              train_range=self.training_config["train_range"],
                              num_train_epochs=self.training_config["num_train_epochs"],
                              per_device_train_batch_size=self.training_config["per_device_train_batch_size"],
                              per_device_eval_batch_size=self.training_config["per_device_eval_batch_size"],
                              warmup_steps=self.training_config["warmup_steps"],
                              weight_decay=self.training_config["weight_decay"],
                              logging_dir=self.training_config["logging_dir"],
                              early_stopping_patience=self.training_config["early_stopping_patience"],
                              early_stopping_threshold=self.training_config["early_stopping_threshold"],
                              output_dir=self.training_config["output_dir"])

        trainer.train()

        self.save_pretrained(directory=save_folder)


if __name__ == "__main__":
    config_file = "src/config.yml"

    wrapper = DIETClassifierWrapper(config=config_file)

    print(wrapper.predict(["I work on office hours"]))

    wrapper.train_model()

    print(wrapper.predict(["What is the average working hours"]))


