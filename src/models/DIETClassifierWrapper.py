from typing import Union, Dict, List, Any, Tuple

import torch
import yaml
from transformers import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig

from src.models.DIETClassifier import DIETClassifier


class DIETClassifierWrapper:
    def __init__(self, config: Union[Dict[str, str], str]):
        if isinstance(config, str):
            try:
                f = open(config, "r")
            except Exception as ex:
                raise RuntimeError(f"Cannot read config file from {config}: {ex}")
            config = yaml.load(f)

        self.util_config = config.get("util", None)

        model_config_dict = config.get("model", None)
        if not model_config_dict:
            raise ValueError(f"Config file should have 'model' attribute")

        self.dataset_config = model_config_dict

        model_config_attributes = ["model", "intents", "entities"]
        model_config_dict = {k: v for k, v in model_config_dict if k in model_config_attributes}

        self.intents = model_config_dict["intents"]
        self.entities = model_config_dict["entities"]

        self.model_config = PretrainedConfig.from_dict(model_config_dict)

        training_config_dict = config.get("training", None)
        if not training_config_dict:
            raise ValueError(f"Config file should have 'training' attribute")

        self.training_config = training_config_dict
        self.tokenizer = AutoTokenizer(self.model_config.model)
        self.model = DIETClassifier(config=self.model_config)

        if model_config_dict["device"] is not None:
            self.device = torch.device(model_config_dict["device"]) if torch.cuda.is_available() else torch.device(
                "cpu")

        self.model.to(self.device)

        self.softmax = torch.nn.Softmax(dim=-1)

    def tokenize(self, sentences) -> Tuple[Dict[str, Any], List[List[Tuple[int, int]]]]:
        inputs = self.tokenizer(sentences, return_tensors="pt", return_attention_mask=True, return_token_type_ids=True,
                                return_offsets_mapping=True,
                                padding=True, truncation=True)

        offset_mapping = inputs["offset_mapping"]
        inputs = {k: v.to(self.device) for k, v in inputs.items() if k != offset_mapping}

        return inputs, offset_mapping

    def convert_intent_logits(self, intent_logits: torch.tensor) -> List[Dict[str, float]]:
        softmax_intents = self.softmax(intent_logits)

        predicted_intents = []

        for sentence in softmax_intents:
            sentence = sentence[0]

            sorted_sentence = sentence.clone()
            sorted_sentence, _ = torch.sort(sorted_sentence)

            if sorted_sentence[0] >= self.util_config["intent_threshold"] and (
                    sorted_sentence[0] - sorted_sentence[1]) >= self.util_config["ambiguous_threshold"]:
                max_probability = torch.argmax(sentence)
            else:
                max_probability = -1

            predicted_intents.append({
                "intent": None if max_probability == -1 else self.intents[max_probability],
                "intent_ranking": {
                    intent_name: probability for intent_name, probability in zip(self.intents, sentence)
                }
            })

        return predicted_intents

    def convert_entities_logits(self, entities_logits: torch.tensor, offset_mapping: torch.tensor) -> List[
        List[Dict[str, Any]]]:
        softmax_entities = self.softmax(entities_logits)

        predicted_entities = []

        for sentence, offset in zip(softmax_entities, offset_mapping):
            predicted_entities.append([])
            latest_entity = None
            for word, token_offset in zip(sentence, offset[1:]):
                max_probability = torch.argmax(word)
                if word[max_probability] >= self.util_config["entities_threshold"]:
                    if self.entities[max_probability] != latest_entity:
                        predicted_entities[-1].append({
                            "entity_name": self.entities[max_probability],
                            "start": token_offset[0],
                            "end": token_offset[1]
                        })
                    else:
                        predicted_entities[-1][-1]["end"] = token_offset[1]
                else:
                    latest_entity = None

        return predicted_entities

    def predict(self, sentences: List[str]) -> Dict[str, Any]:
        inputs, offset_mapping = self.tokenize(sentences=sentences)
        outputs = self.model(**inputs)
        logits = outputs["logits"]
        predicted_intents = self.convert_intent_logits(intent_logits=logits[1])
        predicted_entities = self.convert_entities_logits(entities_logits=logits[0], offset_mapping=offset_mapping)
        predicted_outputs = []
        for intent_sentence, entities_sentence in zip(predicted_intents, predicted_entities):
            predicted_outputs.append({})
            predicted_outputs[-1].update(intent_sentence)
            predicted_outputs[-1].update({"entities": entities_sentence})

        return predicted_outputs
