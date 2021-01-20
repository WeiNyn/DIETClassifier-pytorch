from typing import List

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import BertForTokenClassification, BertPreTrainedModel, BertModel
from transformers.configuration_utils import PretrainedConfig

from os import path
import json


class DIETClassifierConfig(PretrainedConfig):
    def __init__(self, model: str, entities: List[str] = None, intents: List[str] = None):
        super().__init__()
        self.model = model
        self.entities = entities
        self.intents = intents
        self.hidden_dropout_prob = None
        self.hidden_size = None


class DIETClassifier(BertPreTrainedModel):
    def __init__(self, config: DIETClassifierConfig):
        """
        Create DIETClassifier model

        :param config: config for model
        """
        if path.exists(config.model):
            try:
                json_config = json.load(open(f"{config.model}/config.json", "r"))
            except Exception as ex:
                raise RuntimeError(f"Cannot load configuration fil from {config.model} by error: {ex}")

            try:
                checkpoint = torch.load(f"{config.model}/pytorch_model.bin")
            except Exception as ex:
                raise RuntimeError(f"Cannot load model from {config.model} by error: {ex}")

            pretrained_model = None
            config = PretrainedConfig.from_dict(json_config)
        else:
            pretrained_model = BertForTokenClassification.from_pretrained(config.model)
            checkpoint = None
            if config.intents is None or config.entities is None:
                raise ValueError(f"Using pretrained from transformers should specific entities and intents")
            pretrained_model.config.update({"model": config.model, "entities": config.entities, "intents": config.intents})
            config = pretrained_model.config

        super().__init__(config)

        self.entities_list = ["O"] + config.entities
        self.num_entities = len(self.entities_list)
        self.intents_list = config.intents
        self.num_intents = len(self.intents_list)

        self.bert = BertModel(config, add_pooling_layer=False) if not pretrained_model else pretrained_model.bert

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.entities_classifier = nn.Linear(config.hidden_size, self.num_entities)
        self.intents_classifier = nn.Linear(config.hidden_size, self.num_intents)

        self.init_weights()

        if not pretrained_model:
            try:
                self.load_state_dict(checkpoint)
            except Exception as ex:
                raise  RuntimeError(f"Cannot load state dict from checkpoint by error: {ex}")

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            intent_labels=None,
            entities_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        """
        training model if entities_labels and intent_labels are passed, else inference

        :param input_ids: embedding ids of tokens
        :param attention_mask: attention_mask
        :param token_type_ids: token_type_ids
        :param position_ids: position_ids (optional)
        :param head_mask: head_mask (optional)
        :param inputs_embeds: inputs_embeds (optional)
        :param intent_labels: labels of intent
        :param entities_labels: labels of entities
        :param output_attentions: return attention weight or not
        :param output_hidden_states: return hidden_states or not
        :param return_dict: return dictionary or not
        :return:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0][:, 1:]
        sequence_output = self.dropout(sequence_output)

        pooled_output = outputs[0][:, :1]
        pooled_output = self.dropout(pooled_output)

        entities_logits = self.entities_classifier(sequence_output)
        intent_logits = self.intents_classifier(pooled_output)

        entities_loss = None
        if entities_labels is not None:
            entities_loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask[:, 1:].reshape(-1) == 1
                active_logits = entities_logits.view(-1, self.num_entities)
                active_labels = torch.where(
                    active_loss, entities_labels.view(-1),
                    torch.tensor(entities_loss_fct.ignore_index).type_as(entities_labels)
                )
                entities_loss = entities_loss_fct(active_logits, active_labels)
            else:
                entities_loss = entities_loss_fct(entities_logits.view(-1, self.num_entities), entities_labels.view(-1))

        intent_loss = None
        if intent_labels is not None:
            if self.num_intents == 1:
                intent_loss_fct = MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_labels.view(-1))
            else:
                intent_loss_fct = CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intents), intent_labels.view(-1))

        if (entities_labels is not None) and (intent_labels is not None):
            loss = entities_loss * 0.1 + intent_loss * 0.9
        else:
            loss = None

        if self.training:
            return_dict = True

        if not return_dict:
            output = (loss,) + outputs[2:]
            return ((loss,) + output) if ((entities_loss is not None) and (intent_loss is not None)) else output

        return dict(
            entities_loss=entities_loss,
            intent_loss=intent_loss,
            loss=loss,
            logits=(entities_logits, intent_logits),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == '__main__':
    import os
    import sys

    sys.path.append(os.getcwd())

    from src.data_reader.data_reader import make_dataframe
    from src.data_reader.dataset import DIETClassifierDataset
    from transformers import AutoTokenizer

    files = ["dataset/nlu_QnA_converted.yml", "dataset/nlu_QnA_converted.yml"]
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

    df, entities_list, intents_list, synonym_dict = make_dataframe(files)
    dataset = DIETClassifierDataset(dataframe=df, tokenizer=tokenizer, entities=entities_list, intents=intents_list)

    config = DIETClassifierConfig(
        model="dslim/bert-base-NER",
        entities=entities_list,
        intents=intents_list
    )
    model = DIETClassifier(config=config)

    sentences = ["What if I'm late"]

    inputs = tokenizer(sentences, return_tensors="pt", padding="max_length", max_length=512)
    outputs = model(**{k: v for k, v in inputs.items()})

    print(outputs)
