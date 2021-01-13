from typing import List

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertForTokenClassification, BertPretrainedModel


class DIETClassifier(BertPretrainedModel):
    def __init__(self, model: str, entities: List[str], intents: List[str]):
        pretrained_model = BertForTokenClassification.from_pretrained(model)
        config = pretrained_model.config
        super().__init__(config)
        self.entities_list = ["O"] + entities
        self.num_entities = len(self.entities_list)
        self.intents_list = intents
        self.num_intents = len(self.intents_list)

        self.bert = pretrained_model.bert
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.entities_classifier = nn.Linear(config.hidden_size, self.num_entities)
        self.intents_classifier = nn.Linear(config.hidden_size, self.num_intents)

        self.init_weights()

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
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
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

        sequence_output = outputs[0][1:]
        sequence_output = self.dropout(sequence_output)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        entities_logits = self.classifier(sequence_output)
        intent_logits = self.classifier(pooled_output)

        entities_loss = None
        if entities_labels is not None:
            entities_loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = entities_logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, entities_labels.view(-1),
                    torch.tensor(entities_loss_fct.ignore_index).type_as(entities_labels)
                )
                entities_loss = entities_loss_fct(active_logits, active_labels)
            else:
                entities_loss = entities_loss_fct(entities_logits.view(-1, self.num_labels), entities_labels.view(-1))

        intent_loss = None
        if intent_labels is not None:
            if self.num_intents == 1:
                intent_loss_fct = MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_labels.view(-1))
            else:
                intent_loss_fct = CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intents), intent_labels.view(-1))

        loss = entities_loss + intent_loss

        if not return_dict:
            output = (entities_logits, intent_logits,) + outputs[2:]
            return ((loss,) + output) if ((entities_loss is not None) and (intent_loss is not None)) else output

        return dict(
            loss=loss,
            logits=(entities_logits, intent_logits),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
