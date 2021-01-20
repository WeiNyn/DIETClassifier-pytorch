import json
from typing import List, Dict, Text, Any, Union

import pandas as pd
import torch


class DIETClassifierDataset:
    def __init__(self, dataframe: pd.DataFrame, tokenizer, entities: List[str], intents: List[str]):
        """
        dataset for DIETClassifier

        :param dataframe: dataframe contains ["example", "intent", "entities"] columns
        :param tokenizer: tokenizer from transformers
        :param entities: list of entities class names
        :param intents: list of intents class names
        """
        dataframe = dataframe[dataframe["intent"].isin(intents)]

        self.entities = ["O"] + entities
        dataframe["entities"] = dataframe["entities"].apply(self._remove_entities)

        self.tokenizer = tokenizer
        self.num_entities = len(self.entities)
        self.intents = intents
        self.num_intents = len(intents)

        sentences = dict(
            sentence=[],
            entities=[],
            intent=[]
        )

        for _, row in dataframe.iterrows():
            sentences["sentence"].append(row["example"])
            sentences["entities"].append(row["entities"])
            sentences["intent"].append(row["intent"])

        sentences.update(tokenizer(sentences["sentence"], return_tensors="pt", return_offsets_mapping=True, padding="max_length", truncation=True, max_length=512))

        sentences["entities_labels"] = []

        for index in range(len(sentences["sentence"])):
            entities_labels = []
            for offset in sentences["offset_mapping"][index][1:]:
                is_label = False
                if not (offset[0] == 0 and offset[1] == 0):
                    for entity in sentences["entities"][index]:
                        if entity["position"][0] <= offset[0] and entity["position"][1] >= offset[1]:
                            entities_labels.append(self.entities.index(entity["entity_name"]))
                            is_label = True
                if not is_label:
                    entities_labels.append(self.entities.index("O"))

            sentences["entities_labels"].append(entities_labels)

        sentences["entities_labels"] = torch.tensor(sentences["entities_labels"])
        sentences["intent_labels"] = torch.tensor([self.intents.index(intent) for intent in sentences["intent"]])

        self.data = sentences

    def __len__(self) -> int:
        return len(self.data["sentence"])

    def __getitem__(self, index) -> Dict[Text, Any]:
        item = dict(
            input_ids=self.data["input_ids"][index],
            token_type_ids=self.data["token_type_ids"][index],
            attention_mask=self.data["attention_mask"][index],
            entities_labels=self.data["entities_labels"][index],
            intent_labels=self.data["intent_labels"][index]
        )

        return item

    def _remove_entities(self, entities_list: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if isinstance(entities_list, str):
            try:
                entities_list = json.loads(entities_list)
            except Exception as ex:
                raise RuntimeError(f"Cannot convert entity {entities_list} by error: {ex}")

        entities_list = [entity for entity in entities_list if entity["entity_name"] in self.entities]

        return entities_list


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.getcwd())

    from src.data_reader.data_reader import make_dataframe
    from transformers import AutoTokenizer

    files = ["dataset/nlu_QnA_converted.yml", "dataset/nlu_QnA_converted.yml"]
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

    df, entities_list, intents_list, synonym_dict = make_dataframe(files)
    dataset = DIETClassifierDataset(dataframe=df, tokenizer=tokenizer, entities=entities_list, intents=intents_list)

    print(len(dataset))
    print(dataset[120])
    print(dataset.data["entities"][120])
    print(dataset.data["offset_mapping"][120])

