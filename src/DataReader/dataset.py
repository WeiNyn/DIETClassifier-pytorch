from typing import List, Dict, Text, Any

import pandas as pd
import torch


class DIETClassifierDataset:
    def __init__(self, dataframe: pd.DataFrame, tokenizer, entities: List[str], intents: List[str]):
        """
        Dataset for DIETClassifier

        :param dataframe: dataframe contains ["example", "intent", "entities"] columns
        :param tokenizer: tokenizer from transformers
        :param entities: list of entities class names
        :param intents: list of intents class names
        """
        self.tokenizer = tokenizer
        self.entities = ["O"] + entities
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
            # offset_mapping=self.data["offset_mapping"][index],
            entities_labels=self.data["entities_labels"][index],
            intent_labels=self.data["intent_labels"][index]
        )

        return item


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.getcwd())

    from src.DataReader.DataReader import make_dataframe
    from transformers import AutoTokenizer

    files = ["Dataset/nlu_QnA_converted.yml", "Dataset/nlu_QnA_converted.yml"]
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

    df, entities_list, intents_list = make_dataframe(files)
    dataset = DIETClassifierDataset(dataframe=df, tokenizer=tokenizer, entities=entities_list, intents=intents_list)

    print(len(dataset))
    print(dataset[1])

