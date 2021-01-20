import json
import re
from typing import List, Dict, Tuple

import pandas as pd
import yaml

"""regex filter for preprocessing text data"""
NORMAL_REGEX = "\[[\w+\s*]+\]\([\w+\s*]+\)"
ENTITY_REGEX = "\[[\w+\s*]+\]"
ENTITY_NAME_REGEX = "\([\w+\s*]+\)"
SYNONYM_REGEX = "\[[\w+\s*]+\]\{.+\}"
DATA_REGEX = "\{.+\}"


def make_dataframe(files: List[str]) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, str]]:
    """
    Make data frame for DIETClassifier dataset from files list

    :param files: list of files location
    :return: tuple(dataframe, list of entities class name, list of intent class name, synonym dictionary)
    """

    data = []
    synonym_dict = {}
    for file in files:
        data += read_from_yaml(file=file)

    df = pd.DataFrame(columns=["example", "intent", "entities"])

    for intent in data:
        if not intent.get("intent", None):
            if intent.get("synonym", None):
                target_entity = intent["synonym"]

                entities_list_text = intent["examples"]
                if entities_list_text[:2] == "- ":
                    entities_list_text = entities_list_text[2:]
                if entities_list_text[-1:] == "\n":
                    entities_list_text = entities_list_text[:-1]

                synonym_entities_list = entities_list_text.split("\n- ")
                for entity in synonym_entities_list:
                    synonym_dict[entity] = target_entity

            continue

        intent_name = intent["intent"]
        examples_as_text = intent["examples"]

        if examples_as_text[:2] == "- ":
            examples_as_text = examples_as_text[2:]
        if examples_as_text[-1:] == "\n":
            examples_as_text = examples_as_text[:-1]

        examples = examples_as_text.split("\n- ")

        new_data = dict(
            intent=[intent_name for i in range(len(examples))],
            example=examples,
            entities=[None for i in range(len(examples))]
        )

        df = df.append(pd.DataFrame(data=new_data), ignore_index=True)

    df = get_entity(df=df)
    df, updated_synonym_dict = get_entity_with_synonym(df=df)

    synonym_dict.update(updated_synonym_dict)

    entities_list = []
    intents_list = []
    for _, row in df.iterrows():
        entity_data = row["entities"]
        if isinstance(entity_data, str):
            try:
                entity_data = json.loads(entity_data)
            except Exception as ex:
                raise RuntimeError(f"Cannot convert entity_data to json: {entity_data}")

        for entity in entity_data:
            entity_name = entity.get("entity_name")
            if entity_name not in entities_list:
                entities_list.append(entity_name)

        if row["intent"] not in intents_list:
            intents_list.append(row["intent"])

    return df, entities_list, intents_list, synonym_dict


def read_from_yaml(file: str) -> List[Dict[str, str]]:
    """
    Read data from .yml file

    :param file: file location (this data file need to follow the rasa nlu annotation format)
    :return: list(dict(text, any))
    """
    try:
        f = open(file, "r")
    except Exception as ex:
        raise RuntimeError(f"Cannot read file {file} with error:\t{ex}")

    data = yaml.load(f)["nlu"]
    return data


def get_entity(df: pd.DataFrame) -> pd.DataFrame:
    """
    extract entities in example sentences

    :param df: dataframe to process
    :return: precessed dataframe
    """
    for _, row in df.iterrows():
        entity_data = row["entities"]
        if not entity_data:
            entity_data = []

        while True:
            example = row["example"]
            x = re.search(NORMAL_REGEX, example)
            if x is None:
                break

            start, end = x.span()
            entity = x.group()

            entity_text = re.search(ENTITY_REGEX, entity).group()[1:-1]
            entity_name_text = re.search(ENTITY_NAME_REGEX, entity).group()[1:-1]

            row["example"] = example.replace(entity, entity_text)

            entity_data.append(dict(
                entity=entity_text,
                entity_name=entity_name_text,
                position=(start, end - (len(entity) - len(entity_text)))
            ))

        row["entities"] = entity_data

    return df


def get_entity_with_synonym(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Extract entities with synonym in dataframe.

    :param df: dataframe to process
    :return: tuple(processed dataframe, synonym dictionary)
    """
    synonym_dict = {}
    for _, row in df.iterrows():
        entity_data = row["entities"]
        if not entity_data:
            entity_data = []

        if isinstance(entity_data, str):
            try:
                entity_data = json.loads(entity_data)
            except Exception as ex:
                raise RuntimeError(f"Cannot convert entity_data to json: {entity_data}")

        while True:
            example = row["example"]
            x = re.search(SYNONYM_REGEX, example)
            if x is None:
                break

            start, end = x.span()
            entity = x.group()

            entity_text = re.search(ENTITY_REGEX, entity).group()[1:-1]
            synonym_text = re.search(DATA_REGEX, entity).group()

            try:
                synonym_data = json.loads(synonym_text)
            except Exception as ex:
                raise ValueError(f"Synonym json is incorrect: {synonym_text}")

            entity_name_text = synonym_data.get("entity", None)
            synonym_value = synonym_data.get("value", None)

            if entity_name_text is None or synonym_value is None:
                raise ValueError(f"synonym data should have 'entity' and 'value' attributes")

            row["example"] = example.replace(entity, entity_text)

            entity_data.append(dict(
                entity=entity_text,
                entity_name=entity_name_text,
                position=(start, end - (len(entity) - len(entity_text))),
                synonym=synonym_value
            ))

            synonym_dict[synonym_value] = entity_text

        row["entities"] = entity_data

    return df, synonym_dict


if __name__ == '__main__':
    import os
    import sys

    sys.path.append(os.getcwd())

    files = ["dataset/nlu_QnA_converted.yml", "dataset/nlu_QnA_converted.yml"]

    df, entities_list, intents_list, synonym_dict = make_dataframe(files)
    print(df.head(20))
    print(entities_list)
    print(intents_list)
    print(synonym_dict)
