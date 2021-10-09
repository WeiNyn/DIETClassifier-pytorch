# DIETClassifier - Pytorch


[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

DIETClassifier stand for Dual Intent Entity from Transformers which can be used to do intent classification and entities recognition at the same time.

  - Using Huggingface Transformers's BERT architect
  - Wrapped by python, with various implemented functions (reads dataset from .yml, builds and trains model, gives dictionary ouput)

# Requirements

* [transformers] - Library for using transformers models in nlp task
* [pytorch] - Framework for deep learning task in python
* [fastapi] - Backend building framework

You can also install all requirement packages by:
```sh
git clone https://github.com/WeiNyn/DIETClassifier-pytorch.git
cd DIETClassifier-pytorch/
pip install -r requirements.txt
```

### Demo

You can use demo server to create a server that receive text message and predict intent, entities:

- Download pretrained model from [this link](https://drive.google.com/drive/folders/1cAucUHO0FP_I-_atSpbyRwKEiflPPN7v?usp=sharing)
- extract "latest_model" to "DIETClassifier-pytorch/"
- run
```sh
uvicorn demo.server:app
```

### Configuration

All project configurations stored in [config.yml] file
```yaml
model:
    model: latest_model
    tokenizer: latest_model
    dataset_folder: dataset
    exclude_file: null
    entities:
        - working_type
        - shift_type
    intents:
        - WorkTimesBreaches
        - WorkingTimeBreachDiscipline
        - HolidaysOff
        - AnnualLeaveApplicationProcess
        - SetWorkingType
        - TemporarySetWorkingType
        - WorkingHours
        - WorkingDay
        - BreakTime
        - Pregnant
        - AttendanceRecord
        - SelectShiftType
        - LaborContract
        - Recruitment
        - SickLeave
        - UnpaidLeave
        - PaidLeaveForFamilyEvent
        - UnusedAnnualLeave
        - RegulatedAnnualLeave
        - rating
    device: cuda
training:
    train_range: 0.95
    num_train_epochs: 100
    per_device_train_batch_size: 4
    per_device_eval_batch_size: 4
    warmup_steps: 500
    weight_decay: 0.01
    logging_dir: logs/
    early_stopping_patience: 10
    early_stopping_threshold: 0.0001
    output_dir: results/
util:
    intent_threshold: 0.7
    entities_threshold: 0.5
    ambiguous_threshold: 0.2
```

| Attribute | Explain |
| --------- | ------- |
| model | name of transformers pretrained model or path to local model |
| tokenizer | name of transformers pretrained tokenizer or path to local tokenizer |
| dataset_folder | folder that container dataset files, using rasa nlu format |
| exclude_file | files in folder that will not be used to train |
| entities | list of entities |
| intents | list of intents |
| synonym | synonym list for synonym entities |
| device | device to use ("cpu", "cuda", "cuda:0", etc) |
| train_range | range to split dataset into train and valid set |
| num_train_epochs | number of training epochs |
| per_device_train/eval_batch_size | batch size when train/eval |
| logging_dir | directory to save log file (tensorboard supported) |
| early_stopping_patience/threshold | hyper parameters for early stopping training |
| output_dir | directory to save model while training |

### Usage

You can use DIETClassifierWrapper for loading, training, predicting in python code:
```python
from src.models.wrapper import DIETClassifierWrapper

config_file = "src/config.yml"
wrapper = DIETClassifierWrapper(config=config_file)

#predict
wrapper.predict(["How to check attendance?"])

#train
#after training, wrapper will load best model automatically
wrapper.train_model(save_folder="test_model")
```

You can also use DIETClassifier in src.models.classifier as huggingface transformers model
```python
from src.models.classifier import DIETClassifier, DIETClassifierConfig

config = DIETClassifierConfig(model="BERT-base-uncased", 
                              intents=[str(i) for i in range(10)], 
                              entities=[str(i) for i in range(5)])

model = DIETClassifier(config=config)

```

### Notice

* This DIETClassifier using BERT base as the base architect, if you want to change to RoBerta, ALBert, etc. You need to modify the DIETClassifier Class.
* You can also use any BERT base pretrained from Huggingface transformers for creating and fine tune yourself
* Please read the source code to understand how the dataset be created in case that you want to make dataset in another file format.
* If you get the error: AttributeError: """'NoneType' object has no attribute 'detach'""", please check the issue #5

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
