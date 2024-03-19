# EpiLLM


```shell
.
├── README.md
├── episcape ## GENERATION OF SDeHR
│   ├── __init__.py
│   ├── gen_utils.py
│   ├── patient_gen.py
│   ├── patient_gen_utils
│   ├── pipeline_gen.py
│   └── sdehr_gen.py
├── evaluate ## EVALUATION FOLDER
│   ├── __init__.py
│   ├── evaluate_utils.py
│   └── evaluation.py
├── misc ## MISC
│   └── utils.py
├── models ## MODELS
│   ├── __init__.py
│   ├── base_model.py
│   ├── baseline.py
│   ├── epiLLM.py
│   └── model_utils.py
├── scrape-research ##
└── usecase.ipynb
```


## Usecases 

#### Evaluation

```python
from evaluate import (GeneralExperiment,
                      SpecializedExperiment)
```

#### EpiScape

```python
from episcape import (PatientGenerator,
                      SDeHRGenerator,
                      PipelineGenerator)

patient_gen = PatientGenerator()
sdehr_gen = SDeHRGenerator()

test = patient_gen.generate(10, return_type=pd.DataFrame)
```


#### Models

```python
from models import (EpiLLM,
                    BaseModel,
                    Baseline)
```

