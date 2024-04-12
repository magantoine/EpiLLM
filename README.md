# EpiLLM


```shell
├── README.md
├── docs
│   ├── EPIO.csv
│   ├── benchmarks
│   │   ├── mcq40
│   │   └── self_assessment
│   ├── drugs
│   ├── mimic-iv-note-deidentified-free-text-clinical-notes-2
│   ├── pmc_patiens_fil.hf
│   └── results
├── episcape
│   ├── patient_gen_utils
│   │   ├── distributions
│   │   ├── patients_distribution.py
│   │   └── prompt_template.py
│   ├── patient_gen.py
│   ├── pipeline_gen.py
│   └── sdehr_gen.py
├── evaluation
│   ├── classes.py
│   ├── evaluate_utils.py
│   └── offline_metrics.py
├── misc
│   └── utils.py
├── models
│   ├── baseline.py
│   ├── llms.py
│   ├── model_utils.py
│   └── spec_datasets.py
├── notebooks
│   ├── benchmark.ipynb
│   ├── cooccurence_matrix.ipynb
│   ├── generations_study.ipynb
│   ├── ontology_study.ipynb
│   ├── process_mcq.ipynb
│   ├── test_model.ipynb
│   └── training.ipynb
├── make.py
├── setup.sh
├── usecase.ipynb
├── decorators.py
├── dependencies.sh
└── watcher.py
```


## Setup and workflow


```bash
./setup.sh
```

The `setup.sh` file setup the dependencies in the directory and will create the required `.env` file. For this purpose you will need an **OpenAI** API Key, if you don't have it for now, you can simply skip this question and input it later in the `.env` file manually or by rerunning the command.


```bash
./dependencies.sh
```

We developped a module handling method. Upon adding functions to expose to other modules or new files etc. you can simply run this command to make the dependencies.

You can avoid this work by running in the background the watcher :

```bash
python watcher.py
```

This watcher also serves as a **code checker** and will tell you when something goes wrong in your codebase.

#### EpiScape

```python
patient_gen = PatientGenerator()

test = patient_gen.generate(10)

sdehr_gen = SDeHRGenerator(temperature=1, version=2)
test["sdehr"] = [
    sdehr_gen.generate_sdehr(test_patient)
    for test_patient in test.to_dict(orient="records")
]
```


#### Evaluation

```python
from models import OpenAIGPT
from evaluation import MCQBenchmark


aes8 = MCQBenchmark(
    "docs/benchmarks/self_assessment/aes8_processed.json",
    prompt_template,
)
gpt = OpenAIGPT(model="gpt-3.5-turbo")

gpt_res = aes8.assess(gpt)
print(sum([res["answer"] == res["prediction"][0] for res in gpt_res]) * 100 / len(gpt_res), "% accuracy for GPT-3.5")
```


```python
## generation length
test["sdehr_LEN"] = genlength(test["sdehr"].values)
test["sdehr_LEN"].mean()

## mean embsim
embSim(list(test["sdehr"].values), format="flat").mean()

## self-BLEU
self_BLEU(list(test["sdehr"].values))["bleu"]
```



