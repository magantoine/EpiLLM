```
          ░▒▓████████▓▒░▒▓███████▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓██████████████▓▒░  
          ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░  
          ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░   
          ░▒▓██████▓▒░ ░▒▓███████▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░    
          ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░     
          ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░ 
          ░▒▓████████▓▒░▒▓█▓▒░      ░▒▓█▓▒░▒▓████████▓▒░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░
```

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
├── evaluationv
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

Or directly with `PipelineGenerator` :

```python
from episcape import PipelineGenerator


pip = PipelineGenerator(temperature=1, version=4)
df = pip.generate(10)
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


Using **Huggingface** LLMs using the model string from huggingface hub.

```python
from models import (GenerationArg, HF_LLM)


runarg = GenerationArg(
        temperature=1,
        topk=1,
        topp=1,
        max_new_token=1,
        presence_penalty=0.0,
        frequency_penalty=1.0,
        use_beam_search=False,
        logprobs=5,
        best_of=1,
        stop_seq=None,
        use_vllm=True
    )

meditron = HF_LLM("epfl-llm/meditron-7b",
                  device=DEVICE,
                  use_vllm=True,
                  arg=runarg)
```


Implementation in `notebooks/benchmark.ipynb` , 

We have a `testbench` function implemented that's called using that API :

```python

## benchmark string name
benchnames = ["AES7", "AES8"]

## huggingface hub string name
modnames = ["epfl-llm/meditron-7b", "meta-llama/Llama-2-7b-hf"]

## different generation arguments to try out
allgens = [
    GenerationArg(
        stop_seq="<|STOP|>",
        max_new_token=512,
        use_vllm=True
    ),
    GenerationArg(
        max_new_token=10,
        stop_seq="<|STOP|>",
        use_vllm=True
    )
]

## function to determine with answer the model has chosen form
## the complete generation
def search_ans(gen):
    if(type(gen) != str):
        gen = gen.outputs[0].text
    ans = [l for l in ["A", "B", "C", "D", "E"] if l in gen]
    if(len(ans) == 0):
        return "-1"
    return ans[0]

## format the question into the proper prompt understandable by the model
def prompt_template(q):
    return "system:{sys_prompt}\nuser:\nQuestion:{q}\nassistant:\nAnswer:"


### <----RUN TEST BENCH---->
res = testbench(benchnames=benchnames,
          modnames=modnames,
          runargs=allgens,
          prompt_template=prompt_template,
          search_ans=search_ans)

```


Result example :


![result dataframe](docs/static/fataframe_testbench.png)



#### Training API

To train a certain model we develop an API optimize for our infrastructure.

```shell
python training.py --datasets pmc \
            --save_dir checkpoints \
            --checkpoint epillm_tv0 \
            --base_checkpoint epfl-llm/meditron-7b \
            --lr 2e-5 \
            --eps 1e-5 \
            --wrmp 0.1 \
            --batch_size 4 \
            --n_train_epoch 1 \
```


- **datasets** : *Enter the names of the dataset to use* : `['pubmed', 'pmc_patients', 'mimic', 'all']`, default=`all`
- **save_dir** : *Directory in with you save the model checkpoints*, default=`checkpoints`
- **checkpoint** : *Name of the checkpoint folder*, default=`checkpoint`
- **base_checkpoint** : *Name of the base model*, default=`epfl-llm/meditron 7b`
- **lr** : *Learning rate*, default=`2e-5`
- **eps** : *Epsilon*, default=`1e-5`
- **wrmp** : *Warmup percentgage*, default=`0.1`
- **batch_size** : *Batch size*, default=`4`
- **n_train_epoch** : *Number of train epoch*, default=`1`





