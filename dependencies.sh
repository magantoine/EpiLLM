#!/bin/sh

rm episcape/__init__.py
touch episcape/__init__.py
echo "from .sdehr_gen import *
from .patient_gen import *
from .pipeline_gen import *
" >> episcape/__init__.py
# from .patient_gen_utils import *


rm evaluation/__init__.py
touch evaluation/__init__.py
echo "from .evaluate_utils import  *
from .classes import *
from .offline_metrics import *" >> evaluation/__init__.py

rm misc/__init__.py
touch misc/__init__.py
echo "from .utils import *" >> misc/__init__.py

rm models/__init__.py
touch models/__init__.py
echo "from .llms import *
from .model_utils import *
from .spec_datasets import *" >> models/__init__.py

rm episcape/patient_gen_utils/__init__.py
touch episcape/patient_gen_utils/__init__.py
echo "from .patients_distribution import * 
from .prompt_template import *" >> episcape/patient_gen_utils/__init__.py


python make.py

