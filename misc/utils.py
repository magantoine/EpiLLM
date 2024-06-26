from decorators import expose
from matplotlib import pyplot as plt
import pandas as pd
from typing import List, Dict, Callable, Any
import os
import pickle
from evaluation import MCQBenchmark

@expose
def plot_age_pyramid(df):
    if("Age" not in df.columns
       or "Male" not in df.columns
       or "Female" not in df.columns):
        raise ValueError("Need df with columns : ['Age', 'Male', 'Female']")
    y = range(0, len(df))
    x_male = df['Male']
    x_female = df['Female']

    #define plot parameters
    fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(9, 6))

    #specify background color and plot title
    fig.patch.set_facecolor('xkcd:light grey')
    plt.figtext(.5,.9,"Population Pyramid ", fontsize=15, ha='center')
        
    #define male and female bars
    axes[0].barh(y, x_male, align='center', color='royalblue')
    axes[0].set(title='Males')
    axes[1].barh(y, x_female, align='center', color='lightpink')
    axes[1].set(title='Females')

    #adjust grid parameters and specify labels for y-axis
    axes[1].grid()
    axes[0].set(yticks=y, yticklabels=df['Age'])
    axes[0].invert_xaxis()
    axes[0].grid()

    #display plot
    plt.show()


@expose
def from_list_to_df(patients):
    if(type(patients) != list):
        raise ValueError("The input is not a list.")

    return pd.DataFrame([_.__dict__() for _ in patients])


@expose
def write_values(gens: List[str],
                 fpath: str="docs/temp.md",
                 psep:str="*",
                 seprep:int=100) -> None:
    
    with open(fpath, 'w') as md:
        md.write((f"\n\n\n{psep*seprep}\n\n\n").join(gens))
    



BENCHMARKS_PATHS = {
    "MCQ" : "docs/benchmarks/mcq40/processed.json",
    "aes7" : "docs/benchmarks/self_assessment/aes7_processed.json",
    "aes8" :  "docs/benchmarks/self_assessment/aes8_processed.json",
}

@expose
def load_pickle(path:str,
                available_benchmarks: List[str]=["aes7", "aes8"],
                procfunc: Callable[[Any], List[str]]=lambda obj:[x.outputs[0].text for x in obj]):
    
    with open(path, "rb") as f:
        obj = pickle.load(f)
    gens = procfunc(obj)

    for benchmark in available_benchmarks:
        if(benchmark in path):
            print("Associated benchmark : ", benchmark)
            mcqs = MCQBenchmark(BENCHMARKS_PATHS[benchmark], lambda:()).mcq

    return [(mcq["question"], mcq["answer"], gen) for (mcq, gen) in zip(mcqs, gens)]
