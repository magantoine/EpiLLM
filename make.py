import os
print(os.environ)
DIR_PATH = "/Users/antoinemagron/Documents/EPFL/PDM/crchum/"
MODULES = ["episcape", "evaluate", "misc", "models"]


def get_subfiles(mod, listdir, path):
    subfile = [
        file for file in listdir if (".py" in file and "__init__" not in file)
    ]
    base = os.path.join(DIR_PATH, path)
    subdirs = [
        (
            subdir, os.listdir(
                os.path.join(base, subdir)
            )
        )
        for subdir in listdir if ("." not in subdir and "__pycache__" not in subdir)
    ]
    return [".".join([mod, file.replace(".py", "")]) for file in subfile] + sum([get_subfiles(subdir[0], subdir[1], os.path.join(base, subdir[0])) for subdir in subdirs], [])

def make_module(mod):
    cdir = os.path.join(DIR_PATH, mod)
    ldir = os.listdir(cdir)
    for ctnt in get_subfiles(mod, ldir, cdir):
        print(ctnt)
        print(dir(__import__(ctnt)))


    
    # print(">>> ", imod)
    # for imo in imod :
    #     print(imo)
    #     print(str(imo))
    #     print(mod.__dict__[imo].exposed)
    
import episcape
if __name__ == "__main__":
    print([   
    (episcape.__dict__[x].__dict__["__name__"] ,episcape.__dict__[x].__dict__.get("exposed", False))
    for x in dir(episcape)
    if "class" in str(type(episcape.__dict__[x]))
    ])
    # make_module(MODULES[0])