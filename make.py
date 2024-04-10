import os
DIR_PATH = "/Users/antoinemagron/Documents/EPFL/PDM/crchum/"
TOP, BOTTOM = 0, 1
MODULES = [("episcape", TOP), ("evaluation", TOP), ("misc", TOP), ("models", TOP), ("episcape.patient_gen_utils", BOTTOM)]




def make_dependencies():
    for modn, level in MODULES:
        __init__ = """"""
        mod = __import__(modn)

        ## for nested module
        short_module = modn
        if(level == BOTTOM):
            short_module = modn.split(".")[-1]
        template = "from {module} import {object}\n"
        for f in dir(mod):
            cont = None
            if("dict" in str(type(mod.__dict__[f]))):
                cont = dict(mod.__dict__[f])
            elif(hasattr(mod.__dict__[f], "__dict__") or "dict" in str(type(mod))):
                cont = dict(mod.__dict__[f].__dict__)
            else :
                pass
            if(cont is not None and "exposed" in cont):
                complete_module = mod.__dict__[f].__module__
                object = mod.__dict__[f].__name__
                ## we don't add if comming from top module :
                if(short_module in complete_module):
                    if(not (level == TOP and complete_module.count(".") > 1)):
                        __init__+=(
                            template.format(
                                module=complete_module.replace(modn, ""),
                                object=object
                                )
                            )
        print("_"*100)
        print(modn)    
        print(__init__)


        with open(f"{modn.replace('.', '/')}/__init__.py", "w") as f:
            f.write(__init__)


if __name__ == "__main__":
    make_dependencies()
    