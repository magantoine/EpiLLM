from decorators import expose
from .model_utils import Models

@expose
class BaseModel(Models):
    def __init__(self) -> None:
        pass

