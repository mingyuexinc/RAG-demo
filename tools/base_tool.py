from abc import abstractmethod, ABC
from typing import List
from core.executor import ExecutionContext


class BaseTool(ABC):
    name:str
    input_keys:List[str]
    output_key:str

    def __init__(self,name:str):
        self.name = name

    def run(self,params:dict,context:ExecutionContext,set_input_to_context:bool):
        for key in self.input_keys:
            if key not in params:
                raise ValueError(f"Missing required param: {key}")
            if set_input_to_context:
                context.set(key,params[key])
        self.execute(context)
        return context.get(self.output_key)


    @abstractmethod
    def execute(self,context:ExecutionContext):
        pass