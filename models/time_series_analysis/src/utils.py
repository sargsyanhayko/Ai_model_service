import json
from types import SimpleNamespace

class Config:
    def __init__(self, input_data):
        if isinstance(input_data, str):
            with open(input_data, 'r') as file:
                data = json.load(file)
        elif isinstance(input_data, dict):
            data = input_data
        else:
            raise TypeError('input_data must be a file path (str) or a dictionary.')
        self._data = json.loads(json.dumps(data), object_hook=lambda d: SimpleNamespace(**d))

    def __getattr__(self, item):
        return getattr(self._data, item)
    
    def __repr__(self):
        def recurse_namespace(ns):
            if isinstance(ns, SimpleNamespace):
                return {k: recurse_namespace(v) for k, v in vars(ns).items()}
            return ns

        return json.dumps(recurse_namespace(self._data), indent=4)