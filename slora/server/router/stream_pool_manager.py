import torch

class SingletonInstance:
    __instance = None
    
    @classmethod
    def __getInstance(cls):
        return cls.__instance
    
    @classmethod
    def instance(cls, *args, **kargs):
        cls.__instance = cls(*args, **kargs)
        cls.instance = cls.__getInstance
        return cls.__instance

class StreamPoolManager(SingletonInstance):
    def __init__(self):
        self.lora_stream = torch.cuda.Stream()
    