from abc import ABC, abstractmethod

class Predictor(ABC):
    @abstractmethod
    def predict(self, title:str, abstract:str) -> tuple[list[str], list[str]]:
        """根据标题和摘要预测主题编码和名称。
        
        Returns:
            (codes，names)二元组，格式：tuple[list[str], list[str]]
        """
        pass
    
    def __call__(self, *args, **kwds) -> tuple[list[str], list[str]]:
        return self.predict(args[0], args[1])