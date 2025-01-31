import math
from collections import defaultdict

from llms4subjects.instance import get_embedding_query
from llms4subjects.subject import subject_db_all as subject_db
from llms4subjects.subject import subject_eq

from . import Predictor

QUERY_TOP_K = 5


class PredictByInstance(Predictor):
    def __init__(
        self, dataset_type: str = "all", topk: int = QUERY_TOP_K
    ) -> None:
        super().__init__()
        self.dataset_type = dataset_type
        self.topk = topk

    def predict(
        self, title: str, abstract: str = ""
    ) -> tuple[list[str], list[str]]:
        """根据标题和摘要的Embedding，寻找最相似的样本，将样本对应的codes作为
        结果返回，返回结果为二元组(codes，names)
        
        Returns:
            (codes，names)二元组，格式：tuple[list[str], list[str]]
        """
        input = f"""title:{title}\nabstract:{abstract}"""
        eq = get_embedding_query(self.dataset_type)
        instances = eq.get_instances(input, self.topk)
        codes: dict[str, float] = defaultdict(float)
        for rank_of_instance, inst in enumerate(instances, 1):
            for rank_of_code, code in enumerate(inst.gnd_codes, 1):
                # 第一个文档的重要性最高，其他文档重要性依次降低
                
                score = 1.0 + 1.0 / rank_of_code

                # 考虑出现在title和abstract时，分别加权
                # 同时考虑别名信息
                if rank_of_instance <=5:
                    subject = subject_db.get_subject_by_code(code)
                    names = [n.lower()  for n in subject.alternate_names if not n.isupper()]
                    names.append(subject.name.lower())
                    for name in names:
                        if name in title.lower():
                            score = score + 2
                            break
                        
                        if name in abstract.lower():
                            score = score + 1
                            break
                
                # 按照文档排序，赋权
                value = len(instances)/rank_of_instance
                score = score* (1+math.log(value))
                codes[code] = codes[code] + score

        # 按照出现数量多少排序
        sorted_items: list[tuple[str, float]] = sorted(
            codes.items(), key=lambda item: item[1], reverse=True
        )

        final_codes = [code for code, _ in sorted_items]
        final_names = [subject_db.get_name_by_code(c) for c in final_codes]
        # 最多返回50个
        return final_codes[:50], final_names[:50]


class PredictBySubject(Predictor):
    def __init__(
        self, dataset_type: str = "all", topk: int = QUERY_TOP_K
    ) -> None:
        super().__init__()
        self.dataset_type = dataset_type
        self.topk = topk

    def predict(
        self, title: str, abstract: str = ""
    ) -> tuple[list[str], list[str]]:
        """根据标题和摘要的Embedding，寻找最相似的主题名称，将其对应的codes作为
        结果返回"""
        input = f"""title:{title}\nabstract:{abstract}"""
        namecodes = subject_eq.get_namescode_by_text(input, self.topk)
        final_codes = [code for _, code in namecodes]
        final_names = [name for name, _ in namecodes]
        return final_codes, final_names
