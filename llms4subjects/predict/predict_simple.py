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
        for inst in instances:
            for idx, code in enumerate(inst.gnd_codes):
                codes[code] = codes[code] + 1.0 + 1.0 / (idx + 1)

                # 考虑出现在title和abstract时，分别加权
                name = subject_db.get_name_by_code(code)
                if name in title:
                    codes[code] = codes[code] + 2
                if name in abstract:
                    codes[code] = codes[code] + 1

        # 按照出现数量多少排序
        sorted_items: list[tuple[str, float]] = sorted(
            codes.items(), key=lambda item: item[1], reverse=True
        )

        final_codes = [code for code, _ in sorted_items]
        final_names = [subject_db.get_name_by_code(c) for c in final_codes]
        return final_codes, final_names


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
