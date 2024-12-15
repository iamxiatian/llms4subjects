import json
from collections import defaultdict

from tqdm import tqdm

from llms4subjects.instance import EmbeddingQuery as InstanceEmbeddingQuery
from llms4subjects.subject import EmbeddingQuery as SubjectEmbeddingQuery
from llms4subjects.subject import SubjectDb


QUERY_TOP_K = 5


subject_db = SubjectDb.open_all()


def by_similar_instances(
    title: str,
    abstract: str = "",
    dataset_type: str = "core",
    topk: int = QUERY_TOP_K,
) -> tuple[list[str], list[str]]:
    """根据标题和摘要的Embedding，寻找最相似的样本，将样本对应的codes作为
    结果返回"""
    input = f"""title:{title}\nabstract:{abstract}"""
    eq = InstanceEmbeddingQuery(f"./db/instance/{dataset_type}")
    instances = eq.get_instances(input, topk)
    codes: dict[str, float] = defaultdict(float)
    for inst in instances:
        for idx, code in enumerate(inst.gnd_codes):
            codes[code] = codes[code] + 1.0 + 1.0 / (idx + 1)

    # 按照出现数量多少排序
    sorted_items: list[tuple[str, float]] = sorted(
        codes.items(), key=lambda item: item[1], reverse=True
    )

    # TODO: 考虑出现在title和abstract时，要分别加权
    final_codes = [code for code, _ in sorted_items]
    final_names = [subject_db.get_name_by_code(c) for c in final_codes]
    return final_codes, final_names


def by_similar_subjects(
    title: str,
    abstract: str = "",
    dataset_type: str = "core",
    topk: int = QUERY_TOP_K,
) -> tuple[list[str], list[str]]:
    """根据标题和摘要的Embedding，寻找最相似的主题名称，将其对应的codes作为
    结果返回"""
    input = f"""title:{title}\nabstract:{abstract}"""
    eq = SubjectEmbeddingQuery(f"./db/subject/{dataset_type}")
    namecodes = eq.get_name_code_list(input, topk)
    final_codes = [code for _, code in namecodes]
    final_names = [name for name, _ in namecodes]
    return final_codes, final_names


def get_dev_dataset(dataset_type: str = "core") -> list:
    """获取所有的开发集数据。
    Args:
        dataset_type: core|all

    Return:
        一个数组，数组元素为四元组，分别为:
        id,title,abstract,true_codes,true_names
    """
    dataset = []
    with open(
        f"./db/instance/{dataset_type}/dev.jsonline", "r", encoding="utf-8"
    ) as dev_f:
        for line in tqdm(dev_f.readlines()):
            record = json.loads(line)
            true_codes = [
                c["@id"].rsplit("/", 1)[-1] for c in record["gnd_codes"]
            ]
            true_codes = [f"gnd:{c}" for c in true_codes]
            true_names = [subject_db.get_name_by_code(c) for c in true_codes]
            dataset.append(
                (
                    record["id"],
                    record["title"],
                    record["abstract"],
                    true_codes,
                    true_names,
                )
            )
    return dataset
