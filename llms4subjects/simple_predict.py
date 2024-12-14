import json
from collections import defaultdict

from tqdm import tqdm

from llms4subjects.instance import EmbeddingQuery as InstanceEmbedding
from llms4subjects.subject import SubjectDb
QUERY_TOP_K = 5


subject_db = SubjectDb.open_all()

def predict(
    title: str, abstract: str = "", dataset_type: str = "core"
) -> tuple[list[str], list[str]]:
    """根据标题和摘要，预测GND Codes和Names"""
    input = f"""title:{title}\nabstract:{abstract}"""
    eq = InstanceEmbedding(f"./db/instance/{dataset_type}")
    instances = eq.get_instances(input, QUERY_TOP_K)
    codes: dict[str, float] = defaultdict(float)
    for inst in instances:
        for idx, code in enumerate(inst.gnd_codes):
            codes[code] = codes[code] + 1.0 + 1.0/(idx+1)

    # 按照出现数量多少排序
    sorted_items: list[tuple[str, float]] = sorted(
        codes.items(), key=lambda item: item[1], reverse=True
    )
    
    # TODO: 考虑出现在title和abstract时，要分别加权
    final_codes = [code for code, _ in sorted_items]
    final_names = [subject_db.get_name_by_code(c) for c in final_codes]
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
                    true_names
                )
            )
    return dataset
