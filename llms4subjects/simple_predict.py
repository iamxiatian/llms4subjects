import json
from collections import defaultdict

from tqdm import tqdm

from llms4subjects.instance import EmbeddingQuery as InstanceEmbedding

QUERY_TOP_K = 5


def predict(
    title: str, abstract: str = "", dataset_type: str = "core"
) -> list[str]:
    """根据标题和摘要，预测GND Codes"""
    input = f"""title:{title}\nabstract:{abstract}"""
    eq = InstanceEmbedding(f"./db/instance/{dataset_type}")
    instances = eq.get_instances(input, QUERY_TOP_K)
    codes: dict[str, int] = defaultdict(int)
    for inst in instances:
        for code in inst.gnd_codes:
            codes[code] = codes[code] + 1

    # 按照出现数量多少排序
    sorted_items: list[tuple[str, int]] = sorted(
        codes.items(), key=lambda item: item[1], reverse=True
    )
    final_codes = [code for code, _ in sorted_items]
    return final_codes


def get_dev_dataset(dataset_type: str = "core") -> list:
    """获取所有的开发集数据。
    Args:
        dataset_type: core|all

    Return:
        一个数组，数组元素为四元组，分别为id、title、abstract、true_codes
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
            dataset.append(
                (
                    record["id"],
                    record["title"],
                    record["abstract"],
                    true_codes,
                )
            )
    return dataset
