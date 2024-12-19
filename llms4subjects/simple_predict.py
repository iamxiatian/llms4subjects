import json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from llms4subjects.instance import EmbeddingQuery as EmbeddingQuery
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
    eq = EmbeddingQuery(f"./db/instance/{dataset_type}")
    instances = eq.get_instances(input, topk)
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
    namecodes = eq.get_namescode_by_text(input, topk)
    final_codes = [code for _, code in namecodes]
    final_names = [name for name, _ in namecodes]
    return final_codes, final_names


def get_dev_dataset(dataset_type: str = "core") -> list[dict]:
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
                {
                    "id": record["id"],
                    "title": record["title"],
                    "abstract": record["abstract"],
                    "true_codes": true_codes,
                    "true_names": true_names,
                }
            )
    return dataset


def p_at_k(pred_codes: list[str], true_codes: list[str], k: int) -> float:
    """计算p@k"""
    above = 0
    for code in pred_codes[:k]:
        if code in true_codes:
            above += 1

    return above * 1.0 / k

def AP(pred_codes: list[str], true_codes: list[str], k: int) -> float:
    """计算AP"""
    ap = 0.0
    n_matched = 0
    for i, code in enumerate(pred_codes, start=1):
        if code in true_codes:
            n_matched += 1
            ap += n_matched/i
    ap = ap / len(true_codes)
    return ap

def eval(dataset_type: str = "core") -> defaultdict[str, float]:
    topk = 5
    metrics: dict[str, float] = defaultdict(float)
    records = get_dev_dataset(dataset_type)
    eval_file = f"./db/eval/{dataset_type}/by_similar_records_{topk}.jsonline"
    Path(eval_file).parent.mkdir(parents=True, exist_ok=True)
    with open(eval_file, "w", encoding="utf-8") as f:
        for record in tqdm(records):
            true_codes = record["true_codes"]
            pred_codes, _ = by_similar_instances(
                record["title"], record["abstract"], "all", 5
            )
            entry = {
                "id": record["id"],
                "title": record["title"],
                "true_codes": record["true_codes"],
                "pred_codes": pred_codes,
            }
            jsonline = json.dumps(entry, ensure_ascii=False)
            f.write(f"{jsonline}\n")
            for k in range(1, 11):
                metrics[f"p@{k}"] += p_at_k(pred_codes, true_codes, k)
                metrics[f"map@{k}"] += AP(pred_codes, true_codes, k)

    for name in metrics.keys():
        metrics[name] = metrics[name] / len(records)

    return metrics


if __name__ == "__main__":
    metrics = eval("all")
    print(metrics)
