import json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from llms4subjects.instance import EmbeddingQuery as EmbeddingQuery
from llms4subjects.subject import subject_db_all as subject_db
from llms4subjects.predict import Predictor
from llms4subjects.predict.predict_simple import PredictByInstance


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
            ap += n_matched / i
    ap = ap / len(true_codes)
    return ap


def eval(
    dataset_type: str,
    predictor: Predictor,
    middle_file: str,
    result_file: str,
) -> defaultdict[str, float]:
    """对指定的dataset_type(all|core)，按照predictor进行预测
    Args:
    - dataset_type: all|core
    - predictor： 预测方式
    - middle_file: 保存每个测试样本的预测结果，为一个jsonline文件
    - result_file: 最终的预测指标结果，是一个文本文件
    """
    metrics: dict[str, float] = defaultdict(float)
    records = get_dev_dataset(dataset_type)
    Path(middle_file).parent.mkdir(parents=True, exist_ok=True)
    with open(middle_file, "w", encoding="utf-8") as f:
        for record in tqdm(records):
            true_codes = record["true_codes"]
            pred_codes, _ = predictor(record["title"], record["abstract"])
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

    lines = [f"{k}\t{v}" for k, v in metrics]
    Path(result_file).write_text("\n".join(lines), encoding="Utf-8")

    return metrics


def main():
    dataset_type = "all"
    middle_file = f"./db/eval/{dataset_type}/by_instance_5.jsonline"
    result_file = f"./db/eval/{dataset_type}/by_instance_5.txt"

    predictor = PredictByInstance(dataset_type=dataset_type)
    metrics = eval(dataset_type, predictor, middle_file, result_file)
    print(metrics)


if __name__ == "__main__":
    main()
