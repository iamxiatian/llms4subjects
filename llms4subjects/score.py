import json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


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


class Score:
    def __init__(
        self,
        pred_file: str,
        out_score_file: str,
        true_field: str = "true_codes",
        pred_field="pred_codes",
    ) -> None:
        """
            对预测结果文件进行分析，得到分值并保存到得分文件中。
            Args:
                - pred_file(str): 预测结果文件
                - out_score_file(str): 存放计算得分的结果文件
                - true_field: 真正的codes在json中对应的字段名称
                - pred_field: 需要评估的codes在json中对应的字段名称
        """
        self.pred_file = pred_file
        self.out_score_file = out_score_file
        self.true_field = true_field
        self.pred_field = pred_field

    def __calculate(self, records) -> defaultdict[str, float]:
        """对指定的dataset_type(all|core)，按照predictor进行预测"""
        metrics: dict[str, float] = defaultdict(float)

        for record in tqdm(records):
            true_codes = record[self.true_field]
            pred_codes = record[self.pred_field]
            for k in range(1, 11):
                metrics[f"p@{k}"] += p_at_k(pred_codes, true_codes, k)
                metrics[f"map@{k}"] += AP(pred_codes, true_codes, k)

        for name in metrics.keys():
            metrics[name] = metrics[name] / len(records)

        return metrics

    def __call__(self, *args, **kwds):
        # 读取最终预测结果，保存到变量 records中
        with open(self.pred_file, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f.readlines()]

        # 计算
        metrics = self.__calculate(records)
        lines = [f"{k}\t{v}" for k, v in metrics.items()]

        # 保存结果
        out_file = Path(self.out_score_file)
        out_file.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    score = Score(
        "./db/eval/merged/r3_fixed.jsonline",
        "./db/eval/merged/r3_score.txt",
        true_field = "true_codes",
        pred_field="r3_codes",
    )

    score()
    print("DONE.")
