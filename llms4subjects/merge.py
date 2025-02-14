import json
import os
from collections import defaultdict
from tqdm import tqdm

from llms4subjects.instance import EmbeddingQuery as EmbeddingQuery
from llms4subjects.parse import parse_jsonld
from llms4subjects.predict import Predictor
from llms4subjects.predict.predict_llm import PredictByExamples
from llms4subjects.predict.predict_sft import PredictBySftLlama
from llms4subjects.predict.predict_simple import PredictByInstance
from llms4subjects.subject import subject_eq


def merge_gndcodes_file(
    old_json_file: str, llm_file: str, out_json_file: str
) -> int:
    n = 0
    final_scores: dict[str, float] = defaultdict(float)
    with open(old_json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        old_codes = data["dcterms:subject"]
        for idx, c in enumerate(old_codes, 1):
            final_scores[c] = 1/idx
            
    if os.path.exists(llm_file):
        print(f"process llm file {llm_file} => {out_json_file}")
        with open(llm_file, "r") as f:
            lines = [line.strip() for line in f.readlines()]
            keywords = [line[1:-2] for line in lines if len(line)>5 and line[0]=='"' and line[-2:]=='",']
            
            code_scores: dict[str, float] = defaultdict(float)
            for i, keyword in enumerate(keywords, 1):
                namecodes = subject_eq.get_namecodes_by_name(keyword, 1)
                codes = [code for _, code in namecodes]
                for code in codes:
                    code_scores[code] = code_scores[code] + 1/i
                    
            # 按照出现数量多少排序
            sorted_items: list[tuple[str, float]] = sorted(
                code_scores.items(), key=lambda item: item[1], reverse=True
            )

            # 原来的codes和LLM预测得到的codes进行合并
            for code, score in sorted_items[:10]:
                final_scores[code] = final_scores[code] + 0.2*score
            
            final_items: list[tuple[str, float]] = sorted(
                final_scores.items(), key=lambda item: item[1], reverse=True
            )
            codes = [code for code, _ in final_items]
            codes = codes[:50]
        n = 1
    else:
        print(f"process old file {old_json_file} => {out_json_file}")
        codes = old_codes
            
    out_data = {"dcterms:subject": codes}
    with open(out_json_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    return n


def merge(
    test_result_dir: str, llm_result_dir: str, merge_out_dir: str
) -> int:
    """
    遍历test_result_dir，处理json文件，合并之前LLM处理完毕的结果，把结果输出到对应的merge_out_dir
    """
    n_merged = 0
    for dirpath, _, filenames in tqdm(os.walk(test_result_dir)):
        for filename in filenames:
            # 检查文件是否以 .jsonld 结尾
            if not filename.endswith(".json"):
                continue

            # 提取不含扩展名的文件名
            file_name_without_ext = os.path.splitext(filename)[0]

            # 构建目标文件夹路径（确保目标文件夹存在）
            merge_out_path = os.path.join(
                merge_out_dir, os.path.relpath(dirpath, test_result_dir)
            )
            os.makedirs(merge_out_path, exist_ok=True)

            # 构建目标文件路径（.json 扩展名）
            out_file = os.path.join(
                merge_out_path, file_name_without_ext + ".json"
            )

            llm_path = os.path.join(
                llm_result_dir, os.path.relpath(dirpath, test_result_dir)
            )
            llm_result_file = os.path.join(
                llm_path, file_name_without_ext + ".txt"
            )

            # 构建源文件路径
            src_file = os.path.join(dirpath, filename)
            n = merge_gndcodes_file(src_file, llm_result_file, out_file)
            n_merged += n
    return n_merged


if __name__ == "__main__":
    n_merged = merge(
        test_result_dir="./db/test_result_core",
        llm_result_dir="./db/temp/llm_output_new",
        merge_out_dir="./db/test_result_core_v2_merged",
    )
    print(f"processed core: {n_merged}")
