import json
import os

from tqdm import tqdm

from llms4subjects.instance import EmbeddingQuery as EmbeddingQuery
from llms4subjects.parse import parse_jsonld
from llms4subjects.predict import Predictor
from llms4subjects.predict.predict_llm import PredictByExamples
from llms4subjects.predict.predict_sft import PredictBySftLlama
from llms4subjects.predict.predict_simple import PredictByInstance
from llms4subjects.subject import subject_db_all as subject_db


def merge_gndcodes_file(
    old_json_file: str, llm_file: str, out_json_file: str
) -> None:
    print(f"process {old_json_file} => {out_json_file}")

    # with open(old_json_file, "r", encoding="utf-8") as f:
    #     data = json.load(f)
    #     codes = data["dcterms:subject"]

    if os.path.exists(llm_file):
        print(llm_file)
        # with open(llm_file, "r", encoding="utf-8") as f:
        #     data = json.load()

    # out_data = {"dcterms:subject": codes}
    # with open(out_json_file, "w", encoding="utf-8") as f:
    #     json.dump(out_data, f, ensure_ascii=False, indent=2)


def merge(
    test_result_dir: str, llm_result_dir: str, merge_out_dir: str
) -> int:
    """
    遍历test_result_dir，处理json文件，合并之前LLM处理完毕的结果，把结果输出到对应的merge_out_dir
    """
    n_processed = 0
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
            merge_gndcodes_file(src_file, llm_result_file, out_file)
            n_processed += 1
    return n_processed


if __name__ == "__main__":
    n_merged = merge(
        test_result_dir="./db/test_result_core",
        llm_result_dir="./db/temp/Classification",
        merge_out_dir="./db/test_result_core_v2_merged",
    )
    print(f"processed core: {n_merged}")
