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


def generate_gndcodes_file(
    jsonld_file: str, out_json_file: str, predictor: Predictor
) -> None:
    #print(f"process {jsonld_file} => {out_json_file}")
    record = parse_jsonld(jsonld_file)

    codes, names = predictor(record["title"], record["abstract"])
    out_data = {"dcterms:subject": codes}
    with open(out_json_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)


def run_test(test_dir: str, out_dir: str, predictor: Predictor) -> int:
    """
    遍历testset_dir，处理jsonld文件，并把结果输出到对应的output_dir
    目录中的文件
    
    Args:
        test_dir: 赛方提供的测试数据集，是一个文件夹字符串，递归遍历其中包含的
            jsonld文件作为测试文件
        out_dir: 预测结果文件夹，目录结构和test_dir保持了一致，每一个jsonld
            文件，会在out_dir的相同相对目录结构下，生成一个json文件，保存预测
            结果
    Returns:
        处理的文件数量，为一个整数
    
    """
    n_processed = 0
    for dirpath, _, filenames in tqdm(os.walk(test_dir)):
        for filename in filenames:
            # 检查文件是否以 .jsonld 结尾
            if not filename.endswith(".jsonld"):
                continue

            # 提取不含扩展名的文件名
            file_name_without_ext = os.path.splitext(filename)[0]

            # 构建目标文件夹路径（确保目标文件夹存在）
            out_path = os.path.join(
                out_dir, os.path.relpath(dirpath, test_dir)
            )
            os.makedirs(out_path, exist_ok=True)

            # 构建目标文件路径（.json 扩展名）
            out_file = os.path.join(out_path, file_name_without_ext + ".json")

            # 构建源文件路径
            src_file = os.path.join(dirpath, filename)

            generate_gndcodes_file(src_file, out_file, predictor)
            n_processed += 1
    return n_processed


if __name__ == "__main__":
    predictor = PredictByInstance(dataset_type="merged_with_dev", topk=50)
    n_processed = run_test(
        test_dir="./test", out_dir="./db/test_result", predictor=predictor
    )
    print(f"processed: {n_processed}")
