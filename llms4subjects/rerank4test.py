import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from llms4subjects.llm import LLM

chatbot = LLM(
    base_url="http://10.96.1.42:7832/v1",
    model="Qwen-QwQ-32B-AWQ",
)


def make_prompt(record) -> str:
    topics = [f"  - {name.strip()}" for name in record["pred_names"]]
    topics = "\n".join(topics)
    title, abstract = record["title"], record["abstract"]

    return f"""
You act as an expert in library subject indexing. Please carefully analyze the given document title and abstract, review the given list of reference topics, and reorder them according to their degree of relevance to the document. Irrelevant topics can be removed, and new topics can also be added. Pay attention that after the intermediate analysis, you must finally output the "Final topic list". In the final topic list, only the topic names should be outputted, with one topic name per line, and there should be no other explanatory information mixed in.

## Here is an example of the input and output format
### Title: xxxx
### Abstract: xxxx
### Reference sorted list of document topics:
  - Topic 1
  - Topic 2
    
### Analysis process
(omitted)

### Final topic list
  - Topic 1
  - Topic 2

## Normal processing starts here

### Title: {title}
### Abstract: {abstract}
### Reference sorted list of document topics:
{topics}

### Analysis process
"""


def rerank_one_by_llm(prompt:str) -> str:
    text = chatbot.chat(user_prompt=prompt)
    data = json.loads(text)
    answer: str = data["choices"][0]["message"]["content"]
    return answer


def rerank_testset(
    test_dir: str = "./db/test/core", start: int = 0, end: int | None = None
) -> None:
    """根据预测的结果文件，利用大模型排序，并将排序的输出结果保存到
    llm_output_file， 供后面合并使用"""
    # 读取带有名称的数据
    test_files: list[Path] = list(Path(test_dir).glob("**/*.detail"))

    # 截取本轮次要处理的文件
    test_files = test_files[start:end]
    lineno = start
    for detail_file in tqdm(test_files):
        start_time = time.time()
        with open(detail_file, "r", encoding="utf-8") as f:
            record = json.load(f)
        prompt = make_prompt(record)
        answer = rerank_one_by_llm(prompt)
        seconds = time.time() - start_time
        record["lineno"] = lineno
        record["finish_time"] = f"{datetime.now()}"
        record["prompt"] = prompt
        record["answer"] = answer
        record["used_seconds"] = seconds

        out_file: Path = detail_file.with_suffix(".llm")
        with out_file.open("w", encoding="utf-8") as out:
            json.dump(record, out, ensure_ascii=False, indent=2)
        lineno += 1
    print("{datetime.now()}: DONE.")


def main() -> None:
    parser = argparse.ArgumentParser(description="rerank from start to end")
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=False, default=None)

    args = parser.parse_args()
    print(f"起始值: {args.start}")
    print(f"结束值: {args.end}")
    
    test_dir: str = "./db/test/core"
    rerank_testset(test_dir, args.start, args.end)

if __name__ == "__main__":
    main()