import argparse
import itertools
import json
import os
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from llms4subjects.llm import LLM
from llms4subjects.subject import subject_eq

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


def rerank_one_by_llm(prompt: str) -> str:
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


def __extract_topics(answer: str) -> list[str]:
    """从LLM输出的答案中提取所有话题，如果没有提取到，则返回[]"""

    if "### Final topic list" not in answer:
        return []

    lines = answer.split("\n")

    # 从倒数第一个话题开始提取，直到遇到 Final topic list
    topics = itertools.takewhile(
        lambda x: not x.startswith("### Final topic list"), reversed(lines)
    )
    topics = reversed(list(topics))

    # 跳过后续多余的解释内容
    topics = itertools.takewhile(
        lambda x: not (
            x.startswith("### Explanation")
            or x.strip() == ""
            or x.strip() == "-"
        ),
        topics,
    )

    # 去除前缀和空格
    topics = [topic.strip() for topic in topics]

    # 确保所有话题都以"- "开头
    n_symbol, n_normal = 0, 0
    for topic in topics:
        if topic.startswith("-") or topic.startswith("*"):
            n_symbol += 1
        else:
            n_normal += 1

    # 检查是否所有话题都以符号开头，或者没有符号
    assert n_symbol == len(topics) or n_normal == len(topics)

    if n_symbol > 0:
        topics = [
            topic[1:].strip() for topic in topics if topic.startswith("-")
        ]

    return topics


def __map_to_namecode(llm_topic_name: str) -> tuple[str, str]:
    items = subject_eq.get_namecodes_by_name(llm_topic_name, 1)
    return items[0]


def generate_final_result(
    testset_dir: str = "./db/test/core",
    final_result_dir: str = "./db/test/r3_core",
) -> None:
    """读取原始的预测结果，以及大模型重排序的原始输出结果，合并形成最终的
    r3结果文件"""

    llm_out_files: list[Path] = list(Path(testset_dir).glob("**/*.llm"))
    for llm_out_file in tqdm(llm_out_files):
        with open(llm_out_file, "r", encoding="utf-8") as f:
            record = json.load(f)

        answer = record["answer"]
        topics = __extract_topics(answer)

        if len(topics) == 0:
            print(f"No topics found for document {llm_out_file}")
            # 采用默认的预测结果
            record["llm_rerank_names"] = record["pred_names"]
            record["r3_names"] = record["pred_names"]
            record["r3_codes"] = record["pred_codes"]
            continue

        r3_names = []
        r3_codes = []
        for name in topics:
            name, code = __map_to_namecode(name)
            # 如果name在原来的pred中存在，则直接使用原来的code，而不是
            # 查询得到的code
            for n, c in zip(record["pred_names"], record["pred_codes"]):
                if n == name:
                    code = c
                    break

            r3_names.append(name)
            r3_codes.append(code)

        # 保存预测结果
        record["llm_rerank_names"] = topics
        record["r3_names"] = r3_names
        record["r3_codes"] = r3_codes

        # 重新保存到llm输出结果中
        with open(llm_out_file, "w", encoding="utf-8") as f1:
            json.dump(record, f1, ensure_ascii=False, indent=2)

        # 按照比赛要求，输出最终结果
        json_file = llm_out_file.with_suffix(".json")
        relative_path = os.path.relpath(json_file, testset_dir)
        result_file = os.path.join(final_result_dir, relative_path)
        Path(result_file).parent.mkdir(parents=True, exist_ok=True)

        with open(result_file, "w", encoding="utf-8") as f2:
            json.dump(
                {"dcterms:subject": r3_codes}, f2, ensure_ascii=False, indent=2
            )


def main_rerank() -> None:
    """调用大模型对之前通过Embedding方式的排序结果进行重排序"""
    parser = argparse.ArgumentParser(description="rerank from start to end")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=False, default=None)

    args = parser.parse_args()
    print(f"起始值: {args.start}")
    print(f"结束值: {args.end}")

    test_dir: str = "./db/test/core"
    rerank_testset(test_dir, args.start, args.end)


def main_gen_result():
    """生成最终需要提交给比赛的结果，需要在main_rerank()的处理完成后，
    再调用该方法生成结果"""
    generate_final_result(
        testset_dir="./db/test/core", final_result_dir="./db/test/r3_core"
    )


if __name__ == "__main__":
    main_gen_result()
    print("DONE.")
