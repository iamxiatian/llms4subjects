import json
import os
from pprint import pprint

from tqdm import tqdm

from llms4subjects.llm import LLM

chatbot = LLM(
    base_url="http://10.96.1.43:7832/v1",
    model="/data/app/yangyahe/base_model/Qwen-QwQ-32B-AWQ",
)


def make_prompt(record):
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


def rerank(record) -> str:
    prompot = make_prompt(record)
    text = chatbot.chat(user_prompt=prompot)
    data = json.loads(text)
    answer: str = data["choices"][0]["message"]["content"]
    return answer


def main():
    # 读取带有名称的数据
    dev_names_file = "./db/eval/merged/by_instance_5.dev2_with_names.jsonline"
    with open(dev_names_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f.readlines()]
    print(f"load {len(records)} records.")
    
    llm_output_file = "./db/eval/merged/by_instance_5.dev2.llm_output.jsonline"
    with open(llm_output_file, "w", encoding="utf-8") as f:
        for lineno, r in tqdm(enumerate(records)):
            answer = rerank(r)
            data = {"lineno": lineno, "id": r["id"], "answer": answer}
            s = json.dumps(data, ensure_ascii=False)
            f.write(s)
            f.write("\n")
            f.flush()
    print("DONE.")


if __name__ == '__main__':
    main()