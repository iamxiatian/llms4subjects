import json
from datetime import datetime
import time
import itertools
from tqdm import tqdm

from llms4subjects.llm import LLM
from llms4subjects.subject import subject_eq

chatbot = LLM(
    base_url="http://10.96.1.42:7832/v1",
    model="Qwen-QwQ-32B-AWQ",
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


def rerank_one_by_llm(prompt) -> str:
    text = chatbot.chat(user_prompt=prompt)
    data = json.loads(text)
    answer: str = data["choices"][0]["message"]["content"]
    return answer


def reranking_all()->None:
    """根据预测的结果文件，利用大模型排序，并将排序的输出结果保存到
    llm_output_file， 供后面合并使用"""
    # 读取带有名称的数据
    dev_names_file = "./db/eval/merged/by_instance_5.dev2_with_names.jsonline"
    with open(dev_names_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f.readlines()]
    print(f"{datetime.now()}: load {len(records)} records.")

    llm_output_file = "./db/eval/merged/by_instance_5.dev2.llm_output.jsonline"
    with open(llm_output_file, "w", encoding="utf-8") as f:
        for lineno, r in tqdm(enumerate(records)):
            start_time = time.time()
            prompt = make_prompt(r)
            answer = rerank_one_by_llm(prompt)
            seconds = time.time() - start_time
            data = {
                "lineno": lineno,
                "id": r["id"],
                "finish_time": f"{datetime.now()}",
                "used_seconds": seconds,
                "prompt": prompt,
                "answer": answer,
            }
            s = json.dumps(data, ensure_ascii=False)
            f.write(s)
            f.write("\n")
            f.flush()
    print("{datetime.now()}: DONE.")


def __extract_topics(answer: str) -> list[str]:
    """从LLM输出的答案中提取所有话题，如果没有提取到，则返回[]"""

    if "### Final topic list" not in answer:
        return []
    
    lines = answer.split("\n")

    # 从倒数第一个话题开始提取，直到遇到 Final topic list
    topics = itertools.takewhile(lambda x: not x.startswith("### Final topic list"), reversed(lines))
    topics = reversed(list(topics))
    
    # 跳过后续多余的解释内容
    topics = itertools.takewhile(lambda x: not (x.startswith("### Explanation") or x.strip()=="" or x.strip() == "-"), topics)

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
        topics = [topic[1:].strip() for topic in topics if topic.startswith("-")]
        
    return topics

    
def __map_to_namecode(llm_topic_name:str) -> tuple[str, str]:
    items = subject_eq.get_namecodes_by_name(llm_topic_name, 1)
    return items[0]
    
def generate_final_result(raw_pred_result_file:str, llm_output_file:str, final_r3_file:str):
    """读取原始的预测结果，以及大模型重排序的原始输出结果，合并形成最终的
    r3结果文件"""
    
    # 读取开发集预测结果，保存到变量 records中
    with open(raw_pred_result_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f.readlines()]

    # 读取LLM再次排序后的输出文件，保存到变量 documents中
    with open(llm_output_file, "r", encoding="utf-8") as f:
        documents = [json.loads(line) for line in f.readlines()]
    
    with open(final_r3_file, "w", encoding="utf-8") as f:
        for idx, (record, doc) in tqdm(enumerate(zip(records, documents))):
            answer = doc["answer"]
            topics = __extract_topics(answer)
            
            if len(topics) == 0:
                print(f"No topics found for document {idx}")
                # 采用默认的预测结果
                llm_rerank_names = record["pred_names"]
            else:
                llm_rerank_names = topics
        
            r3_names = []
            r3_codes = []
            for name in llm_rerank_names:
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
            record['llm_rerank_names'] = llm_rerank_names
            record['r3_names'] = r3_names
            record['r3_codes'] = r3_codes
        
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

if __name__ == "__main__":
    generate_final_result(
        "./db/eval/merged/by_instance_5.dev2_with_names.jsonline", 
        "./db/eval/merged/by_instance_5.dev2.llm_output.jsonline", 
        "./db/eval/merged/r3.jsonline")
