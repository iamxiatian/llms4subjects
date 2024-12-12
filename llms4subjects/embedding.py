"""
读取已经导出的TIBKAT jsonline文件，内容包含了从jsonld文件中抽取出的title、abstract、gnd_ids三个字段，已经从文件名称中抽取得到的id字段。即如下四个字段：
- id(str)
- title(str)
- abstract(str)
- gnd_ids(list)

然后遍历文件，将title和abstract拼接在一起，调用Embedding服务生成Embedding结果，
存入文本文件之中。
"""

import json
from tqdm import tqdm
import requests
import re
import faiss
from pathlib import Path
import numpy as np

TIBKAT_core_file = "./TIBKAT-core.jsonline"
TIBKAT_all_file = "./TIBKAT-all.jsonline"

EMBEDDING_SERVER_URL = "http://10.96.1.43:8188/emb"


def generate_api(text: str) -> list[float]:
    """生成embedding的接口，通过调用RESTful API实现"""
    headers = {
        "Content-Type": "application/json",
    }
    data = {"text": [text]}
    response = requests.post(EMBEDDING_SERVER_URL, headers=headers, json=data)
    response = response.json()
    embedding = response["result"][0]
    return embedding


def generate_file(jl_file: str, out_file: str) -> None:
    """读取导出的TIBKAT jsonline文件，生成embedding，保存到out_file中。
    文件中每一行为id、制表符、逗号分割的embedding数值"""

    with open(jl_file, "r", encoding="utf-8") as f_in:
        with open(out_file, "w", encoding="utf-8") as f_out:
            for line in tqdm(f_in.readlines()):
                record = json.loads(line)
                id, title, abstract = (
                    record["id"],
                    record["title"],
                    record["abstract"],
                )
                text = f"""title: "{title}"\n abstract: {abstract}"""
                embedding = generate_api(text)
                embedding = [str(e) for e in embedding]
                embedding = ",".join(embedding)
                f_out.write(f"{id}\t{embedding}\n")
                f_out.flush()


def save_faiss(
    embedding_txt_file: str, embedding_idx_file: str, embedding_ids_file: str
) -> None:
    """
    将embedding_txt_file中的embedding保存到faiss中，方便检索.
    Args:
    - embedding_txt_file: 保存有文件名称id和embedding结果的文本文件
    - embedding_idx_file： 保存faiss索引的文件
    - embedding_ids_file：保存有ids的文件， 方便查询时能够得到向量对应的id
    """
    dim = 1024
    # 使用Inner Product (IP) 距离的IndexFlat
    index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
    ids = []
    with open(embedding_txt_file, "r", encoding="utf-8") as f_in:
        for line in tqdm(f_in.readlines()):
            parts = re.split(r"[,\t]", line)
            ids.append(parts[0])
            value = [float(v) for v in parts[1:]]
            value = np.array(value, dtype=np.float32).reshape(1, dim)
            index.add(value)
    Path(embedding_ids_file).write_text("\n".join(ids), encoding="utf-8")
    faiss.write_index(index, embedding_idx_file)


class EmbeddingQuery:
    def __init__(self, idx_file: str, ids_file: str):
        """读取已经利用FAISS索引的数据文件以及对应的id文件"""
        with open(ids_file) as f:
            ids = [line.strip() for line in f.readlines()]
        self.ids = ids
        self.index: faiss.IndexFlatIP = faiss.read_index(idx_file)
        self.dim = 1024

    def query(self, text: str, topk) -> list[int]:
        q = np.array(generate_api(text), dtype=np.float32).reshape(1, self.dim)
        _, labels = self.index.search(q, topk)
        ids:list[int] = labels[0].tolist()
        return ids


if __name__ == "__main__":
    # generate_file(TIBKAT_core_file, "./embedding-core.txt")
    # generate_file(TIBKAT_all_file, "./embedding-all.txt")
    save_faiss(
        "./embedding-core.txt", "./embedding-core.idx", "./embedding-core.ids"
    )

    save_faiss(
        "./embedding-core.txt", "./embedding-core.idx", "./embedding-core.ids"
    )
