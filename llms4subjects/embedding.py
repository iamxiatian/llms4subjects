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
from llms4subjects.instance import tibkat_db
from llms4subjects.api import get_embedding as generate_api


class EmbeddingQuery:
    def __init__(self, idx_file: str):
        """读取已经利用FAISS索引的数据文件以及对应的id文件"""
        self.index: faiss.IndexFlatIP = faiss.read_index(idx_file)
        self.dim = 1024

    def query(self, text: str, topk) -> list[int]:
        """将文本转换为embedding，然后利用faiss查找，将匹配结果的序号返回"""
        q = np.array(generate_api(text), dtype=np.float32).reshape(1, -1)
        _, labels = self.index.search(q, topk)
        label_ids:list[int] = labels[0].tolist()
        return label_ids


if __name__ == "__main__":
    # print("Step 1: generate embedding files...")
    # generate_file(TIBKAT_core_file, "./embedding-core.txt")
    # generate_file(TIBKAT_all_file, "./embedding-all.txt")
    
    # print("Step 2: index embeddings by faiss...")
    # save_faiss(
    #     "./embedding-core.txt", "./embedding-core.idx", "./embedding-core.ids"
    # )
    # save_faiss(
    #     "./embedding-core.txt", "./embedding-core.idx", "./embedding-core.ids"
    # )
    
    print("Step 3: query test...")
    eq = EmbeddingQuery("./embedding-core.idx", "./embedding-core.ids")
    doc_ids = eq.query("title: information retrieval", 5)
    for doc_id in doc_ids:
        title, codes = tibkat_db.title_with_gnd_codes(doc_id)
        print(f"{title}:\t {codes}")
