"""
GND subject的别名与code映射信息，从GND-Subjects-all.json中提取出来，
存放在db/alias/all目录下，目录下存在以下文件：

- alias.sqlite 将name, code的映射信息，保存到sqlite，方便观察和查询，同时记录了embedding_id和名称与code的对应关系。
- alias.jsonline embedding_id和name对应的文件
- embedding.txt 根据alias.jsonline中的name，生成的embedding值
- embedding.idx 根据embedding.txt，生成的FAISS索引
"""

import json
from dataclasses import dataclass
from pathlib import Path
from sqlite3 import Row

import faiss
import numpy as np
from tqdm import tqdm

from llms4subjects.api import EMBEDDING_DIM, get_embedding
from llms4subjects.sqlite import SqliteDb


@dataclass
class Alias:
    embedding_id: int
    name: str
    codes: list[str]

    @classmethod
    def from_row(cls, row: Row) -> "Alias":
        """从sqlite3.Row对象创建实例"""
        return cls(
            embedding_id=row["embedding_id"],
            name=row["name"],
            codes=row["codes"].split(","),
        )


class AliasDb(SqliteDb):
    def __init__(self, db_file: str):
        SqliteDb.__init__(self, db_file)

        self.create_table("""CREATE TABLE IF NOT EXISTS alias (
            embedding_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            codes TEXT NOT NULL
        );""")

    @classmethod
    def open(cls) -> "AliasDb":
        db = AliasDb("./db/alias/all/subject.sqlite")
        return db

    def insert_alias(
        self,
        embedding_id: int,
        name: str,
        codes: str,
    ) -> int:
        sql = "INSERT INTO alias (embedding_id, name, codes) VALUES (?, ?, ?)"
        self.insert(sql, (embedding_id, name, codes))

    def get_codes_by_name(self, name: str) -> list[str]:
        sql = "SELECT codes from alias WHERE name = ?"
        rows = self.query(sql=sql, parameters=(name,))
        codes = []
        for row in rows:
            codes.extend(row["code"].split(","))
        return codes

    def get_by_embedding_id(self, embedding_id: int) -> Alias:
        """根据embedding_id返回code"""
        sql = "SELECT * FROM alias where embedding_id = ?"
        rows = self.query(sql, (embedding_id,))
        return Alias.from_row(rows[0])

    def exists(self, name, code) -> bool:
        sql = "SELECT codes from alias WHERE name = ?"
        rows = self.query(sql=sql, parameters=(name,))
        if not rows:
            return False

        return code in rows[0]["code"].split(",")

    def update_codes(self, name: str, codes: list[str]):
        sql = "update alias set codes = ? WHERE name = ?"
        self.update(
            sql,
            (
                ",".join(codes),
                name,
            ),
        )


def initialize(gnd_file: str, db_home: Path):
    """初始化，生成embedding等文件"""
    db_home.mkdir(parents=True, exist_ok=True)
    with open(gnd_file, "r", encoding="utf-8") as f:
        # 使用json.load()方法将文件内容解析为Python对象
        subjects: dict = json.load(f)
    alias_file = Path(db_home, "alias.jsonline").open("w", encoding="utf-8")
    embedding_file = Path(db_home, "embedding.txt").open("w", encoding="utf-8")
    # 使用Inner Product (IP) 距离的IndexFlat
    index: faiss.IndexFlatIP = faiss.IndexFlatIP(EMBEDDING_DIM)
    alias_db = AliasDb(Path(db_home, "alias.sqlite").as_posix())
    embedding_id = 0
    for entry in tqdm(subjects):
        code, name = entry["Code"], entry["Name"]
        codes = alias_db.get_codes_by_name(name)

        if codes:
            # 该name已经存在
            codes.append(code)
            alias_db.update_codes(name, codes)
            continue

        # 新记录
        alias_db.insert_alias(embedding_id, name, [code])

        record = json.dumps(
            {"embedding_id": embedding_id, "name": name},
            ensure_ascii=False,
        )
        alias_file.write(f"{record}\n")

        # insert embedding
        embedding = get_embedding(name)
        embedding_file.write(",".join([str(e) for e in embedding]))
        embedding_file.write("\n")
        value = np.array(embedding, dtype=np.float32).reshape(1, EMBEDDING_DIM)
        index.add(value)
        embedding_id += 1

    faiss.write_index(index, Path(db_home, "embedding.idx").as_posix())
    alias_file.close()
    embedding_file.close()
    alias_db.close()


class EmbeddingQuery:
    def __init__(self, db_path: Path):
        """读取已经利用FAISS索引的数据文件以及对应的id文件"""
        db_file = Path(db_path, "alias.sqlite").as_posix()
        idx_file = Path(db_path, "embedding.idx").as_posix()
        self.db = AliasDb(db_file)
        self.index: faiss.IndexFlatIP = faiss.read_index(idx_file)

    def get_embedding_ids(self, text: str, topk) -> list[int]:
        """将文本转换为embedding，然后利用faiss查找，将匹配结果的序号返回"""
        q = np.array(get_embedding(text), dtype=np.float32).reshape(1, -1)
        _, labels = self.index.search(q, topk)
        label_ids: list[int] = labels[0].tolist()
        return label_ids

    def get_alias_list(self, text: str, topk) -> list[Alias]:
        """将文本转换为embedding，然后利用faiss查找，将匹配结果的
        (name, code)返回"""
        q = np.array(get_embedding(text), dtype=np.float32).reshape(1, -1)
        _, labels = self.index.search(q, topk)
        label_ids: list[int] = labels[0].tolist()
        return [self.db.get_by_embedding_id(i) for i in label_ids]

    def close(self) -> None:
        self.db.close()


if __name__ == "__main__":
    gnd_file = "data/shared-task-datasets/GND/dataset/GND-Subjects-all.json"
    initialize(gnd_file, Path("./db/alias/all"))
    print("DONE")
