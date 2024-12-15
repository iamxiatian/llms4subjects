"""
GND subject信息，存放在db/subjuect/core或者db/subject/all目录下，目录下存在以下文件：

- subject.sqlite 将subject的信息，保存到sqlite，方便观察和查询，同时记录了embedding_id和名称与code的对应关系。
- name_code.jsonline embedding_id、名称和code对应的文件
- embedding.txt 根据name-code.jsonline中的name，生成的embedding值
- embedding.idx 根据embedding.txt，生成的FAISS索引
"""

import json
from dataclasses import dataclass
from pathlib import Path
from sqlite3 import Row
import textwrap
import faiss
import numpy as np
from tqdm import tqdm

from llms4subjects.api import EMBEDDING_DIM, get_embedding
from llms4subjects.sqlite import SqliteDb
from llms4subjects.translate import translate_by_llm

GND_subjects_all_file = (
    "data/shared-task-datasets/GND/dataset/GND-Subjects-all.json"
)

GND_subjects_core_file = (
    "data/shared-task-datasets/GND/dataset/GND-Subjects-tib-core.json"
)


@dataclass
class Subject:
    embedding_id: int
    code: str
    name: str
    cls_num: str
    cls_name: str
    alternate_names: list[str]
    related_subjects: list[str]
    source: str
    definition: str

    @classmethod
    def from_row(cls, row: Row) -> "Subject":
        """从sqlite3.Row对象创建实例"""
        return cls(
            embedding_id=row["embedding_id"],
            code=row["code"],
            name=row["name"],
            cls_num=row["cls_num"],
            cls_name=row["cls_name"],
            alternate_names=row["alternate_names"].split("\n"),
            related_subjects=row["related_subjects"].split("\n"),
            source=row["source"],
            definition=row["definition"],
        )


class SubjectDb(SqliteDb):
    def __init__(self, db_file: str):
        SqliteDb.__init__(self, db_file)
        self.create_table("""CREATE TABLE IF NOT EXISTS subject (  
            embedding_id INTEGER PRIMARY KEY,
            code TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            cls_num TEXT,
            cls_name TEXT,
            alternate_names TEXT,
            related_subjects TEXT,
            source TEXT,
            definition TEXT
        );""")

        # self.create_table("""CREATE TABLE IF NOT EXISTS embedding (
        #     embedding_id INTEGER PRIMARY KEY,
        #     name TEXT NOT NULL,
        #     code TEXT NOT NULL
        # );""")

    @classmethod
    def open_core(cls) -> "SubjectDb":
        db = SubjectDb("./db/subject/core/subject.sqlite")
        return db

    @classmethod
    def open_all(cls) -> "SubjectDb":
        db = SubjectDb("./db/subject/all/subject.sqlite")
        return db

    def insert_subject(
        self,
        embedding_id: int,
        code: str,
        name: str,
        cls_num: str,
        cls_name: str,
        alternate_names: list[str],
        related_subjects: list[str],
        source: str,
        definition: str,
    ) -> int:
        #  insert subject
        sql = """INSERT INTO subject (embedding_id, code, name, cls_num, cls_name, alternate_names, related_subjects, source, definition)  
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        value = (
            embedding_id,
            code,
            name,
            cls_num,
            cls_name,
            "\n".join(alternate_names),
            "\n".join(related_subjects),
            source,
            definition,
        )
        return self.insert(sql, value)

    def get_subject_by_code(self, code: str) -> Subject:
        sql = "SELECT * from subject WHERE code = ?"
        rows = self.query(sql=sql, parameters=(code,))
        if not rows:
            print(f"Error: no subject for code: {code}")
        return Subject.from_row(rows[0])

    def get_name_by_code(self, code: str) -> str:
        sql = "SELECT name from subject WHERE code = ?"
        rows = self.query(sql=sql, parameters=(code,))
        if not rows:
            print(f"Error: no subject for code: {code}")
        return rows[0]["name"]
    
    def get_code_by_name(self, name: str) -> str|None:
        sql = "SELECT code from subject WHERE name = ?"
        rows = self.query(sql=sql, parameters=(name,))
        if not rows:
            return None
        else:
            return rows[0]["code"]

    # def insert_name_code_id(
    #     self,
    #     embedding_id: int,
    #     name: str,
    #     code: str,
    # ) -> int:
    #     sql = (
    #         "INSERT INTO embedding (embedding_id, name, code) VALUES (?, ?, ?)"
    #     )
    #     self.insert(sql, (embedding_id, name, code))

    def get_by_embedding_id(self, embedding_id: int) -> tuple[str, str]:
        """根据embedding_id返回name和code二元组，正常二元组应该存在，因此没有判断是否为空"""
        sql = "SELECT name, code FROM subject where embedding_id = ?"
        rows = self.query(sql, (embedding_id,))
        row = rows[0]
        return row["name"], row["code"]


def translate_names(out_file: str = "name-mapping.jsonline"):
    """从GND-Subjects-all.json中读取所有的名称，把所有的名称都翻译成英语和德语两种，并保存到文件out_file中, 每一行的格式为：
    {"name": name, "EN": 英文名, "DE": 德文名}"""

    names = []
    with open(GND_subjects_all_file, "r", encoding="utf-8") as f:
        # 使用json.load()方法将文件内容解析为Python对象
        subjects_all = json.load(f)

    for entry in subjects_all:
        names.append(entry["Classification Name"])
        names.append(entry["Name"])
        if "Alternate Name" in entry:
            names.extend(entry["Alternate Name"])
        if "Related Subjects" in entry:
            names.extend(entry["Related Subjects"])

    # 先读取输出文件，避免重复处理
    with open(out_file, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f.readlines()]
        generated_names = {item["name"] for item in items}

    # 开始生成subject的英文和德文版本的名称
    names = set(names)
    with open(out_file, "a+", encoding="utf-8") as f:
        for name in tqdm(names):
            if name not in generated_names:
                en = translate_by_llm(name, "English")
                de = translate_by_llm(name, "German")
                json_record = json.dumps(
                    {"name": name, "EN": en, "DE": de}, ensure_ascii=False
                )
                f.write(json_record + "\n")
                f.flush()


def initialize(gnd_file: str, db_home: Path):
    """初始化，生成embedding等文件"""
    db_home.mkdir(parents=True, exist_ok=True)
    with open(gnd_file, "r", encoding="utf-8") as f:
        # 使用json.load()方法将文件内容解析为Python对象
        subjects: dict = json.load(f)
    db = SubjectDb(Path(db_home, "subject.sqlite").as_posix())
    name_code_file = Path(db_home, "name_code.jsonline").open(
        "w", encoding="utf-8"
    )
    embedding_file = Path(db_home, "embedding.txt").open("w", encoding="utf-8")
    # 使用Inner Product (IP) 距离的IndexFlat
    index: faiss.IndexFlatIP = faiss.IndexFlatIP(EMBEDDING_DIM)

    embedding_id = 0
    for entry in tqdm(subjects):
        code, name = entry["Code"], entry["Name"]
        source = str(entry.get("Source"))
        definition = str(entry.get("Definition"))
        cls_num = entry["Classification Number"]
        cls_name = entry["Classification Name"]
        alternate_names: list[str] = entry["Alternate Name"]
        related_subjects: list[str] = entry["Related Subjects"]

        #  insert subject
        db.insert_subject(
            embedding_id,
            code,
            name,
            cls_num,
            cls_name,
            alternate_names,
            related_subjects,
            source,
            definition,
        )

        record = json.dumps(
            {"embedding_id": embedding_id, "name": name, "code": code},
            ensure_ascii=False,
        )
        name_code_file.write(f"{record}\n")

        # insert embedding
        text = textwrap.dedent(f"""Subject:{name}
                               Related subjects: {"".join(related_subjects)}
                               Classification Name: {cls_name}""")
        embedding = get_embedding(text)
        embedding_file.write(",".join([str(e) for e in embedding]))
        embedding_file.write("\n")
        value = np.array(embedding, dtype=np.float32).reshape(1, EMBEDDING_DIM)
        index.add(value)
        embedding_id += 1

    faiss.write_index(index, Path(db_home, "embedding.idx").as_posix())
    name_code_file.close()
    embedding_file.close()
    db.close()


class EmbeddingQuery:
    def __init__(self, db_path: Path):
        """读取已经利用FAISS索引的数据文件以及对应的id文件"""
        db_file = Path(db_path, "subject.sqlite").as_posix()
        idx_file = Path(db_path, "embedding.idx").as_posix()
        self.db = SubjectDb(db_file)
        self.index: faiss.IndexFlatIP = faiss.read_index(idx_file)

    def get_embedding_ids(self, text: str, topk) -> list[int]:
        """将文本转换为embedding，然后利用faiss查找，将匹配结果的序号返回"""
        q = np.array(get_embedding(text), dtype=np.float32).reshape(1, -1)
        _, labels = self.index.search(q, topk)
        label_ids: list[int] = labels[0].tolist()
        return label_ids

    def get_namecodes_by_name(
        self, subject_name: str, topk: int
    ) -> list[tuple[str, str]]:
        """获取和subject_name相似的条目"""
        text = textwrap.dedent(f"""Subject:{subject_name}
                               Related subjects: 
                               Classification Name: """)
        return self.get_namescode_by_text(text, topk)

    def get_namescode_by_text(self, text: str, topk) -> list[tuple[str, str]]:
        """将文本转换为embedding，然后利用faiss查找，将匹配结果的
        (name, code)返回"""
        q = np.array(get_embedding(text), dtype=np.float32).reshape(1, -1)
        _, labels = self.index.search(q, topk)
        label_ids: list[int] = labels[0].tolist()
        namecodes = [self.db.get_by_embedding_id(i) for i in label_ids]
        return namecodes

    def close(self) -> None:
        self.db.close()


if __name__ == "__main__":
    # translate_names()
    initialize(GND_subjects_core_file, Path("./db/subject/core"))
    initialize(GND_subjects_all_file, Path("./db/subject/all"))
    print("DONE")
