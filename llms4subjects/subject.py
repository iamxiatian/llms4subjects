"""
GND subject信息，存放在db/subjuect/core或者db/subject/all目录下，目录下存在以下文件：

- subject.sqlite 将subject的信息，保存到sqlite，方便观察和查询
- name_code.jsonline 名称和code对应的文件
- embedding.txt 根据name-code.jsonline中的name，生成的embedding值
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
from llms4subjects.translate import translate_by_llm

GND_subjects_all_file = (
    "data/shared-task-datasets/GND/dataset/GND-Subjects-all.json"
)

GND_subjects_core_file = (
    "data/shared-task-datasets/GND/dataset/GND-Subjects-tib-core.json"
)


@dataclass
class Subject:
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
            code TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            cls_num TEXT,
            cls_name TEXT,
            alternate_names TEXT,
            related_subjects TEXT,
            source TEXT,
            definition TEXT
        );""")

        self.create_table("""CREATE TABLE IF NOT EXISTS namecode (  
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            code TEXT NOT NULL
        );""")

    def insert_subject(
        self,
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
        sql = """INSERT INTO subject (code, name, cls_num, cls_name, alternate_names, related_subjects, source, definition)   VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""
        value = (
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

    def insert_name_code_id(
        self,
        my_id: int,
        name: str,
        code: str,
    ) -> int:
        sql = "INSERT INTO namecode (id, name, code) VALUES (?, ?, ?)"
        self.insert(sql, (my_id, name, code))


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

    existed_names = set()
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
            code,
            name,
            cls_num,
            cls_name,
            alternate_names,
            related_subjects,
            source,
            definition,
        )

        # insert embedding
        alternate_names.insert(0, name)
        for n in alternate_names:
            if n not in existed_names:
                existed_names.add(n)

                name_code_file.write(f"{n}\n")

                embedding = get_embedding(n)
                embedding_file.write(",".join([str(e) for e in embedding]))
                embedding_file.write("\n")

                value = np.array(embedding, dtype=np.float32).reshape(
                    1, EMBEDDING_DIM
                )
                index.add(value)

                db.insert_name_code_id(embedding_id, code, n)
                embedding_id += 1
    faiss.write_index(index, Path(db_home, "embedding.idx").as_posix())
    name_code_file.close()
    embedding_file.close()
    db.close()


if __name__ == "__main__":
    # translate_names()
    # initialize(GND_subjects_core_file, Path("./db/subject/core"))
    initialize(GND_subjects_all_file, Path("./db/subject/all"))
