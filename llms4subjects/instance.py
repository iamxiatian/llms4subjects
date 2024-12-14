"""
TIBKAT实例数据处理，存放在db/instance/core或者db/instance/all目录下，目录下
存在以下文件：

- instance.sqlite 将instance的信息，保存到sqlite，方便观察和查询
- instance.jsonline 从训练目录下的每一个jsonld文件中抽取基本信息加以保存
- embedding.txt 根据Instance的名称和摘要，生成的embedding值
- embedding.idx 根据embedding.txt，生成的FAISS索引
- dev.jsonline 从dev目录下抽取基本信息加以保存


其中，instance.jsonline文件中的字段信息包含了从jsonld文件中抽取出的title、abstract、gnd_ids三个字段，已经从文件名称中抽取得到的id字段。即如下四个字段：

- id(str)
- title(str)
- abstract(str)
- gnd_ids(list)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from sqlite3 import Row

import faiss
import numpy as np
from pyld import jsonld
from tqdm import tqdm

from llms4subjects.api import EMBEDDING_DIM, get_embedding
from llms4subjects.sqlite import SqliteDb


def _parse_jsonld(jsonld_file: Path) -> dict | None:
    tibkat_id = jsonld_file.stem
    language = jsonld_file.parent.name
    doctype = jsonld_file.parent.parent.name
    # 读取JSON-LD文件
    with open(jsonld_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # 使用pyld库加载和展开JSON-LD
    expanded_data = jsonld.expand(data)
    for e in expanded_data[0]["@graph"]:
        my_id = e["@id"]
        if "https://www.tib.eu/de/suchen/id/TIBKAT" in my_id:
            title = e["http://purl.org/dc/elements/1.1/title"][0]["@value"]
            abstract = e["http://purl.org/dc/terms/abstract"][0]["@value"]
            gnd_codes = e["http://purl.org/dc/terms/subject"]
            entry = {
                "id": tibkat_id,
                "title": title,
                "abstract": abstract,
                "gnd_codes": gnd_codes,
                "language": language,
                "doctype": doctype,
            }
            return entry


def gen_jsonline_file(instance_dir: str, out_jsonline_file: str) -> None:
    """把文件夹instance_dir下的所有的jsonld文件，读取出其中的标题和摘要，
    作为jsonline文件的一行，进行保存，形成一个完整的jsonline文件。"""
    with open(out_jsonline_file, "w", encoding="utf-8") as f:
        train_files = list(Path(instance_dir).glob("**/*.jsonld"))
        for jsonld_file in tqdm(train_files):
            entry = _parse_jsonld(jsonld_file)
            json_record = json.dumps(entry, ensure_ascii=False)
            f.write(json_record + "\n")


@dataclass
class Instance:
    embedding_id: int
    instance_id: str
    title: str
    abstract: str
    gnd_codes: list[str]
    language: str
    doctype: str

    @classmethod
    def from_row(cls, row: Row) -> "Instance":
        """从sqlite3.Row对象创建实例"""
        return cls(
            embedding_id=row["embedding_id"],
            instance_id=row["instance_id"],
            title=row["title"],
            abstract=row["abstract"],
            gnd_codes=row["gnd_codes"].split(","),
            language=row["language"],
            doctype=row["doctype"],
        )


class InstanceDb(SqliteDb):
    """存放实例数据和实例与主题code映射关系的数据库，为方便和embedding对齐，
    instance表中的embedding_id从0开始编号"""

    def __init__(self, db_file: str):
        SqliteDb.__init__(self, db_file)
        self.create_table("""CREATE TABLE IF NOT EXISTS instance (  
            embedding_id INTEGER PRIMARY KEY,              
            instance_id TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            abstract TEXT,
            gnd_codes TEXT NOT NULL,
            language TEXT NOT NULL,
            doctype TEXT NOT NULL
        );""")

        self.create_table("""CREATE TABLE IF NOT EXISTS mapping (  
            seq_id INTEGER PRIMARY KEY,              
            instance_id TEXT NOT NULL,
            gnd_code TEXT NOT NULL,
            language TEXT NOT NULL,
            doctype TEXT NOT NULL
        );""")

    def insert_instance(
        self,
        embedding_id: int,
        instance_id: str,
        title: str,
        abstract: str,
        gnd_codes: list[str],
        language: str,
        doctype: str,
    ) -> int:
        sql = """INSERT INTO instance (embedding_id, instance_id, title, abstract, gnd_codes, language, doctype)   VALUES (?, ?, ?, ?, ?, ?, ?)"""

        value = (
            embedding_id,
            instance_id,
            title,
            abstract,
            ",".join(gnd_codes),
            language,
            doctype,
        )
        return self.insert(sql, value)

    def insert_mapping(
        self,
        seq_id: int,
        instance_id: str,
        gnd_code: str,
        language: str,
        doctype: str,
    ) -> int:
        sql = """INSERT INTO mapping (seq_id, instance_id, gnd_code, language, doctype)  VALUES (?, ?, ?, ?, ?)"""

        return self.insert(
            sql,
            (
                seq_id,
                instance_id,
                gnd_code,
                language,
                doctype,
            ),
        )

    def get_by_instance_ids(self, instance_ids: list[str]) -> list[Instance]:
        # 加上引号， 拼接成'id1', 'id2'的形式
        ids_str = ",".join([f"'{e}'" for e in instance_ids])
        sql = f"""SELECT * from instance WHERE instance_id in ({ids_str})"""
        return [Instance.from_row(row) for row in self.query(sql=sql)]

    def get_by_embedding_id(self, embedding_id: int) -> Instance:
        sql = "SELECT * from instance WHERE embedding_id = ?"
        rows = self.query(sql=sql, parameters=(embedding_id,))
        return Instance.from_row(rows[0])

    @classmethod
    def open_core(cls) -> "InstanceDb":
        db = InstanceDb("./db/instance/core/subject.sqlite")
        return db

    @classmethod
    def open_all(cls) -> "InstanceDb":
        db = InstanceDb("./db/instance/all/subject.sqlite")
        return db


def initialize(
    train_jsonld_dir: str, dev_jsonld_dir: str, db_home: Path
) -> None:
    """初始化，生成embedding等文件"""
    db_home.mkdir(parents=True, exist_ok=True)
    db = InstanceDb(Path(db_home, "instance.sqlite").as_posix())
    jsonline_file = Path(db_home, "instance.jsonline")

    # 将开发集文件保存成jsonline的形式
    gen_jsonline_file(dev_jsonld_dir, Path(db_home, "dev.jsonline").as_posix())

    # 将样本文件保存成jsonline的形式
    gen_jsonline_file(train_jsonld_dir, jsonline_file.as_posix())

    embedding_writer = Path(db_home, "embedding.txt").open(
        "w", encoding="utf-8"
    )
    # 使用Inner Product (IP) 距离的IndexFlat
    index: faiss.IndexFlatIP = faiss.IndexFlatIP(EMBEDDING_DIM)

    embedding_id = 0
    seq_id = 1  # instance和gnd_code的映射关系
    jsonline_reader = jsonline_file.open("r", encoding="utf-8")
    for line in tqdm(jsonline_reader.readlines()):
        instance = json.loads(line)
        tibkat_id, title, abstract, gnd_codes, language, doctype = (
            instance["id"],
            instance["title"],
            instance["abstract"],
            instance["gnd_codes"],
            instance["language"],
            instance["doctype"],
        )

        # 读出gnd_codes, 并保证gnd code的开始一定为"gnd:"
        # @id的内容形式：http://d-nb.info/gnd/4028944-8，需要最右侧截取code
        gnd_codes = [e["@id"].rsplit("/", 1)[-1] for e in gnd_codes]
        gnd_codes = [f"gnd:{c}" for c in gnd_codes]

        db.insert_instance(
            embedding_id,
            tibkat_id,
            title,
            abstract,
            gnd_codes,
            language,
            doctype,
        )
        embedding_id += 1

        for code in gnd_codes:
            db.insert_mapping(seq_id, tibkat_id, code, language, doctype)
            seq_id += 1

        text = f"""title: "{title}"\n abstract: {abstract}"""
        embedding = get_embedding(text)
        embedding_writer.write(",".join([str(e) for e in embedding]))
        embedding_writer.write("\n")

        value = np.array(embedding, dtype=np.float32).reshape(1, EMBEDDING_DIM)
        index.add(value)
    faiss.write_index(index, Path(db_home, "embedding.idx").as_posix())
    embedding_writer.close()
    jsonline_reader.close()
    db.close()


class EmbeddingQuery:
    def __init__(self, db_path: Path):
        """读取已经利用FAISS索引的数据文件以及对应的id文件"""
        db_file = Path(db_path, "instance.sqlite").as_posix()
        idx_file = Path(db_path, "embedding.idx").as_posix()
        self.db = InstanceDb(db_file)
        self.index: faiss.IndexFlatIP = faiss.read_index(idx_file)

    def get_embedding_ids(self, text: str, topk) -> list[int]:
        """将文本转换为embedding，然后利用faiss查找，将匹配结果的序号返回"""
        q = np.array(get_embedding(text), dtype=np.float32).reshape(1, -1)
        _, labels = self.index.search(q, topk)
        label_ids: list[int] = labels[0].tolist()
        return label_ids

    def get_instances(self, text: str, topk) -> list[Instance]:
        """将文本转换为embedding，然后利用faiss查找，将匹配结果的序号返回"""
        q = np.array(get_embedding(text), dtype=np.float32).reshape(1, -1)
        _, labels = self.index.search(q, topk)
        label_ids: list[int] = labels[0].tolist()
        instances = [self.db.get_by_embedding_id(i) for i in label_ids]
        return instances

    def close(self) -> None:
        self.db.close()


if __name__ == "__main__":
    # initialize(
    #     "./data/shared-task-datasets/TIBKAT/tib-core-subjects/data/train/",
    #     "./data/shared-task-datasets/TIBKAT/tib-core-subjects/data/dev/",
    #     Path("./db/instance/core"),
    # )

    # initialize(
    #     "./data/shared-task-datasets/TIBKAT/all-subjects/data/train/",
    #     "./data/shared-task-datasets/TIBKAT/all-subjects/data/dev/",
    #     Path("./db/instance/all"),
    # )
    print("DONE.")
