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
import logging
from dataclasses import dataclass
from pathlib import Path
from sqlite3 import Row

import faiss
import numpy as np
from tqdm import tqdm

from llms4subjects.api import EMBEDDING_DIM, get_embedding
from llms4subjects.sqlite import SqliteDb
from llms4subjects.prompt import to_alpaca_entry
from llms4subjects.parse import parse_jsonld

logger = logging.getLogger(__name__)


def gen_jsonline_file(
    instance_dirs: list[str], out_jsonline_file: str
) -> None:
    """把文件夹instance_dir下的所有的jsonld文件，读取出其中的标题和摘要，
    作为jsonline文件的一行，进行保存，形成一个完整的jsonline文件。"""
    with open(out_jsonline_file, "w", encoding="utf-8") as f:
        for instance_dir in instance_dirs:
            train_files = list(Path(instance_dir).glob("**/*.jsonld"))
            for jsonld_file in tqdm(train_files):
                entry = parse_jsonld(jsonld_file)
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
    
    def get_by_instance_id(self, instance_id: str) -> Instance:
        # 加上引号， 拼接成'id1', 'id2'的形式
        sql = "SELECT * from instance WHERE instance_id =? "
        rows = self.query(sql=sql, parameters=(instance_id,))
        if not rows:
            total = self.tatal("instance")
            message = f"no instance_id {instance_id} in {self.db_file}, \
                total records: {total}"
            raise Exception(message)
        return Instance.from_row(rows[0])

    def get_by_embedding_id(self, embedding_id: int) -> Instance:
        sql = "SELECT * from instance WHERE embedding_id = ?"
        rows = self.query(sql=sql, parameters=(embedding_id,))
        if not rows:
            total = self.tatal("instance")
            message = f"no embedding_id {embedding_id} in {self.db_file}, \
                total records: {total}"
            raise Exception(message)
        return Instance.from_row(rows[0])

    def num(self) -> int:
        return self.total("instance")

    def to_alpaca(self, json_file: str):
        """将instance转换为alpaca格式的LLM训练文件"""
        from llms4subjects.subject import subject_db_all

        sql = "SELECT * from instance"
        rows = self.query(sql=sql, parameters=())
        entries = []
        for row in tqdm(rows):
            title, abstract = row["title"], row["abstract"]
            codes = row["gnd_codes"].split(",")
            subjects = [subject_db_all.get_name_by_code(c) for c in codes]

            entry = to_alpaca_entry(title, abstract, subjects)
            entries.append(entry)

        # 写入文件
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        n = len(entries)
        logger.info(f"convert {n} instance to alpaca format.")

    @classmethod
    def open_core(cls) -> "InstanceDb":
        db = InstanceDb("./db/instance/core/instance.sqlite")
        return db

    @classmethod
    def open_all(cls) -> "InstanceDb":
        db = InstanceDb("./db/instance/all/instance.sqlite")
        return db

    @classmethod
    def open_merged_no_dev(cls) -> "InstanceDb":
        db = InstanceDb("./db/instance/merged_no_dev/instance.sqlite")
        return db

    @classmethod
    def open_merged_with_dev(cls) -> "InstanceDb":
        db = InstanceDb("./db/instance/merged_with_dev/instance.sqlite")
        return db


def initialize(
    train_jsonld_dir: str, dev_jsonld_dir: str, db_home: Path
) -> None:
    """初始化，生成embedding等文件"""
    db_home.mkdir(parents=True, exist_ok=True)
    db = InstanceDb(Path(db_home, "instance.sqlite").as_posix())
    jsonline_file = Path(db_home, "instance.jsonline")

    # 将开发集文件保存成jsonline的形式
    gen_jsonline_file(
        [dev_jsonld_dir], Path(db_home, "dev.jsonline").as_posix()
    )

    # 将样本文件保存成jsonline的形式
    gen_jsonline_file([train_jsonld_dir], jsonline_file.as_posix())

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


def init_for_test(
    train_jsonld_dir: str, dev_jsonld_dir: str, db_home: Path
) -> None:
    """初始化，生成embedding等文件，把train和dev数据合并到一起以充分利用数据"""
    db_home.mkdir(parents=True, exist_ok=True)
    db = InstanceDb(Path(db_home, "instance.sqlite").as_posix())
    jsonline_file = Path(db_home, "instance.jsonline")

    # 将样本文件和开发集文件保存成jsonline的形式
    gen_jsonline_file(
        [train_jsonld_dir, dev_jsonld_dir], jsonline_file.as_posix()
    )

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
    def __init__(self, db_path: Path, db: InstanceDb):
        """读取已经利用FAISS索引的数据文件以及对应的id文件"""
        idx_file = Path(db_path, "embedding.idx").as_posix()
        self.db = db
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


instance_db_all = InstanceDb.open_all()
instance_db_core = InstanceDb.open_core()
instance_db_merged_no_dev = InstanceDb.open_merged_no_dev()
instance_db_merged_with_dev = InstanceDb.open_merged_with_dev()


instance_eq_all = EmbeddingQuery(Path("./db/instance/all"), instance_db_all)
instance_eq_core = EmbeddingQuery(Path("./db/instance/core"), instance_db_core)
instance_eq_merged_no_dev = EmbeddingQuery(
    Path("./db/instance/merged_no_dev"), instance_db_merged_no_dev
)
instance_eq_merged_with_dev = EmbeddingQuery(
    Path("./db/instance/merged_with_dev"), instance_db_merged_with_dev
)


def get_instance_db(dataset_type: str) -> InstanceDb:
    if dataset_type == "core":
        return instance_db_core
    elif dataset_type == "all":
        return instance_db_all
    elif dataset_type == "merged_no_dev":
        return instance_db_merged_no_dev
    else:
        return instance_db_merged_with_dev


def get_embedding_query(dataset_type: str) -> EmbeddingQuery:
    if dataset_type == "core":
        return instance_eq_core
    elif dataset_type == "all":
        return instance_eq_all
    elif dataset_type == "merged_no_dev":
        return instance_eq_merged_no_dev
    else:
        return instance_eq_merged_with_dev


def load_jsonline_file(jsonline_file: str) -> list[dict]:
    from llms4subjects.subject import subject_db_all as subject_db

    dataset = []
    with open(jsonline_file, "r", encoding="utf-8") as dev_f:
        for line in tqdm(dev_f.readlines()):
            record = json.loads(line)
            true_codes = [
                c["@id"].rsplit("/", 1)[-1] for c in record["gnd_codes"]
            ]
            true_codes = [f"gnd:{c}" for c in true_codes]
            true_names = [subject_db.get_name_by_code(c) for c in true_codes]
            dataset.append(
                {
                    "id": record["id"],
                    "title": record["title"],
                    "abstract": record["abstract"],
                    "gnd_codes": true_codes,
                    "gnd_names": true_names,
                }
            )
    return dataset


def merge_all_to_alpaca(out_json_file: str):
    """把merged下所有的文件，都转成alpaca"""
    home = Path("./db/instance/merged_no_dev")
    jsonline_file = Path(home, "instance.jsonline")
    dev_file = Path(home, "dev.jsonline")

    ds = load_jsonline_file(jsonline_file.as_posix())
    ds2 = load_jsonline_file(dev_file.as_posix())
    ds.extend(ds2)
    entries = []
    for row in tqdm(ds):
        title, abstract = row["title"], row["abstract"]
        subjects = row["gnd_names"]

        entry = to_alpaca_entry(title, abstract, subjects)
        entries.append(entry)

    # 写入文件
    with open(out_json_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    n = len(entries)
    logger.info(f"convert {n} instance to alpaca format.")


if __name__ == "__main__":
    # initialize(
    #     "./data/shared-task-datasets/TIBKAT/tib-core-subjects/data/train/",
    #     "./data/shared-task-datasets/TIBKAT/tib-core-subjects/data/dev/",
    #     Path("./db/instance/core"),
    # )

    # initialize(
    #     "./data/shared-task-datasets/TIBKAT/merged-subjects/data/train/",
    #     "./data/shared-task-datasets/TIBKAT/merged-subjects/data/dev/",
    #     Path("./db/instance/merged"),
    # )
    # gen_jsonline_file(
    #     "./data/shared-task-datasets/TIBKAT/merged-subjects/data/dev2/",
    #     "./db/instance/merged/dev2.jsonline",
    # )

    # 把训练集和测试集合并到一起
    init_for_test(
        "./data/shared-task-datasets/TIBKAT/merged-subjects/data/train/",
        "./data/shared-task-datasets/TIBKAT/merged-subjects/data/dev2/",
        Path("./db/instance/merged_with_dev"),
    )
    print("DONE.")
