"""存放TIBKAT数据的sqlite数据库，方便存取"""

from llms4subjects.sqlite import SqliteDb
from dataclasses import dataclass
from sqlite3 import Row


@dataclass
class Record:
    id: str
    title: str
    abstract: str
    embedding: str
    gnd_ids: list[str]
    source:str

    @classmethod
    def from_row(cls, row: Row) -> "Record":
        """从sqlite3.Row对象创建WRecord实例"""
        return cls(
            id=row["id"],
            title=row["title"],
            abstract=row["abstract"],
            embedding=row["embedding"],
            gnd_ids=row["gnd_ids"].split(","),
            source=row["source"]
        )


class TibkatDb(SqliteDb):
    def __init__(self, db_file: str="TIBKAT.db"):
        SqliteDb.__init__(self, db_file)
        self.table_name = "record"
        self.create_table(f"""CREATE TABLE IF NOT EXISTS {self.table_name} (  
            id TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            abstract TEXT,
            embedding TEXT,
            gnd_ids TEXT NOT NULL
        );""")

    def insert_one(
        self,
        id: str,
        title: str,
        abstract: str,
        embedding: str,
        gnd_ids: list[str],
        source:str
    ) -> int:
        sql = f"""INSERT INTO {self.table_name} (id, title, abstract, embedding, gnd_ids, source)   VALUES (?, ?, ?, ?, ?, ?)"""

        return self.insert(
            sql, (id, title, abstract, embedding, ",".join(gnd_ids), source)
        )

    def select(self, ids:list[str]) -> list[Record]:
        # 加上引号， 拼接成'id1', 'id2'的形式
        ids = ",".join([f"'{e}'" for e in ids])
        sql = f"""SELECT * from {self.table_name} WHERE id in ({ids})"""
        return [Record.from_row(row) for row in self.query(sql=sql)]