import sqlite3


class SqliteDb:
    def __init__(self, db_file: str) -> None:
        self.db_file = db_file
        self.conn = self.create_connection()

    # 数据库操作函数
    def create_connection(self) -> sqlite3.Connection:
        """创建数据库连接"""
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row  # 使得查询结果可以直接通过列名访问
        return conn

    def create_table(self, sql: str) -> None:
        """创建表"""
        cursor = self.conn.cursor()
        cursor.execute(sql)
        self.conn.commit()

    def insert(self, sql: str, parameters) -> int:
        """插入/更新记录"""
        cursor = self.conn.cursor()
        cursor.execute(sql, parameters)
        self.conn.commit()
        return cursor.lastrowid

    def query(self, sql: str, parameters=[]) -> list[sqlite3.Row]:
        cursor = self.conn.cursor()
        if not parameters:
            cursor.execute(sql)
        else:
            cursor.execute(sql, parameters)
        rows = cursor.fetchall()
        return rows

    def update(self, sql: str, parameters) -> None:
        cursor = self.conn.cursor()
        cursor.execute(sql, parameters)
        self.conn.commit()
        
    def total(self, table_name:str) -> int:
        sql = f"SELECT COUNT(*) FROM {table_name}"
        row = self.query(sql)[0]
        return row[0]

    def close(self) -> None:
        self.conn.close()
