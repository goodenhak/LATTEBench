import sqlite3
import json
import time
from typing import List, Dict, Any

class ScoreStore:
    def __init__(self, db_path: str = "history.db"):
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        """创建数据库表，包含id、metadata、thought、score和structure字段"""
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS plans (
            id TEXT PRIMARY KEY,
            metadata TEXT,
            thought TEXT,
            score REAL,
            structure TEXT  -- 存储JSON序列化后的字符串列表
        )
        """)
        self.conn.commit()

    def clear_table_data(self, db_path, table_name):
        """清空指定表的数据"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"DELETE FROM {table_name};")
            conn.commit()
            print(f"表 {table_name} 的数据已清空")
            
        except sqlite3.Error as e:
            print(f"清空数据时出错: {e}")
            conn.rollback()
        finally:
            conn.close()

    def add(self, metadata: Dict[str, Any], thought: List[Dict[str, Any]], score: float, structure: List[str] = None) -> str:
        """
        存储metadata(dict)、thought(dict list)、score和structure，并返回它的id(秒级时间戳)
        
        参数:
            metadata: 元数据字典
            thought: 思考过程字典列表
            score: 分数
            structure: 结构信息字符串列表
            
        返回:
            记录的ID（时间戳）
        """
        ts = str(int(time.time()))  # 秒级时间戳
        
        # 将dict和dict list序列化为JSON字符串
        metadata_json = json.dumps(metadata)
        thought_json = json.dumps(thought)
        # 将字符串列表序列化为JSON字符串
        structure_json = json.dumps(structure) if structure is not None else "[]"

        try:
            self.cur.execute(
                "INSERT INTO plans (id, metadata, thought, score, structure) VALUES (?, ?, ?, ?, ?)",
                (ts, metadata_json, thought_json, score, structure_json)
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            # 如果该秒已经存在一条数据，可以在id后追加一个自增后缀
            ts = ts + "_" + str(int(time.time() * 1000) % 1000)  # 追加毫秒后缀
            self.cur.execute(
                "INSERT INTO plans (id, metadata, thought, score, structure) VALUES (?, ?, ?, ?, ?)",
                (ts, metadata_json, thought_json, score, structure_json)
            )
            self.conn.commit()
        return ts

    def top_k(self, k: int = 10):
        """
        返回按score排序的前k个记录，metadata、thought和structure已反序列化
        
        参数:
            k: 返回的记录数量
            
        返回:
            包含(metadata, thought, score, structure)的元组列表
        """
        self.cur.execute(
            "SELECT metadata, thought, score, structure FROM plans ORDER BY score DESC LIMIT ?", (k,)
        )
        rows = self.cur.fetchall()
        # 反序列化JSON字符串为原始数据类型
        rpns = []
        scores = []
        thoughts = []
        for row in rows:
            metadata, thought, score, structure = row
            thoughts.append(json.loads(thought))
            rpns.append(json.loads(structure))
            scores.append(score)
        return thoughts,rpns,scores

    def load(self, id_: str):
        """
        根据id加载记录，metadata、thought和structure已反序列化
        
        参数:
            id_: 记录ID
            
        返回:
            包含(metadata, thought, score, structure)的元组，如果不存在则返回None
        """
        self.cur.execute("SELECT metadata, thought, score, structure FROM plans WHERE id=?", (id_,))
        row = self.cur.fetchone()
        if row:
            metadata, thought, score, structure = row
            return (
                json.loads(metadata), 
                json.loads(thought), 
                score, 
                json.loads(structure)  # 反序列化为字符串列表
            )
        return None

    def all(self):
        """
        返回所有记录，按score降序，metadata、thought和structure已反序列化
        
        返回:
            包含(metadata, thought, score, structure)的元组列表
        """
        self.cur.execute("SELECT metadata, thought, score, structure FROM plans ORDER BY score DESC")
        rows = self.cur.fetchall()
        # 反序列化JSON字符串为原始数据类型
        result = []
        for row in rows:
            metadata, thought, score, structure = row
            result.append((
                json.loads(metadata), 
                json.loads(thought), 
                score, 
                json.loads(structure)  # 反序列化为字符串列表
            ))
        return result

    def close(self):
        """关闭数据库连接"""
        self.conn.close()