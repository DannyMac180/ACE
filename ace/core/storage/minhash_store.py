import pickle

from datasketch import MinHash

from .db import DatabaseConnection


class MinHashStore:
    def __init__(self, db_conn: DatabaseConnection, num_perm: int = 128):
        self.db = db_conn
        self.num_perm = num_perm

    def generate_signature(self, text: str) -> bytes:
        m = MinHash(num_perm=self.num_perm)
        for word in text.split():
            m.update(word.encode("utf8"))
        return pickle.dumps(m)

    def add_signature(self, bullet_id: str, text: str):
        sig = self.generate_signature(text)
        self.db.execute(
            "INSERT OR REPLACE INTO minhash_sigs (bullet_id, signature) VALUES (?, ?)",
            (bullet_id, sig),
        )

    def get_signature(self, bullet_id: str) -> MinHash:
        rows = self.db.fetchall(
            "SELECT signature FROM minhash_sigs WHERE bullet_id = ?", (bullet_id,)
        )
        if not rows:
            return None
        return pickle.loads(rows[0][0])

    def compute_jaccard(self, sig1: MinHash, sig2: MinHash) -> float:
        return sig1.jaccard(sig2)

    def remove_signature(self, bullet_id: str):
        self.db.execute("DELETE FROM minhash_sigs WHERE bullet_id = ?", (bullet_id,))
