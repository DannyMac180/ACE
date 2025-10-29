# ace/core/merge.py
from typing import Any

from .schema import Bullet, Playbook
from .storage.store_adapter import Store


class DeltaOp:
    def __init__(self, op_type: str, **kwargs):
        self.op_type = op_type
        self.data = kwargs

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DeltaOp":
        return cls(d["op"], **{k: v for k, v in d.items() if k != "op"})


class Delta:
    def __init__(self, ops: list[DeltaOp]):
        self.ops = ops

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Delta":
        return cls([DeltaOp.from_dict(op) for op in d["ops"]])


def apply_delta(playbook: Playbook, delta: Delta, store: Store) -> Playbook:
    bullets = {b.id: b for b in playbook.bullets}

    for op in delta.ops:
        if op.op_type == "ADD":
            new_bullet = Bullet(
                id=op.data["new_bullet"]["id"],
                section=op.data["new_bullet"]["section"],
                content=op.data["new_bullet"]["content"],
                tags=op.data["new_bullet"].get("tags", []),
            )
            bullets[new_bullet.id] = new_bullet
            store.save_bullet(new_bullet)
        elif op.op_type == "PATCH":
            if op.data["target_id"] in bullets:
                bullet = bullets[op.data["target_id"]]
                bullet.content = op.data["patch"]
                store.save_bullet(bullet)
        elif op.op_type == "INCR_HELPFUL":
            if op.data["target_id"] in bullets:
                bullet = bullets[op.data["target_id"]]
                bullet.helpful += 1
                store.save_bullet(bullet)
        elif op.op_type == "INCR_HARMFUL":
            if op.data["target_id"] in bullets:
                bullet = bullets[op.data["target_id"]]
                bullet.harmful += 1
                store.save_bullet(bullet)
        elif op.op_type == "DEPRECATE":
            if op.data["target_id"] in bullets:
                # For deprecate, perhaps mark as harmful or remove
                bullet = bullets[op.data["target_id"]]
                bullet.harmful += 1  # or remove
                store.save_bullet(bullet)

    new_version = playbook.version + 1
    store.set_version(new_version)
    return Playbook(version=new_version, bullets=list(bullets.values()))
