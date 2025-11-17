import uuid
from datetime import datetime

from ace.core.schema import Bullet, DeltaOp, Playbook


class PlaybookManager:
    def __init__(self):
        self.playbook = Playbook(version=0)

    def load_playbook(self, path: str) -> Playbook:
        """Load playbook from disk. To be implemented."""
        raise NotImplementedError("Persistence not yet implemented")

    def save_playbook(self, path: str):
        """Save playbook to disk. To be implemented."""
        raise NotImplementedError("Persistence not yet implemented")

    def _find_bullet(self, bullet_id: str) -> Bullet:
        """Find a bullet by ID."""
        for bullet in self.playbook.bullets:
            if bullet.id == bullet_id:
                return bullet
        raise ValueError(f"Bullet not found: {bullet_id}")

    def apply_delta(self, delta: DeltaOp):
        """Apply a delta operation to modify playbook state."""
        if delta.op == "ADD":
            if not delta.new_bullet:
                raise ValueError("'ADD' operation requires new_bullet")

            if "id" in delta.new_bullet:
                bullet_id = delta.new_bullet["id"]
                for existing_bullet in self.playbook.bullets:
                    if existing_bullet.id == bullet_id:
                        # No-op: bullet already exists (idempotent replay)
                        return
            else:
                bullet_id = f"{delta.new_bullet['section'][:4]}-{str(uuid.uuid4())[:5]}"

            bullet = Bullet(
                id=bullet_id,
                section=delta.new_bullet["section"],
                content=delta.new_bullet["content"],
                tags=delta.new_bullet.get("tags", []),
                added_at=datetime.utcnow(),
            )
            self.playbook.bullets.append(bullet)
            self.playbook.version += 1

        elif delta.op == "PATCH":
            if not delta.target_id or not delta.patch:
                raise ValueError("'PATCH' operation requires target_id and patch")

            bullet = self._find_bullet(delta.target_id)
            bullet.content = delta.patch
            self.playbook.version += 1

        elif delta.op == "INCR_HELPFUL":
            if not delta.target_id:
                raise ValueError("'INCR_HELPFUL' operation requires target_id")

            bullet = self._find_bullet(delta.target_id)
            bullet.helpful += 1
            bullet.last_used = datetime.utcnow()
            self.playbook.version += 1

        elif delta.op == "INCR_HARMFUL":
            if not delta.target_id:
                raise ValueError("'INCR_HARMFUL' operation requires target_id")

            bullet = self._find_bullet(delta.target_id)
            bullet.harmful += 1
            self.playbook.version += 1

        elif delta.op == "DEPRECATE":
            if not delta.target_id:
                raise ValueError("'DEPRECATE' operation requires target_id")

            bullet = self._find_bullet(delta.target_id)
            self.playbook.bullets.remove(bullet)
            self.playbook.version += 1

        else:
            raise ValueError(f"Invalid operation: {delta.op}")
