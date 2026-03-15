import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from ace.core.schema import Bullet, DeltaOp, Playbook


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(UTC)


class PlaybookManager:
    def __init__(self):
        self.playbook = Playbook(version=0)

    def load_playbook(self, path: str) -> Playbook:
        """Load a playbook from a JSON file and replace the in-memory state."""
        payload = Path(path).read_text(encoding="utf-8")
        self.playbook = Playbook.model_validate_json(payload)
        return self.playbook

    def save_playbook(self, path: str) -> None:
        """Persist the current playbook to a JSON file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.playbook.model_dump(mode="json"), indent=2)
        output_path.write_text(payload + "\n", encoding="utf-8")

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
                added_at=_utcnow(),
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
            bullet.last_used = _utcnow()
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
