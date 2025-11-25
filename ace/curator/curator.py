# ace/curator/curator.py
from ace.core.schema import Bullet, Delta, DeltaOp
from ace.reflector.schema import Reflection

from .semantic_matcher import SemanticMatcher


def curate(
    reflection: Reflection,
    existing_bullets: list[Bullet] | None = None,
    threshold: float = 0.90,
) -> Delta:
    """
    Convert a Reflection object into a Delta object containing operations to update the Playbook.

    This function is deterministic and rule-based:
    - For each BulletTag in reflection.bullet_tags, generate an INCR_HELPFUL or
      INCR_HARMFUL operation
    - For each CandidateBullet in reflection.candidate_bullets:
      - If a semantically similar bullet exists in existing_bullets, generate a PATCH operation
      - Otherwise, generate an ADD operation

    Per the ACE paper: "The Curator then synthesizes these lessons into compact delta entries...
    Identify what's genuinely new" - we check for duplicates BEFORE emitting ADD or PATCH.

    Args:
        reflection: A Reflection object from the Reflector
        existing_bullets: Optional list of existing Bullet objects to check for duplicates.
                         If None, all candidates become ADD operations (legacy behavior).
        threshold: Cosine similarity threshold for duplicate detection (default: 0.90)

    Returns:
        Delta: A Delta object containing the list of operations to apply to the Playbook
    """
    ops = []

    # Process bullet tags (helpful/harmful feedback on existing bullets)
    for bullet_tag in reflection.bullet_tags:
        if bullet_tag.tag == "helpful":
            ops.append(DeltaOp(op="INCR_HELPFUL", target_id=bullet_tag.id))
        elif bullet_tag.tag == "harmful":
            ops.append(DeltaOp(op="INCR_HARMFUL", target_id=bullet_tag.id))

    # Process candidate bullets with semantic duplicate detection
    if existing_bullets is not None:
        matcher = SemanticMatcher(threshold=threshold)

        for candidate in reflection.candidate_bullets:
            duplicate = matcher.find_duplicate(candidate.content, existing_bullets)

            if duplicate is not None:
                ops.append(
                    DeltaOp(
                        op="PATCH",
                        target_id=duplicate.id,
                        patch=candidate.content,
                    )
                )
            else:
                ops.append(
                    DeltaOp(
                        op="ADD",
                        new_bullet={
                            "section": candidate.section,
                            "content": candidate.content,
                            "tags": candidate.tags,
                        },
                    )
                )
    else:
        # Legacy behavior: no dedup check, all candidates become ADD
        for candidate in reflection.candidate_bullets:
            ops.append(
                DeltaOp(
                    op="ADD",
                    new_bullet={
                        "section": candidate.section,
                        "content": candidate.content,
                        "tags": candidate.tags,
                    },
                )
            )

    return Delta(ops=ops)
