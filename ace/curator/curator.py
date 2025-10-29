# ace/curator/curator.py
from ace.core.schema import Delta, DeltaOp
from ace.reflector.schema import Reflection


def curate(reflection: Reflection) -> Delta:
    """
    Convert a Reflection object into a Delta object containing operations to update the Playbook.

    This function is deterministic and rule-based:
    - For each CandidateBullet in reflection.candidate_bullets, generate an ADD operation
    - For each BulletTag in reflection.bullet_tags, generate an INCR_HELPFUL or INCR_HARMFUL operation

    Args:
        reflection: A Reflection object from the Reflector

    Returns:
        Delta: A Delta object containing the list of operations to apply to the Playbook
    """
    ops = []

    # Process bullet tags (helpful/harmful feedback on existing bullets)
    for bullet_tag in reflection.bullet_tags:
        if bullet_tag.tag == "helpful":
            ops.append(DeltaOp(
                op="INCR_HELPFUL",
                target_id=bullet_tag.id
            ))
        elif bullet_tag.tag == "harmful":
            ops.append(DeltaOp(
                op="INCR_HARMFUL",
                target_id=bullet_tag.id
            ))

    # Process candidate bullets (new bullets to add)
    for candidate in reflection.candidate_bullets:
        ops.append(DeltaOp(
            op="ADD",
            new_bullet={
                "section": candidate.section,
                "content": candidate.content,
                "tags": candidate.tags
            }
        ))

    return Delta(ops=ops)
