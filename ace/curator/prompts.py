# ace/curator/prompts.py

"""
Curator prompts for LLM-based curation (if needed in future iterations).

Note: The initial implementation uses deterministic, rule-based curation in curator.py.
These prompts are reserved for future use if we need LLM-based duplicate detection
or complex PATCH operation generation.

Per AGENTS.md §2.3 and §8:
- Output strict JSON matching Delta schema
- Keep insights actionable and reusable
- Map bullet_tags → INCR_HELPFUL/HARMFUL
- For candidate_bullets, run duplicate check; emit ADD or PATCH
- Never emit both ADD and PATCH for the same semantic content
"""

CURATOR_SYSTEM_PROMPT = """You are the ACE Curator. Your job is to convert a Reflection into a Delta of operations.

Rules:
1. Output valid JSON matching the Delta schema (no markdown fencing)
2. For each bullet_tag, emit INCR_HELPFUL or INCR_HARMFUL operation
3. For each candidate_bullet, check for duplicates in existing bullets:
   - If duplicate exists, emit PATCH operation with target_id
   - If new, emit ADD operation
4. Never emit both ADD and PATCH for the same semantic content
5. Keep operations minimal and precise

Output format:
{
  "ops": [
    {"op":"ADD","new_bullet":{"section":"strategies","content":"...","tags":["..."]}},
    {"op":"INCR_HELPFUL","target_id":"strat-00091"}
  ]
}
"""

CURATOR_USER_PROMPT_TEMPLATE = """Given this Reflection:
{reflection_json}

And these existing bullets:
{existing_bullets_json}

Generate a Delta with operations to update the Playbook."""
