# ace/reflector/prompts.py

REFLECTOR_SYSTEM_PROMPT = """You are an expert software development reflector that analyzes task outcomes and generates actionable insights.

Your role is to:
1. Identify errors or failures in the execution
2. Analyze root causes
3. Propose correct approaches
4. Extract key insights that can be reused
5. Tag existing bullets as helpful or harmful based on retrieval
6. Generate new candidate bullets that are SHORT, REUSABLE, and DOMAIN-RICH

CRITICAL RULES:
- Output ONLY valid JSON matching the Reflection schema
- NO markdown fencing (no ```json)
- Keep insights actionable and concise (1-2 sentences max)
- Prefer one small bullet over a paragraph
- Always include tags (repo/service/topic/tool) for new bullets
- Avoid narrative or chain-of-thought in output"""

REFLECTOR_USER_TEMPLATE = """Analyze this task execution and generate a Reflection:

**Query/Task:** {query}

**Retrieved Bullet IDs:** {retrieved_bullet_ids}

**Code Changes:**
{code_diff}

**Test Output:**
{test_output}

**Logs:**
{logs}

**Environment/Metadata:**
{env_meta}

Generate a Reflection JSON with:
- error_identification: what went wrong (if anything)
- root_cause_analysis: why it happened
- correct_approach: what should have been done
- key_insight: reusable lesson learned
- bullet_tags: mark retrieved bullets as helpful/harmful
- candidate_bullets: new short, tagged bullets to add

Output pure JSON (no markdown fencing):"""


def format_reflector_prompt(
    query: str,
    retrieved_bullet_ids: list[str],
    code_diff: str = "",
    test_output: str = "",
    logs: str = "",
    env_meta: dict | None = None,
) -> tuple[str, str]:
    """Format the reflector prompt with input data.

    Returns:
        tuple: (system_prompt, user_prompt)
    """
    env_meta_str = str(env_meta) if env_meta else "None"

    user_prompt = REFLECTOR_USER_TEMPLATE.format(
        query=query,
        retrieved_bullet_ids=", ".join(retrieved_bullet_ids) if retrieved_bullet_ids else "None",
        code_diff=code_diff or "None",
        test_output=test_output or "None",
        logs=logs or "None",
        env_meta=env_meta_str,
    )

    return REFLECTOR_SYSTEM_PROMPT, user_prompt
