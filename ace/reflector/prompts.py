# ace/reflector/prompts.py

REFLECTOR_SYSTEM_PROMPT = """You are an expert software development reflector \
that analyzes task outcomes and generates actionable insights.

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


QUALITY_EVAL_SYSTEM_PROMPT = """You are a quality evaluator for reflection insights.
Assess the quality of a Reflection on three dimensions:
1. SPECIFICITY: Are insights concrete and domain-specific (not generic platitudes)?
2. ACTIONABILITY: Can the candidate bullets be directly applied to future tasks?
3. REDUNDANCY: Do candidate bullets overlap with already-retrieved bullets?

CRITICAL RULES:
- Output ONLY valid JSON matching the schema
- NO markdown fencing
- Scores are floats from 0.0 to 1.0
- Include feedback only if overall_score < 0.7"""

QUALITY_EVAL_USER_TEMPLATE = """Evaluate this Reflection for quality:

**Original Query:** {query}

**Retrieved Bullets (for redundancy comparison):**
{retrieved_bullets}

**Reflection:**
{reflection_json}

Evaluate and output JSON:
{{
  "specificity": <0.0-1.0>,
  "actionability": <0.0-1.0>,
  "redundancy": <0.0-1.0>,
  "feedback": "<improvement suggestions if needed>"
}}

Output pure JSON (no markdown fencing):"""

REFINEMENT_SYSTEM_PROMPT = """You are refining a Reflection to improve its quality.
Based on feedback, generate an improved version with:
- More specific, domain-rich insights
- More actionable candidate bullets
- Less overlap with existing bullets

CRITICAL RULES:
- Output ONLY valid JSON matching the Reflection schema
- NO markdown fencing
- Keep improvements focused on the feedback provided
- Avoid generic or verbose outputs"""

REFINEMENT_USER_TEMPLATE = """Improve this Reflection based on the feedback:

**Original Query:** {query}

**Current Reflection:**
{reflection_json}

**Quality Feedback:** {feedback}

Generate an improved Reflection JSON:"""


def format_quality_eval_prompt(
    query: str,
    retrieved_bullets: list[dict[str, str]],
    reflection_json: str,
) -> tuple[str, str]:
    """Format the quality evaluation prompt.

    Args:
        query: The original task query
        retrieved_bullets: List of dicts with 'id' and 'content' keys
        reflection_json: JSON string of the reflection to evaluate
    """
    if retrieved_bullets:
        bullets_str = "\n".join(
            f"- [{b['id']}]: {b['content']}" for b in retrieved_bullets
        )
    else:
        bullets_str = "None"
    return (
        QUALITY_EVAL_SYSTEM_PROMPT,
        QUALITY_EVAL_USER_TEMPLATE.format(
            query=query,
            retrieved_bullets=bullets_str,
            reflection_json=reflection_json,
        ),
    )


def format_refinement_prompt(
    query: str,
    reflection_json: str,
    feedback: str,
) -> tuple[str, str]:
    """Format the refinement prompt."""
    return (
        REFINEMENT_SYSTEM_PROMPT,
        REFINEMENT_USER_TEMPLATE.format(
            query=query,
            reflection_json=reflection_json,
            feedback=feedback,
        ),
    )
