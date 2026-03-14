# ACE CMO Launch Copy Pack

Date: 2026-03-11
Owner: CMO

## Purpose

This is the execution layer for the existing CMO strategy artifacts.

Use this file to publish, schedule, or delegate ACE launch content without having
to reinterpret the positioning each time.

Week-one scoreboard status:

- for Monday, March 16, 2026 through Friday, March 20, 2026, the only live scoreboard and execution log is `plans/2026-03-16-cmo-launch-operations-log.md`
- any scoreboard guidance in this file is historical planning context, not the active reporting surface

## Guardrails

- Only make claims the current repo and docs support.
- Do not imply customer traction, benchmarks, or usage numbers that do not exist yet.
- Keep the contrast sharp: adaptive context versus brittle prompt rewrites.
- Point every post at one concrete destination.

## Message Core

Category:

- `Agentic Context Engineering`

Product line:

- `ACE is the adaptive context layer for coding agents.`

Expanded line:

- `ACE helps coding agents retrieve the right context, learn from execution, and update reusable instructions through deterministic bullet-level deltas instead of brittle prompt rewrites.`

Approved proof points:

- hybrid retrieval instead of stuffing the whole playbook into the prompt
- deterministic delta operations like `ADD`, `PATCH`, `DEPRECATE`, and `INCR_*`
- MCP, CLI, and Python usage paths
- refine and stats flows for inspecting context quality
- online adaptation and warm-start support

## CTA Map

Use one destination per post so attribution stays clean.

| CTA code | Destination | Intent |
| --- | --- | --- |
| `thesis` | `README.md` | explain what ACE is and why it exists |
| `proof_demo` | `docs/ace-proof-demo.md` | show the ACE loop with a real reusable asset |
| `example` | `docs/getting-started-example.md` | convert curiosity into first value |
| `mcp` | `docs/MCP_USAGE_GUIDE.md` | help builders run ACE through MCP |
| `api` | `docs/api-reference.md` | support technically skeptical readers |
| `diagram` | `docs/arch-diagrams/online-adaptation-loop.md` | explain the loop visually |
| `issue` | GitHub issues or discussions | collect adoption signals and feedback |

## Tracking Scheme

If Dan posts from X, use one stable URL convention for every campaign link:

`?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=<post_code>`

Recommended `post_code` values:

- `w1_thread_readme`
- `w1_proof_demo`
- `w1_adoption_example`
- `w2_delta`
- `w3_mcp`
- `w4_launch`

Minimum weekly scoreboard:

| Metric | Owner | How to capture |
| --- | --- | --- |
| impressions | Dan | X analytics |
| engagement rate | Dan | X analytics |
| outbound clicks | Dan | tracked links |
| GitHub stars | operator | daily repo check |
| issues or discussions opened | operator | daily repo check |
| serious builder conversations | Dan | manual count in notes |
| repeat-user signals within 14 days | Dan | manual count in notes |

## Week 1 Launch Kit

Goal:

- make builders repeat back the category line and click into the repo thesis

Primary destination:

- `README.md`

Primary metric:

- qualified README clicks

### Anchor Thread

Use this as the first category post.

Approved week-one CTA for this draft:

- post code: `w1_thread_readme`
- tracked URL: `https://github.com/DannyMac180/ACE/blob/main/README.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=w1_thread_readme`

1. `Prompt engineering breaks down once your coding agent has to learn over time.`
2. `The problem is not just prompt quality. The problem is context that never gets retrieved, critiqued, patched, or refined.`
3. `That is the category I want to push on: Agentic Context Engineering.`
4. `ACE is the adaptive context layer for coding agents.`
5. `Instead of rewriting the whole prompt after every failure, ACE updates reusable bullets with deterministic ops like ADD, PATCH, DEPRECATE, and refine.`
6. `That means the learning path stays inspectable. You can see what changed, why it changed, and whether it helped.`
7. `The stack already supports hybrid retrieval, MCP, CLI, Python usage, playbook stats, and online adaptation.`
8. `If you are building coding agents and fighting prompt drift, long system prompts, or weak memory, this is the layer to look at.`
9. `Read the thesis here: <tracked README link>`

### Short Post A

`Static prompts do not scale to long-running coding agents.`

`What you want is adaptive context: retrieve the right guidance, learn from execution, patch the reusable playbook, and refine it over time.`

`That is what I mean by Agentic Context Engineering.`

`ACE is the toolkit we are using to make that concrete. <tracked README link>`

### Short Post B

`Most agent stacks still treat memory like a blob or a prompt append.`

`ACE takes a different path: bullet-level context, hybrid retrieval, deterministic delta ops, and a refine loop.`

`More inspectable. Less magic. Better fit for coding agents. <tracked README link>`

### Short Post C

`If your agent keeps making the same mistake, a longer prompt is usually the wrong fix.`

`You need a context layer that can actually learn from execution.`

`That is the ACE thesis. <tracked README link>`

### Proof Post

Approved week-one CTA for this draft:

- post code: `w1_proof_demo`
- tracked URL: `https://github.com/DannyMac180/ACE/blob/main/docs/ace-proof-demo.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=w1_proof_demo`

`Here is the ACE loop in one minute: retrieve -> reflect -> curate -> commit.`

`The key point is not more prompt text. It is a context layer that learns through deterministic bullet-level updates you can inspect.`

`The repo now has the transcript, still, and caption-ready demo asset here: <tracked proof-demo link>`

### Adoption Post

Approved week-one CTA for this draft:

- post code: `w1_adoption_example`
- tracked URL: `https://github.com/DannyMac180/ACE/blob/main/docs/getting-started-example.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=w1_adoption_example`

`You do not need a giant agent platform to start improving context quality.`

`If you already have a coding agent loop, the first useful step is simple: run one end-to-end example that retrieves the right instructions, captures failures, and commits small reusable deltas instead of rewriting the whole prompt.`

`Start with the published getting-started example, then reply with the failure pattern you want ACE to learn from: <tracked example link>`

## Founder Reply Bank

Use these in adjacent X conversations. Do not spam. Only reply where the pain is already present.

### Prompt Drift

`This is the part most teams underestimate: the prompt keeps growing, but the system never gets better at retrieving the right context at the right time.`

`We have been framing that as Agentic Context Engineering. ACE is the toolkit version of that idea. <tracked README link>`

### Long System Prompts

`A giant system prompt is usually hiding a retrieval problem.`

`ACE treats context as a playbook that can be retrieved, patched, and refined instead of endlessly rewritten. <tracked README link>`

### MCP Discussion

`One reason we built ACE around MCP as well as CLI and Python is that builders already have the client loop.`

`The missing layer is adaptive context, not another orchestration story. <tracked MCP link>`

### Agent Memory Failure

`If the agent can fail, reflect, and still not improve, the memory path is underpowered.`

`ACE focuses on small deterministic context updates rather than one more rewrite pass. <tracked README link>`

## DM Follow-Up Script

Use this after a strong reply, repost, or inbound question.

`Thanks for engaging on ACE. The shortest path to first value is to read the README thesis, then try either the MCP guide or a CLI flow depending on how you work.`

`If you want the fastest hands-on path, start with the getting-started example. If you hit a real failure pattern after that, send it to me and we can see whether the adaptive-context loop encodes something useful.`

## Week 2 Through Week 4 Reusable Templates

### Delta Mechanism Post

`The important design choice in ACE is not just reflection. It is the merge path.`

`LLMs can propose changes, but the playbook updates happen through deterministic ops like ADD, PATCH, DEPRECATE, and refine.`

`That keeps the learning loop inspectable instead of mystical. <tracked link>`

### MCP Fit Post

`ACE is not asking builders to throw away their current agent stack.`

`If you already use MCP, CLI workflows, or Python glue, the missing piece is adaptive context that can improve over time. <tracked MCP link>`

### Design Partner Post

`If you run coding agents and see repeat failure patterns, I want those cases.`

`Reply or open an issue with the workflow. The goal is to make Agentic Context Engineering useful on real systems, not just publish a theory.`

## Weekly Publishing Checklist

Before each post:

- confirm the linked destination exists and matches the claim
- confirm the CTA is singular
- confirm there is one proof point, not five loose claims
- confirm the post can be understood by a builder who has never heard of ACE

After each post:

- log the post code and destination
- log impressions, engagement rate, and clicks after 24 hours
- note every serious reply, DM, issue, or setup confirmation
- extract objections that should influence product or messaging

## Recommended Immediate Sequence

1. Publish the anchor thread once the README opening is tightened.
2. Follow with one short category post the next day using the same README destination.
3. Publish the proof post against the live demo asset on Wednesday.
4. Use reply-bank scripts daily on adjacent X conversations.
5. Review the weekly scoreboard every Friday and adjust the next week's thesis based on click quality, not vanity reach.
