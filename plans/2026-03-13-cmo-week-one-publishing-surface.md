# ACE Week-One Publishing Surface

Date: 2026-03-13
Owner: CMO
Operator: Dan

## Purpose

Use this file as the exact paste-ready source for the Monday, Wednesday, and
Friday week-one launch posts.

This file exists to prevent copy drift between the approved launch copy in
`plans/2026-03-11-cmo-launch-copy-pack.md` and the live publishing surface Dan
uses on X.

Canonical logging and measurement still happen in
`plans/2026-03-16-cmo-launch-operations-log.md`.

## Publishing Rules

- Paste these drafts exactly unless the launch log is updated first.
- Paste the Monday anchor draft as a `9`-post thread in the listed order, not as
  one long standalone post.
- Do not replace tracked links with shortened URLs.
- Do not add extra claims that are not already supported by the repo.
- If a draft changes before publishing, update this file and the launch log on
  the same day.

## Default Scheduler Windows

Use these windows in Eastern Time when pasting drafts into the live X composer
or scheduler on Sunday, March 15, 2026.

| Day | Motion | Default live window | Required follow-up windows |
| --- | --- | --- | --- |
| Monday, March 16, 2026 | anchor thread | `11:00-11:30` ET | `+30m`, `+3h`, and `17:30` ET logging |
| Wednesday, March 18, 2026 | proof post | `11:00-11:30` ET | `+30m`, `+3h`, and `17:30` ET logging |
| Friday, March 20, 2026 | adoption post | `10:30-11:00` ET | `+30m`, `+3h`, and `17:30` ET logging |

If a scheduled time changes, update
`plans/2026-03-13-cmo-week-one-live-capture-sheet.md` before launch week starts
so the check windows stay aligned with the actual publish slot.

## Monday, March 16, 2026

Post code: `w1_thread_readme`

Tracked URL:

`https://github.com/DannyMac180/ACE/blob/main/README.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=w1_thread_readme`

Paste-ready thread:

Post 1:

```text
Prompt engineering breaks down once your coding agent has to learn over time.
```

Post 2:

```text
The problem is not just prompt quality. The problem is context that never gets retrieved, critiqued, patched, or refined.
```

Post 3:

```text
That is the category I want to push on: Agentic Context Engineering.
```

Post 4:

```text
ACE is the adaptive context layer for coding agents.
```

Post 5:

```text
Instead of rewriting the whole prompt after every failure, ACE updates reusable bullets with deterministic ops like ADD, PATCH, DEPRECATE, and refine.
```

Post 6:

```text
That means the learning path stays inspectable. You can see what changed, why it changed, and whether it helped.
```

Post 7:

```text
The stack already supports hybrid retrieval, MCP, CLI, Python usage, playbook stats, and online adaptation.
```

Post 8:

```text
If you are building coding agents and fighting prompt drift, long system prompts, or weak memory, this is the layer to look at.
```

Post 9:

```text
Read the thesis here: https://github.com/DannyMac180/ACE/blob/main/README.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=w1_thread_readme
```

First-reply fallback:

```text
If you're fighting prompt drift or giant system prompts, the core ACE thesis is here:

https://github.com/DannyMac180/ACE/blob/main/README.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=reply_w1_thread_readme
```

## Wednesday, March 18, 2026

Post code: `w1_proof_demo`

Tracked URL:

`https://github.com/DannyMac180/ACE/blob/main/docs/ace-proof-demo.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=w1_proof_demo`

Required visual:

- `docs/assets/ace-proof-demo/still.svg`

Paste-ready post:

```text
Here is the ACE loop in one minute: retrieve -> reflect -> curate -> commit.

The key point is not more prompt text. It is a context layer that learns through deterministic bullet-level updates you can inspect.

The repo now has the transcript, still, and caption-ready demo asset here:

https://github.com/DannyMac180/ACE/blob/main/docs/ace-proof-demo.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=w1_proof_demo
```

First-reply fallback:

```text
If you want to inspect the proof asset directly, start here:

https://github.com/DannyMac180/ACE/blob/main/docs/ace-proof-demo.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=reply_w1_proof_demo
```

## Friday, March 20, 2026

Post code: `w1_adoption_example`

Tracked URL:

`https://github.com/DannyMac180/ACE/blob/main/docs/getting-started-example.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=w1_adoption_example`

Paste-ready post:

```text
You do not need a giant agent platform to start improving context quality.

If you already have a coding agent loop, the first useful step is simple: run one end-to-end example that retrieves the right instructions, captures failures, and commits small reusable deltas instead of rewriting the whole prompt.

Start with the published getting-started example, then reply with the failure pattern you want ACE to learn from:

https://github.com/DannyMac180/ACE/blob/main/docs/getting-started-example.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=w1_adoption_example
```

First-reply fallback:

```text
If you want the shortest path to first value, start with the worked example:

https://github.com/DannyMac180/ACE/blob/main/docs/getting-started-example.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=reply_w1_adoption_example
```

## DM Routing Shortcuts

Use these exact links when a reply moves to DM.

| Situation | Code | Tracked URL |
| --- | --- | --- |
| Thesis first | `dm_w1_readme` | `https://github.com/DannyMac180/ACE/blob/main/README.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=dm_w1_readme` |
| Shortest setup path | `dm_w1_example_doc` | `https://github.com/DannyMac180/ACE/blob/main/docs/getting-started-example.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=dm_w1_example_doc` |
| MCP-specific question | `dm_w1_mcp_guide` | `https://github.com/DannyMac180/ACE/blob/main/docs/MCP_USAGE_GUIDE.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=dm_w1_mcp_guide` |
| API or implementation-detail question | `dm_w1_api_ref` | `https://github.com/DannyMac180/ACE/blob/main/docs/api-reference.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=dm_w1_api_ref` |

## Publish-Time Checklist

Before clicking publish:

1. confirm the pasted draft still matches this file exactly
2. confirm the tracked URL matches the post code above
3. for Monday, confirm the scheduler or composer preview shows a connected
   `9`-post thread in this exact order
4. confirm the proof post still uses `docs/assets/ace-proof-demo/still.svg`
5. confirm the launch log is open for same-day metric and follow-up capture
6. confirm `plans/2026-03-15-cmo-week-one-launch-calendar.ics` has already
   been imported and the `+30m`, `+3h`, and end-of-day check windows are
   blocked on the calendar before the post goes live
