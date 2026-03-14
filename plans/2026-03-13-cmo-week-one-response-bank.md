# ACE Week-One Response Bank

Date: 2026-03-13
Owner: CMO
Operator: Dan

## Purpose

Use this file as the canonical paste-ready source for week-one public replies,
DM openers, qualification follow-up, and escalation asks.

This exists to reduce launch-day drift. Dan should not have to improvise copy
or jump back into older planning docs while the Monday, Wednesday, and Friday
motions are live.

Canonical logging and measurement still happen in
`plans/2026-03-16-cmo-launch-operations-log.md`.

## Operating Rules

- Use the exact tracked URL that matches the reply, DM, or issue-intake code in
  the launch log.
- Keep replies short in public; move detail into DM, docs, or a GitHub issue.
- Ask for one concrete failure pattern before offering deeper help.
- Do not promise features or timelines that are not already supported in the
  repo.
- After any qualified reply or DM, log the outcome the same day in the launch
  log or the live capture sheet.

## Public Reply Bank

### Category Curiosity

Use when someone gets the category idea but has not asked how to start yet.

Code: `reply_w1_thread_readme`

```text
The core ACE thesis is adaptive context instead of one bigger static prompt.

If you want the sharp version, start here:
https://github.com/DannyMac180/ACE/blob/main/README.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=reply_w1_thread_readme
```

### Prompt Drift Or Long System Prompt Pain

Use when the reply already mentions prompt bloat, retrieval gaps, or weak
memory.

Code: `reply_w1_thread_readme`

```text
That is exactly the failure mode we are aiming at: the prompt keeps growing, but the system never gets better at retrieving and updating the right context.

ACE treats context as a playbook that can be retrieved, patched, and refined:
https://github.com/DannyMac180/ACE/blob/main/README.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=reply_w1_thread_readme
```

### Proof Skepticism

Use when someone asks what the loop actually does or wants inspectable proof.

Code: `reply_w1_proof_demo`

```text
Fair push. The useful question is whether the loop is inspectable, not whether it sounds smart.

The proof asset shows the retrieve -> reflect -> curate -> commit path directly:
https://github.com/DannyMac180/ACE/blob/main/docs/ace-proof-demo.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=reply_w1_proof_demo
```

### Setup Question

Use when the reply asks how to start or how ACE fits an existing coding-agent
workflow.

Code: `reply_w1_adoption_example`

```text
If you want the shortest path to first value, start with the worked example instead of the whole repo:

https://github.com/DannyMac180/ACE/blob/main/docs/getting-started-example.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=reply_w1_adoption_example
```

### Adjacent Conversation: Prompt Drift

Use on Tuesday distribution passes when Dan is replying in someone else's
thread about long system prompts, retrieval misses, or agent-memory failures.

Code: `reply_w1_thread_readme`

```text
This is the failure mode ACE is aimed at: the prompt keeps getting longer, but the system never gets better at retrieving and updating the right context.

The sharp thesis is here:
https://github.com/DannyMac180/ACE/blob/main/README.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=reply_w1_thread_readme
```

### Adjacent Conversation: Setup Path

Use on Tuesday or Thursday when the conversation is already about implementation
fit and the next useful move is a concrete setup path.

Code: `reply_w1_adoption_example`

```text
If you already have a coding-agent loop, the shortest way to evaluate fit is to run the worked example first and then test it against one repeated failure pattern:

https://github.com/DannyMac180/ACE/blob/main/docs/getting-started-example.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=reply_w1_adoption_example
```

### Move To DM

Use when the person already sounds qualified and the next step should be one
clear DM.

No tracked link in the public reply. Route with a DM code immediately after.

```text
This sounds like a real fit case. DM me your stack, one repeated failure pattern, and whether you prefer MCP or CLI. I will point you to the fastest setup path.
```

### GitHub Issue Ask

Use when the workflow description would help public adoption or product triage.

Code: `issue_w1_failure_pattern`

```text
If you are open to making this reusable, put that workflow into the issue template too:

https://github.com/DannyMac180/ACE/issues/new?template=workflow-failure-pattern.yml&title=Workflow+failure+pattern%3A+&utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=issue_w1_failure_pattern

That gives us a public failure pattern the docs and playbook can improve against.
```

## DM Bank

### Thesis-First DM

Use when the builder is category-curious but not ready for setup.

Code: `dm_w1_readme`

```text
Thanks for engaging on ACE. The sharp version is that we are treating context as a layer that gets retrieved, critiqued, patched, and refined over time instead of endlessly rewriting one giant prompt.

Start here:
https://github.com/DannyMac180/ACE/blob/main/README.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=dm_w1_readme

If a concrete workflow comes to mind after reading it, send me the stack and one repeated failure pattern.
```

### Fastest Setup DM

Use when the builder wants the shortest path to first value.

Code: `dm_w1_example_doc`

```text
If you want the shortest setup path, use the worked example first:

https://github.com/DannyMac180/ACE/blob/main/docs/getting-started-example.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=dm_w1_example_doc

If you hit a real failure pattern after that, send me the workflow, the failure, and whether you are running through MCP or CLI.
```

### MCP-Specific DM

Use only when the builder explicitly asks about MCP.

Code: `dm_w1_mcp_guide`

```text
If your current loop already runs through MCP, start here:

https://github.com/DannyMac180/ACE/blob/main/docs/MCP_USAGE_GUIDE.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=dm_w1_mcp_guide

The main thing I need from you after that is one repeated failure pattern you would want ACE to encode.
```

### API Or Implementation-Detail DM

Use only when the builder asks for technical detail beyond the worked example.

Code: `dm_w1_api_ref`

```text
If you want the technical surface area directly, start with the API reference:

https://github.com/DannyMac180/ACE/blob/main/docs/api-reference.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=dm_w1_api_ref

If you are evaluating fit, send me the workflow and one repeated failure pattern so I can point you at the right setup path.
```

### Tuesday Or Thursday Follow-Up DM

Use when a builder engaged publicly during a distribution pass and needs one
clean next step instead of a long custom message.

Code: `dm_w1_example_doc` unless they asked specifically for MCP or API detail.

```text
Following up here because your workflow sounds real. The fastest way to pressure-test ACE is still the worked example:

https://github.com/DannyMac180/ACE/blob/main/docs/getting-started-example.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=dm_w1_example_doc

If you try it, send back the stack and the first repeated failure pattern you want ACE to learn from.
```

## Qualification Prompts

Use one or two of these after the first DM. Do not send a questionnaire wall.

### Fit Check

```text
Helpful to know before we go further: are you trying to improve a real workflow this week, or are you still exploring the space?
```

### Failure Pattern Ask

```text
What is the repeated failure pattern you most want the system to stop repeating?
```

### Tuesday Distribution Fit Ask

```text
Are you looking at this as a live workflow problem right now, or as a general architecture idea for later?
```

### Stack And Route Ask

```text
What stack are you already using, and do you want the fastest path through MCP or CLI?
```

### Proof Asset Ask

```text
If this works, what proof would matter most to you: fewer repeated failures, cleaner context, or a setup path your team can inspect?
```

## Escalation Copy

### Call Invite

Use when the builder has a real workflow, a repeated failure pattern, and
enough technical fluency to run ACE with help.

```text
This looks like a strong design-partner case. If you want, we can do a short working session focused on one failure pattern, one setup path, and one success criterion.
```

### GitHub Issue Or Discussion Ask

Use when the workflow should become a public artifact instead of staying in DM.

```text
This would be useful as a public issue or discussion too. If you write up the workflow and failure pattern, it gives us a concrete target for docs and product improvements.
```

### Post-Call Follow-Up

```text
Thanks for the session. The next useful step is to run the agreed setup path and send back the first friction point or result. Concrete failure data is more useful than general impressions.
```

## Logging Minimum

After any qualified public reply or DM, capture these fields before the day
ends:

- handle or name
- source motion
- code used
- GitHub URL if a public workflow issue or discussion was opened
- stack if known
- repeated failure pattern or objection
- qualification score
- next step and due date
- whether the interaction belongs in the design-partner pipeline

## Distribution Rule

For Tuesday and Thursday launch passes:

- use these templates only on conversations that already match ACE pain
- prefer the README template for category contrast and the worked example for
  hands-on setup intent
- if a conversation needs a brand-new public post to make sense, do not invent
  one on the fly; update the launch registry first or keep the motion inside
  replies and DMs

If the interaction surfaces a repeated message, proof, or setup blocker, copy
it into the launch log friction section the same day.
