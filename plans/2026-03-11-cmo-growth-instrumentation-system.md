# ACE CMO Growth Instrumentation System

Date: 2026-03-11
Owner: CMO

## Objective

Measure whether ACE marketing is creating real adoption, not just impressions.

This system is designed for the first four campaign weeks, from March 16, 2026
through April 10, 2026, with minimal tooling and low operating overhead.

Week-one scoreboard status:

- the live reporting surface for Monday, March 16, 2026 through Friday, March 20, 2026 is `plans/2026-03-16-cmo-launch-operations-log.md`
- this file remains the definitions-and-governance reference, not the active week-one execution log

## What This System Must Answer

Every week, the team should be able to answer:

1. Which posts produced the most qualified traffic?
2. Which destination converted best: README, MCP guide, API docs, or issues/discussions?
3. Which channels produced serious builder conversations, not just engagement?
4. Which conversations turned into setup starts, setup completions, or repeat-user signals?
5. Which objections or friction points should shape the next week's content and docs?

## Measurement Principles

- One post should drive one primary CTA.
- Every campaign link needs a unique tracked URL.
- Manual logging is acceptable if it is consistent.
- Adoption signals matter more than vanity reach.
- The weekly review should change the next week's publishing decisions.

## Asset And Link Map

Use one destination per post and a stable asset code for every campaign output.

| Asset code | Destination | Primary job | Primary KPI |
| --- | --- | --- | --- |
| `readme` | `README.md` | explain category and product thesis | qualified clicks |
| `mcp_guide` | `docs/MCP_USAGE_GUIDE.md` | convert technical curiosity into setup intent | setup starts |
| `api_ref` | `docs/api-reference.md` | satisfy skeptical technical readers | deep clicks and replies |
| `example_doc` | getting-started or worked example doc | move builders to first value | setup completions |
| `demo_asset` | demo clip, GIF, or proof doc | prove the loop visually | saves, reposts, proof clicks |
| `issue_or_discussion` | GitHub issue or discussion | collect real use cases and friction | adoption signals |

## UTM Convention

Use this exact pattern for X traffic:

`?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=<post_code>`

Use `post_code` in this format:

`<week>_<post_type>_<asset>`

Examples:

- `w1_thread_readme`
- `w1_proof_demo`
- `w1_adoption_example`
- `w2_thread_diagram`
- `w3_proof_mcp_guide`
- `w4_launch_example_doc`

If a reply thread carries a tracked link, prefix it with `reply_`:

- `reply_w1_thread_readme`

If a DM follow-up carries a tracked link, prefix it with `dm_`:

- `dm_w1_mcp_guide`

## Weekly Asset Register

Before each week starts, log the planned outputs in this format.

For week one, treat the sample rows below as historical planning context only.
These rows should still mirror the final locked week-one registry so older
planning examples do not point at a stale destination. The active week-one
asset register now lives in
`plans/2026-03-16-cmo-launch-operations-log.md` and
`plans/2026-03-12-ceo-growth-instrumentation-pack.md`.

| Week | Post code | Channel | Format | Destination | Goal | Owner | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `w1_thread_readme` | X | thread | README | category clicks | Dan | planned |
| 1 | `w1_proof_demo` | X | proof post | demo asset | proof clicks | Dan | planned |
| 1 | `w1_adoption_example` | X | adoption post | getting-started example doc | setup starts and adoption signals | Dan | planned |

## Weekly Scoreboard

Update this once per week, ideally every Friday after publishing and follow-up.

For week one, update the live scoreboard row only in
`plans/2026-03-16-cmo-launch-operations-log.md`.

| Week | Qualified impressions | Engagement rate | Outbound clicks | Setup starts | Setup completions | Adoption signals | Serious conversations | Repeat-user signals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 |  |  |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |  |  |

## Metric Definitions

Use these definitions consistently.

| Metric | Definition | Capture method |
| --- | --- | --- |
| Qualified impressions | Impressions on ACE campaign posts and targeted reply threads only | X analytics |
| Engagement rate | `(likes + replies + reposts + bookmarks) / impressions` per campaign post | X analytics |
| Outbound clicks | Clicks on tracked ACE links from posts, replies, or DMs | link analytics or manual counts |
| Setup starts | People who indicate they opened the guide, cloned the repo, or began setup | DM log, GitHub issue, discussion, or direct reply |
| Setup completions | People who confirm ACE ran locally, via CLI, or via MCP | DM log, GitHub thread, or direct reply |
| Adoption signals | Stars, issues, discussions, replies saying they tested or configured ACE | GitHub plus X |
| Serious conversations | DMs, issue threads, or call requests with a real workflow and failure pattern | manual pipeline log |
| Repeat-user signals | Second meaningful touch from the same builder within 14 days | manual pipeline log |

## Design Partner Pipeline Log

Track high-signal builders in a simple table.

| Handle or name | Source | Post code | Stack | Failure pattern | Qualification score | Next step | Last touch | Current stage | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |

Use these stage values:

- `engaged`
- `qualified`
- `call_booked`
- `call_done`
- `setup_started`
- `setup_completed`
- `meaningful_result`
- `inactive`

## Operating Cadence

Week-one scoreboard status:

- the active launch-week reporting sheet now lives in `plans/2026-03-16-cmo-launch-operations-log.md`
- this file keeps definitions, thresholds, and ownership guidance only

### Monday

- confirm the week's three post codes and destinations
- verify tracked links resolve to live assets
- log the week's planned outputs in the asset register

### Wednesday

- capture performance on Monday's post
- record high-signal replies and DMs in the pipeline log
- note whether objections are message, proof, or setup problems

### Friday

- update the weekly scoreboard
- update the design partner pipeline log
- decide the next week's thesis based on click quality and setup signals

## Decision Thresholds

Use these thresholds to decide what to fix next.

| Signal | Threshold | Interpretation | Required response |
| --- | --- | --- | --- |
| Engagement rate | under `1.5%` | hook or framing is weak | tighten opening line and contrast |
| Click-through from X | under `0.7%` of impressions | CTA or destination is weak | simplify CTA and improve destination page |
| Setup starts from clicks | under `10%` | proof is interesting but action path is weak | improve setup doc and demo clarity |
| Setup completions from starts | under `40%` | product or documentation friction is too high | prioritize docs fixes and onboarding support |
| Repeat-user signals | under `2` by end of week 2 | audience is curious but not committed | push design-partner outreach and stronger proof |

## Owners

| Area | Owner | Responsibility |
| --- | --- | --- |
| Post publishing | Dan | publish, reply, and log X analytics |
| Weekly scoreboard | operator or CEO | consolidate the weekly numbers |
| Design partner pipeline | Dan with CMO guidance | maintain prospect status and next steps |
| Destination quality | Founding Engineer | ensure README, docs, demo, and example are conversion-ready |
| Weekly decision review | CMO | recommend next content and asset priorities |

## Source Of Truth

Until better automation exists, the source of truth should be one shared markdown
log in the repo plus the existing design-partner notes.

Current shared files:

- this file for measurement definitions, thresholds, and operating rules
- `plans/2026-03-16-cmo-launch-operations-log.md` for the live week-one execution log and scoreboard
- `plans/2026-03-11-cmo-design-partner-engine.md` for qualification and outreach
- `plans/2026-03-12-ceo-growth-instrumentation-pack.md` for the locked week-one tracked-link registry

## Immediate Next Step

The immediate implementation step is complete: week one now has a live asset
register, tracked-link registry, and scoreboard process in the launch operations
log. Use this file only to guide later-week measurement changes and threshold
decisions.
