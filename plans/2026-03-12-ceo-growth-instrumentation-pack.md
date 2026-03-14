# ACE Week-One Growth Instrumentation Pack

Date: 2026-03-12
Owner: CEO
Launch window: Monday, March 16, 2026 through Friday, March 20, 2026

## Purpose

Stand up the minimum operating system needed to ensure zero untracked ACE launch
posts during launch week and one shared place to review what converted.

This pack is the implementation artifact for `ACE-7`.

## Weekly Metric Owner

The CEO owns the weekly scoreboard and is responsible for publishing the final
Friday update in the repo.

Supporting roles:

- Dan owns post publishing, raw X analytics capture, and DM follow-up
- Founding Engineer owns destination quality for README, docs, and proof assets
- CEO owns metric consolidation, quality control, and the weekly closeout call

## Canonical Base URLs

- repo: `https://github.com/DannyMac180/ACE`
- README: `https://github.com/DannyMac180/ACE/blob/main/README.md`
- proof demo: `https://github.com/DannyMac180/ACE/blob/main/docs/ace-proof-demo.md`
- getting-started example: `https://github.com/DannyMac180/ACE/blob/main/docs/getting-started-example.md`
- MCP guide: `https://github.com/DannyMac180/ACE/blob/main/docs/MCP_USAGE_GUIDE.md`
- API reference: `https://github.com/DannyMac180/ACE/blob/main/docs/api-reference.md`
- adaptation loop diagram: `https://github.com/DannyMac180/ACE/blob/main/docs/arch-diagrams/online-adaptation-loop.md`
- issue intake: `https://github.com/DannyMac180/ACE/issues/new`

## Tracking Convention

Use this exact parameter pattern on every launch link:

`?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=<post_code>`

Rules:

- every public post gets one primary CTA and one unique `utm_content`
- every public reply uses a `reply_` prefixed `utm_content`
- every DM follow-up uses a `dm_` prefixed `utm_content`
- if a destination changes, update both this file and the launch operations log
  before the link is used

## Attribution Guardrail

Treat the GitHub launch URLs as governed routing labels, not as a complete
analytics system by themselves.

- the `utm_*` parameters keep post, reply, DM, and profile routes distinct, but
  GitHub does not turn those parameters into reliable per-code click reporting
- for public X posts and replies, use X-native link-click data when it is
  available
- for DMs, profile-link traffic, or any surface where a reliable click count is
  not exposed, log the exact code used and count downstream evidence such as a
  reply, DM response, issue, setup start, or setup completion instead of
  inventing a click number
- never backfill per-code click totals from mixed GitHub page traffic or
  memory-based estimates

## Final Week-One Tracked-Link Registry

Use these exact URLs for the week of Monday, March 16, 2026.

| Motion | Post code | Destination | Tracked URL | Owner | Status |
| --- | --- | --- | --- | --- | --- |
| Anchor thread | `w1_thread_readme` | README | `https://github.com/DannyMac180/ACE/blob/main/README.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=w1_thread_readme` | Dan | ready |
| Anchor-thread replies | `reply_w1_thread_readme` | README | `https://github.com/DannyMac180/ACE/blob/main/README.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=reply_w1_thread_readme` | Dan | ready |
| Proof post | `w1_proof_demo` | proof demo | `https://github.com/DannyMac180/ACE/blob/main/docs/ace-proof-demo.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=w1_proof_demo` | Dan | ready |
| Proof-post replies | `reply_w1_proof_demo` | proof demo | `https://github.com/DannyMac180/ACE/blob/main/docs/ace-proof-demo.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=reply_w1_proof_demo` | Dan | ready |
| Adoption post | `w1_adoption_example` | getting-started example | `https://github.com/DannyMac180/ACE/blob/main/docs/getting-started-example.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=w1_adoption_example` | Dan | ready |
| Adoption-post replies | `reply_w1_adoption_example` | getting-started example | `https://github.com/DannyMac180/ACE/blob/main/docs/getting-started-example.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=reply_w1_adoption_example` | Dan | ready |
| README DM follow-up | `dm_w1_readme` | README | `https://github.com/DannyMac180/ACE/blob/main/README.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=dm_w1_readme` | Dan | ready |
| Example-doc DM follow-up | `dm_w1_example_doc` | getting-started example | `https://github.com/DannyMac180/ACE/blob/main/docs/getting-started-example.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=dm_w1_example_doc` | Dan | ready |
| MCP-guide DM follow-up | `dm_w1_mcp_guide` | MCP guide | `https://github.com/DannyMac180/ACE/blob/main/docs/MCP_USAGE_GUIDE.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=dm_w1_mcp_guide` | Dan | ready |
| Technical-reference DM follow-up | `dm_w1_api_ref` | API reference | `https://github.com/DannyMac180/ACE/blob/main/docs/api-reference.md?utm_source=x&utm_medium=social&utm_campaign=ace_launch&utm_content=dm_w1_api_ref` | Dan | ready |

## Publishing Rules

- Use `w1_proof_demo` as the default Wednesday proof CTA.
- Use `w1_adoption_example` as the default Friday adoption CTA.
- Use `dm_w1_readme` for category-curious DMs that need thesis context before a
  setup path.
- Use `dm_w1_example_doc` for high-intent DMs before sending deeper technical
  docs.
- Use `dm_w1_mcp_guide` only when the builder explicitly asks how to run ACE
  through MCP.
- Reserve `issue intake` for collecting specific failure patterns after someone
  has already seen the example.
- Do not publish any launch link that is not listed in the registry above.

## Week-One Asset Register

| Week | Asset or motion | Post code | Destination | Goal | Owner | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Anchor thread | `w1_thread_readme` | README | qualified category clicks | Dan | ready | main Monday, March 16, 2026 CTA |
| 1 | Anchor-thread replies | `reply_w1_thread_readme` | README | convert high-signal replies into deeper clicks | Dan | ready | use in same-day replies only |
| 1 | Proof post | `w1_proof_demo` | proof demo | proof clicks | Dan | ready | use the published demo package as the default proof surface |
| 1 | Proof-post replies | `reply_w1_proof_demo` | proof demo | move skeptics to mechanism proof | Dan | ready | use for technical objections |
| 1 | Adoption post | `w1_adoption_example` | getting-started example | setup starts and adoption signals | Dan | ready | Friday, March 20, 2026 CTA |
| 1 | Adoption-post replies | `reply_w1_adoption_example` | getting-started example | move interested builders into first value | Dan | ready | use on qualified public threads |
| 1 | DM thesis follow-up | `dm_w1_readme` | README | category clarification | Dan | ready | use when a builder wants the thesis before a setup path |
| 1 | DM worked example | `dm_w1_example_doc` | getting-started example | setup starts and setup completions | Dan | ready | primary DM for high-intent builders; send before API reference unless asked otherwise |
| 1 | DM MCP setup path | `dm_w1_mcp_guide` | MCP guide | MCP setup starts | Dan | ready | use only when the builder explicitly asks how to run ACE through MCP |
| 1 | DM technical reference | `dm_w1_api_ref` | API reference | technical validation | Dan | ready | use only when asked for implementation detail |
| 1 | Demo asset | n/a | reusable proof asset | stronger saves and proof clicks | Founding Engineer | ready | delivered in `docs/ace-proof-demo.md` |
| 1 | Example-driven adoption doc | n/a | worked example | setup completions | Founding Engineer | ready | delivered in `docs/getting-started-example.md` |

## Shared Scoreboard Update Process

The shared scoreboard lives in
`plans/2026-03-16-cmo-launch-operations-log.md`.

Update cadence:

1. Monday, March 16, 2026 preflight
   - Dan confirms the exact launch link used for the anchor thread.
   - CEO verifies the post code exists in the registry above.
2. Same day after each post
   - Dan records the live post URL, impression count, engagement count, click
     count if available, and any high-signal replies or DMs.
   - Dan adds one row to `plans/2026-03-16-cmo-launch-operations-log.md`'s
     `Engagement Follow-Up Log` for each qualified reply burst, outbound DM, or
     inbound DM thread that creates follow-up work.
3. Wednesday, March 18, 2026 checkpoint
   - CEO updates the operating log with interim proof-post results, audits the
     engagement follow-up rows, and decides whether proof or adoption traffic
     should be re-routed based on click quality.
4. Friday, March 20, 2026 closeout
   - CEO updates the scoreboard row, pipeline table, friction log, and week-one
     closeout notes in the shared launch log.

## Required Metrics

Every Friday scoreboard update must include these fields:

- qualified impressions
- engagement rate
- outbound clicks
- README clicks
- proof clicks
- setup starts
- setup completions
- adoption signals
- serious conversations
- repeat-user signals

## Source Inputs By Metric

| Metric | Primary source | Capture owner |
| --- | --- | --- |
| Qualified impressions | X analytics for campaign posts and targeted replies | Dan |
| Engagement rate | X analytics | Dan |
| Outbound clicks | X-native link clicks for governed public posts and replies, plus documented manual counts only when the surface exposes a real number | Dan |
| README clicks | measurable governed public-click counts tied to README-coded launch links; if DM or profile clicks are not exposed, count downstream setup or conversation signals instead | CEO |
| Proof clicks | measurable governed public-click counts tied to proof-demo-coded launch links; do not infer proof clicks from mixed GitHub traffic | CEO |
| Setup starts | DMs, GitHub issues, or discussions showing setup intent | CEO |
| Setup completions | direct confirmations in DMs or GitHub | CEO |
| Adoption signals | GitHub stars, issues, discussions, and tested-it replies | CEO |
| Serious conversations | qualified DMs, calls, or multi-turn issue threads | CEO |
| Repeat-user signals | second meaningful touch within 14 days | CEO |

## Completion Standard

`ACE-7` is complete when:

- every week-one launch motion has a named tracked URL
- the CEO is explicitly named as weekly metric owner
- the shared scoreboard update process is written down in-repo
- the week-one asset register is ready to use before Monday, March 16, 2026
