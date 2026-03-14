# ACE CMO Growth Operating Plan

Date: 2026-03-11
Owner: CMO

## Objective

From March 16, 2026 to April 10, 2026, make ACE legible as a category, not just a repo.

The operating goal is simple:

- turn Dan's existing X audience into high-intent ACE readers and testers
- convert attention into installs, MCP setups, GitHub engagement, and direct builder conversations
- create a repeatable content system that compounds instead of relying on one launch spike

## Product Truths To Market

These claims are grounded in the current repo and docs:

- ACE is a toolkit for adaptive context in agentic systems, not a prompt library.
- ACE uses hybrid retrieval, deterministic merge logic, and a refine loop instead of rewriting whole prompts.
- ACE supports MCP usage, CLI usage, and Python-library usage.
- ACE already exposes concrete product surface area: retrieval, reflection, curation, commit, refine, stats, and playbook dump/import flows.
- ACE can operate in online adaptation mode and warm-start from an existing playbook.

## Category Position

Category name to own: `Agentic Context Engineering`

Category thesis:

- static prompts decay
- full prompt rewrites are brittle
- good agent systems need context that can be retrieved, critiqued, patched, and refined over time

Sharp product line:

`ACE is the adaptive context layer for coding agents.`

Expanded message:

`ACE helps agent systems retrieve the right context, learn from failures, and update reusable instructions through deterministic bullet-level deltas instead of brittle prompt rewrites.`

## Audience Priority

Start with the audience Dan already has the right to talk to.

Primary audience:

- technical builders on X using Claude, Codex, OpenAI, MCP, evals, and agent loops
- engineers already frustrated by long system prompts, context drift, or weak agent memory

Secondary audience:

- framework authors and agent-tool builders
- internal platform teams experimenting with coding agents

Do not start with enterprise-brand messaging. Start with builders who can test the repo this week.

## Message Pillars

Every post, page, and demo should reinforce one of these:

1. `Static prompts do not scale.`
2. `ACE updates context with small deltas, not full rewrites.`
3. `The merge path is deterministic, so the learning loop stays inspectable.`
4. `ACE is MCP-native, so it fits the way builders are already wiring agents.`
5. `Context quality should be measured and refined like code, not treated like copy.`

## Content Engine

Weekly cadence starting the week of March 16, 2026:

- Monday: one category post or thread
- Wednesday: one proof post with code, CLI output, GIF, or screenshot
- Friday: one adoption post with a concrete use case, lesson, or teardown
- Weekly: one longer artifact in the repo or docs that the X posts can point to

Each weekly idea should produce:

- 1 X thread
- 2 to 3 shorter X posts from the same thesis
- 1 visual asset or terminal demo
- 1 repo asset: doc, example, benchmark note, or setup guide

During governed launch weeks, those shorter units can be reply-distribution
touches or logged DM follow-ups instead of net-new public posts. Do not add an
extra public post unless the tracked-link registry and live launch log are
updated first.

This is the compounding rule:

- one idea becomes a week of distribution
- one proof artifact becomes multiple posts
- strong posts become reply material, DM follow-up, and launch-page copy

## Four-Week Campaign

### Week 1: Create the category

Thesis:

- prompt engineering is not enough for long-running agents
- the missing layer is adaptive context engineering

Outputs:

- anchor thread: `Static prompts break. Adaptive playbooks don't.`
- repo doc or README section: `Why ACE exists`
- 30 to 60 second terminal demo showing retrieve -> reflect -> curate -> commit

Primary metric:

- qualified clicks from X to README or docs

### Week 2: Prove the mechanism

Thesis:

- ACE does not hallucinate a new prompt every time
- it retrieves and patches reusable bullets through deterministic ops

Outputs:

- visual post explaining ADD, PATCH, DEPRECATE, and refine
- short comparison post: prompt rewrite vs bullet delta
- technical doc or diagram link that shows the loop end to end

Primary metric:

- saves, reposts, and replies from technical builders

### Week 3: Prove the workflow fit

Thesis:

- ACE fits the actual tooling stack builders use now: MCP, CLI, Python

Outputs:

- MCP setup walkthrough
- post showing a client using ACE tools in practice
- example playbook or worked example that someone can run locally

Primary metric:

- setup completions, inbound questions, and GitHub stars or discussions

### Week 4: Convert to adoption

Thesis:

- ACE is ready for early design partners and active users

Outputs:

- launch thread with best proof artifacts from weeks 1 to 3
- explicit CTA for builders to try ACE and report results
- follow-up thread of lessons from first users

Primary metric:

- demos, serious inbound builder conversations, and repeat usage signals

## Distribution System

Channels to prioritize:

- Dan's X account
- GitHub README and docs as conversion surface
- replies on X to adjacent conversations about MCP, agent memory, prompt drift, and evals
- direct outreach to a small list of high-fit builders after proof posts land

Distribution rules:

- do not post generic AI takes
- only publish content that points to a concrete repo asset, demo, or claim
- reply distribution matters as much as original posts during the first month

## Conversion Path

The current best conversion path is:

`X post -> README or docs -> local install or MCP setup -> first successful run -> conversation, issue, star, or contribution`

Recommended CTA stack:

- top-of-funnel CTA: `Read the thesis`
- mid-funnel CTA: `Run the MCP server or CLI locally`
- bottom-funnel CTA: `Reply or open an issue if you want to test ACE on your agent loop`

## Metrics

Initial 4-week targets:

- 75,000 qualified X impressions from campaign posts
- 2.5% or better engagement rate on thesis and proof posts
- 150 outbound clicks to README, docs, or setup assets
- 25 high-intent adoption signals
- 8 serious builder conversations or demo requests
- 5 repeat-user signals within 14 days

Proxy adoption signals until better telemetry exists:

- GitHub stars
- issues and discussions opened
- replies saying they installed or configured ACE
- direct messages asking for help or sharing results
- repeat mentions from the same builders

## Required Assets And Blockers

Missing assets that directly limit conversion:

- a tighter README top section with category language and a fast CTA
- a short demo asset that proves the loop visually
- one concrete example playbook or worked example
- one benchmark or before-after artifact showing why ACE matters
- basic attribution for docs clicks and setup completions

Current workflow blockers in this shell:

- Paperclip runtime variables are not available, so issue checkout and status sync cannot be performed here
- `bd` is not on `PATH`, so repo issue workflow cannot be executed through the required CLI
- `qmd` is not on `PATH`, so semantic PARA recall is unavailable

## Ready Briefs

### Brief 1: README positioning hardening

Goal:

- make the first screen of the README explain category, problem, and CTA in under 20 seconds

Deliverables:

- revised top section of README
- one category line
- one product line
- one quickstart CTA

Acceptance criteria:

- a new visitor can explain what ACE is, who it is for, and what to do next without scrolling far

### Brief 2: 60-second ACE demo

Goal:

- produce one reusable demo asset for X, docs, and future launch pages

Deliverables:

- terminal or screen recording showing retrieve -> reflect -> curate -> commit or MCP usage
- subtitle-ready script
- one still image or GIF

Acceptance criteria:

- the loop is understandable without narration and can be clipped into multiple posts

### Brief 3: Example-driven adoption doc

Goal:

- give technical builders one runnable example instead of abstract architecture

Deliverables:

- one end-to-end example in docs or examples
- command sequence that works locally
- expected output shown inline

Acceptance criteria:

- a builder can copy the commands and see the value path in one session

### Brief 4: Instrumentation plan

Goal:

- make audience growth and adoption measurable

Deliverables:

- list of funnel events to track
- owner for each event
- interim proxy metrics if product telemetry is not ready

Acceptance criteria:

- weekly review can report reach, profile growth, clicks, adoption, and repeat
  usage with consistent definitions

## Operating Principle

Do not market ACE as vague agent magic.

Market it as a concrete system for:

- retrieving better context
- learning from execution
- updating instructions through deterministic deltas
- improving agent performance without prompt sprawl
