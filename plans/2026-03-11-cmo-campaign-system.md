# ACE CMO Campaign System

Date: 2026-03-11
Owner: CMO

## North Star

From March 16, 2026 through April 10, 2026, turn `Agentic Context Engineering` from a repo label into a category builders can repeat back.

Primary business outcome:

- convert Dan's X audience into ACE installs, MCP setups, GitHub engagement, and serious builder conversations

Primary operating outcome:

- make every campaign week produce reusable proof assets, not one-off posts

## Message Stack

### Category

`Agentic Context Engineering`

Definition:

- the discipline of retrieving, critiquing, patching, and refining agent context over time instead of treating prompts as static text

### Product Line

`ACE is the adaptive context layer for coding agents.`

Expanded line:

- ACE helps coding agents retrieve the right context, learn from execution, and update reusable instructions through deterministic bullet-level deltas instead of brittle prompt rewrites.

### Enemy

The contrast should stay consistent across posts, docs, and demos:

- static prompts decay
- prompt rewrites are brittle
- long system prompts hide failure instead of teaching the system

### Proof Pillars

Only use claims the repo can support today:

1. ACE uses hybrid retrieval instead of stuffing the whole playbook into the prompt.
2. ACE commits deterministic delta operations like `ADD`, `PATCH`, `DEPRECATE`, and `INCR_*`.
3. ACE supports MCP, CLI, and Python usage patterns.
4. ACE includes refine and stats flows, so context quality can be inspected and improved.
5. ACE can run in an online adaptation loop and warm-start from an existing playbook.

## Audience Map

### Primary

- builders on X already working with Claude, Codex, MCP, evals, and coding agents
- engineers frustrated by prompt drift, context bloat, or weak agent memory

### Secondary

- framework maintainers
- agent-tool authors
- internal platform teams experimenting with coding agents

### Audience Promise

ACE should read like:

- technically credible
- inspectable instead of magical
- useful this week, not after a six-month platform build

## Campaign Structure

Each week runs one thesis across three post types:

1. Category post
2. Proof post
3. Adoption post

Each week also needs one repo asset:

- README section
- docs page
- runnable example
- benchmark note
- demo clip or GIF

Compounding rule:

- one thesis becomes one thread, two or three short posts, one reply set, one proof artifact, and one outbound CTA

## Four-Week Calendar

### Week 1: Create the category

Theme:

- prompt engineering stops where adaptive context begins

Monday post:

- thread title: `Static prompts break. Adaptive playbooks don't.`
- CTA: read the thesis in the README or docs

Wednesday proof:

- 30 to 60 second terminal demo showing `retrieve -> reflect -> curate -> commit`
- CTA: watch the loop and try it locally

Friday adoption:

- post: `If your coding agent keeps repeating the same mistakes, your prompt is doing too much and learning too little.`
- CTA: reply if you want to test ACE on a real agent loop

Success metric:

- qualified clicks to README or docs

### Week 2: Prove the mechanism

Theme:

- ACE patches context with deterministic deltas instead of rewriting the whole prompt

Monday post:

- diagram or text carousel showing `ADD`, `PATCH`, `DEPRECATE`, `INCR_HELPFUL`, and `refine`
- CTA: compare prompt rewrite vs bullet delta

Wednesday proof:

- short clip or screenshot sequence of a delta being proposed and merged
- CTA: inspect the delta path in the docs

Friday adoption:

- post showing one concrete failure pattern and the reusable bullet it should have created
- CTA: open an issue or reply with a failure mode you want to encode

Success metric:

- saves, reposts, and technical replies

### Week 3: Prove workflow fit

Theme:

- ACE fits how builders already wire agent systems: MCP, CLI, and Python

Monday post:

- setup walkthrough for ACE as an MCP server
- CTA: run it with your client this week

Wednesday proof:

- worked example using a real ACE command sequence with expected output
- CTA: copy the commands and verify locally

Friday adoption:

- post: `You do not need a giant agent platform to start measuring context quality.`
- CTA: star the repo and report your first setup result

Success metric:

- setup completions, GitHub stars, inbound setup questions

### Week 4: Convert to adoption

Theme:

- early builders can now pressure-test ACE and shape the product

Monday post:

- launch thread synthesizing the best proof artifacts from weeks 1 to 3
- CTA: install ACE or run the MCP setup

Wednesday proof:

- early user lesson, benchmark note, or before/after artifact
- CTA: share your results

Friday adoption:

- call for design partners running coding agents with repeat failure patterns
- CTA: DM or open an issue with your use case

Success metric:

- serious builder conversations, demos, repeat-user signals

## Content Production Workflow

### Monday build sequence

1. Pick one thesis.
2. Pull one proof artifact from the repo.
3. Write one thread and two shorter derivatives.
4. Publish the highest-context version first.

### Wednesday build sequence

1. Ship a visual or terminal artifact.
2. Cut it into one proof post plus one reply asset.
3. Link directly to the matching repo surface.

### Friday build sequence

1. Translate the week's proof into a builder workflow.
2. Ask for a concrete action: install, star, issue, reply, or demo request.
3. Follow up manually with every high-signal responder.

## Distribution System

Priority channels:

- Dan's X account
- GitHub README
- ACE docs
- replies on adjacent X posts about MCP, agent memory, evals, prompt drift, and coding agents

Distribution rules:

- never post without a concrete proof, artifact, or opinionated claim
- every campaign post should point at a repo surface or setup path
- reply distribution is a first-class channel, not cleanup work

## Founder-Led Reply Strategy

Dan should actively reply in adjacent conversations when the topic matches one of these:

- prompt drift
- long system prompts
- agent memory failures
- MCP workflow design
- eval regressions caused by weak context

Reply pattern:

1. name the pain in one sentence
2. introduce the ACE framing in one sentence
3. link the proof asset, not the whole repo by default

## CTA Ladder

Top of funnel:

- `Read why ACE exists`

Middle of funnel:

- `Run the MCP server or CLI example`

Bottom of funnel:

- `Reply, open an issue, or DM if you want to test ACE on your agent loop`

## Measurement System

Week-one scoreboard status:

- for Monday, March 16, 2026 through Friday, March 20, 2026, the only live scoreboard is `plans/2026-03-16-cmo-launch-operations-log.md`
- this section still defines targets and measurement language, but it is not the active reporting sheet

### Weekly scoreboard

Track these every Monday:

| Metric | Definition | Target |
|---|---|---|
| Qualified X impressions | Impressions from ACE campaign posts and related replies | 75,000 over 4 weeks |
| Engagement rate | `(likes + replies + reposts + bookmarks) / impressions` on campaign posts | 2.5%+ |
| Outbound clicks | Clicks from X to README, docs, or setup assets | 150 over 4 weeks |
| Adoption signals | Stars, issues, discussions, replies saying they installed or configured ACE | 25 |
| Serious conversations | DMs, demo asks, or detailed replies from relevant builders | 8 |
| Repeat-user signals | Same builder returns with a second touchpoint inside 14 days | 5 |

### Instrumentation requirements

Use distinct tracked links for each campaign class:

- thesis link
- demo link
- setup link
- proof link

Minimum tracking implementation:

- one unique URL per CTA destination
- weekly manual log of X metrics
- weekly manual log of GitHub stars, issues, discussions, and inbound conversations

## Operating Rhythm

Monday:

- publish thesis post
- update scoreboard
- confirm the week's proof artifact

Wednesday:

- publish proof post
- collect replies and objections
- record reusable language that resonates

Friday:

- publish adoption post
- DM or reply to high-signal builders
- log outcomes and friction

## Risks

1. ACE currently reads more like a toolkit than a category-defining product on first impression, so messaging must tighten the README and demo path quickly.
2. The repo needs a stronger visual proof asset, otherwise category claims will outrun evidence.
3. If tracked links and manual logging are not set up before campaign week one, audience growth will be noisy and hard to learn from.
4. There is still no assigned Paperclip marketing issue, so cross-agent execution must be formally assigned before coordinated delivery can happen.

## Immediate Next Assets

1. README top-section rewrite with category language and one fast CTA.
2. One 60-second demo asset for X and docs.
3. One example-driven adoption doc with copy-paste commands.
4. One measurement sheet or log template for weekly reporting.
