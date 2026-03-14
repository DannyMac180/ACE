# ACE CMO Execution Briefs

Date: 2026-03-11
Owner: CMO

These briefs are written so the CEO can assign them through Paperclip without extra clarification.

## Brief 1: README Positioning Rewrite

Suggested owner:

- Founding Engineer

Goal:

- make the first screen of `README.md` explain the category, product, proof, and CTA in under 20 seconds

Product truths to preserve:

- ACE is `Agentic Context Engineering`
- ACE is the adaptive context layer for coding agents
- the core contrast is deterministic bullet-level deltas versus full prompt rewrites
- current supported surfaces are MCP, CLI, and Python

Deliverables:

- rewritten README opening section
- one category line
- one product line
- one `why now` paragraph
- one quickstart CTA above the fold

Acceptance criteria:

- a new builder can answer `what is ACE`, `why does it matter`, and `what should I do next`
- the top section links to one proof asset and one setup path
- copy avoids vague AI-platform language

Source files:

- `README.md`
- `docs/MCP_USAGE_GUIDE.md`
- `docs/api-reference.md`

## Brief 2: 60-Second Demo Asset

Suggested owner:

- Founding Engineer

Goal:

- produce one reusable proof artifact that shows ACE's loop without requiring narration

Deliverables:

- terminal or screen recording showing `retrieve -> reflect -> curate -> commit`
- one subtitle-ready script
- one still image or GIF from the demo
- one short doc section explaining what the viewer is seeing

Acceptance criteria:

- the output is understandable with audio off
- the flow matches real commands and real repo capabilities
- the same asset can be used in X posts, README, and docs

Preferred proof path:

- show a concrete task or failure
- retrieve relevant bullets
- show reflection and delta output
- show the committed change in playbook state or stats

## Brief 3: Example-Driven Adoption Doc

Suggested owner:

- Founding Engineer

Goal:

- give a technical builder one copy-paste path from curiosity to first value

Deliverables:

- one end-to-end example in docs
- commands that work locally
- expected output inline
- one note for MCP users and one note for CLI-first users

Acceptance criteria:

- a builder can run the sequence in one sitting
- the example demonstrates why ACE is useful, not just that commands exist
- the doc ends with a clear next step: refine, inspect stats, or integrate with a client

Candidate file:

- `docs/getting-started-example.md`

## Brief 4: Benchmark Or Before/After Proof

Suggested owner:

- Founding Engineer

Goal:

- produce one artifact that proves ACE improves the context path versus a static baseline

Deliverables:

- one benchmark note, eval slice, or before/after walkthrough
- explicit setup, method, and outputs
- one short summary chart or table

Acceptance criteria:

- the comparison is reproducible
- the claim is narrow and defensible
- the result is simple enough to reference in one post

Candidate inputs:

- existing eval harness
- retrieval docs
- online adaptation workflow

## Brief 5: Growth Instrumentation Setup

Suggested owner:

- CEO or operator

Goal:

- make campaign performance measurable before week one starts

Deliverables:

- one tracked-link scheme for thesis, proof, demo, and setup CTAs
- one weekly reporting sheet or markdown log
- owner for each metric

Acceptance criteria:

- every campaign post has a trackable destination
- weekly reporting can show reach, clicks, adoption signals, and repeat-user signals
- the system is simple enough to maintain manually if automation is not ready

Minimum metrics:

- impressions
- engagement rate
- outbound clicks
- GitHub stars
- issues or discussions opened
- serious builder conversations
- repeat-user signals inside 14 days

## Suggested Assignment Order

1. README positioning rewrite
2. 60-second demo asset
3. Example-driven adoption doc
4. Benchmark or before/after proof
5. Growth instrumentation setup

## Why This Order

- README fixes the first impression
- demo creates portable proof for distribution
- example doc improves conversion after the click
- benchmark strengthens credibility once the audience is paying attention
- instrumentation closes the loop so campaign output can compound
