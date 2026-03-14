# ACE CMO Design Partner Engine

Date: 2026-03-11
Owner: CMO

## Objective

Turn early audience attention into real adoption signals.

From March 16, 2026 to April 10, 2026, the goal is to convert high-signal builders from Dan's existing audience into:

- design partner conversations
- active ACE testers
- repeat usage feedback
- public proof points the broader market can trust

This system exists so launch attention does not die as passive impressions.

## Primary Outcome

Create a repeatable path from:

`post engagement -> qualifying conversation -> local install or MCP setup -> real workflow feedback -> public or private proof artifact`

## Target Counts

Four-week targets:

- 20 high-signal inbound or outbound prospects
- 10 qualified design-partner conversations
- 5 builders who complete a real ACE setup
- 3 builders who report at least one meaningful workflow result
- 2 reusable proof artifacts drawn from real external usage

## Ideal Design Partner Profile

Prioritize people who already feel the problem.

Tier 1:

- builders actively posting about coding agents, MCP, evals, prompt drift, or agent memory
- developers already running agent loops in real repos
- framework or tooling authors who can influence adjacent audiences

Tier 2:

- internal platform engineers experimenting with coding agents
- technical founders building agent-heavy products

Deprioritize for now:

- enterprise buyers asking only for roadmap slides
- generic AI enthusiasts without a real workflow to test
- people who want broad consulting instead of product usage

## Qualification Rubric

Score each prospect from 0 to 6.

Give one point for each:

- has a live agent workflow today
- can describe a repeated failure pattern
- has enough technical fluency to run ACE locally
- is willing to share logs, failures, or setup friction
- can test within 7 days
- has audience or team leverage beyond a single experiment

Priority rule:

- `5-6`: immediate follow-up
- `3-4`: nurture and invite into the next proof asset
- `0-2`: low priority unless strategically important

## Where Prospects Come From

Primary sources:

- replies to Dan's X thesis, proof, and adoption posts
- DMs after strong X engagement
- GitHub issues and discussions from real workflow questions
- Hacker News or Reddit commenters who show concrete pain

Secondary sources:

- manual outreach to a short list of builders already known to Dan
- warm intros from operators or friendly technical founders

## Trigger Events

Move someone into the pipeline when they do any of the following:

- reply with a concrete failure pattern
- ask how ACE fits MCP, CLI, or an existing agent stack
- star the repo and open an issue or discussion
- ask for a demo, walkthrough, or setup help
- compare ACE to prompt engineering, memory systems, or eval tooling in a detailed way

## Offer

The offer should stay narrow and credible.

Position it like this:

`We are looking for a small number of builders running coding agents who want to test adaptive context on a real workflow. If you have a repeat failure pattern, we want to help encode it and learn from the result.`

What they get:

- direct setup help
- a fast path to shape the examples and docs
- early influence on the playbook and workflow

What ACE gets:

- real failure cases
- proof artifacts
- language the market will actually use

## Outreach Motion

### Inbound

For every high-signal reply or DM:

1. respond publicly if useful so the conversation creates social proof
2. move to DM with one clear next step
3. ask for the workflow, stack, and failure pattern
4. decide whether to send docs, offer a short call, or ask for a GitHub issue

### Outbound

Use outbound only for a short, high-fit list.

Good outbound targets:

- people already talking about prompt drift or agent memory failures
- builders who engage with Dan's posts multiple times
- technical peers who can evaluate fast and give sharp feedback

Bad outbound targets:

- cold broad lists
- non-technical buyers
- anyone who would need heavy education before even understanding the problem

## Message Templates

### Public Reply

`This is exactly the kind of workflow we want to test ACE against. The thesis is adaptive context instead of a bigger static prompt. If you're open to it, DM me the failure pattern or open a GitHub issue and we'll map it to a concrete setup path.`

### First DM

`You look like a strong fit for early ACE testing. If you already run coding agents, send me three things: your current stack, one repeated failure pattern, and whether you prefer MCP or CLI. If it fits, I'll point you to the fastest setup path or suggest a short call.`

### Qualifying Follow-Up

`Helpful to know before we go further: are you trying to improve a real workflow this week, or are you still exploring the space? ACE is most useful when there is already a concrete repeat failure to work on.`

### Call Invite

`This looks like a good design-partner case. If you want, let's do a short working session focused on one failure pattern, one setup path, and one success criterion.`

### Post-Call Follow-Up

`Thanks for the session. The next useful step is to run the agreed setup path and send back the first friction point or result. We care more about concrete failure data than generic impressions.`

## Routing Logic

Choose the next step based on friction and intent.

Send docs only when:

- the prospect is technically capable
- the workflow is clear
- setup friction is likely low

Offer a short call when:

- the workflow is real but integration questions are blocking action
- the person is high leverage
- the failure pattern could become a public proof asset

Ask for a GitHub issue or discussion when:

- the problem statement is useful for the community
- the prospect is willing to document the workflow
- the thread could help future users convert

## Call Structure

Target length:

- 20 minutes

Agenda:

1. current agent workflow
2. repeated failure pattern
3. where ACE fits or does not fit
4. setup path and success criterion
5. what proof can be made public later

Exit condition:

- one concrete next step with an owner and a date

## Internal Handoff Requirements

Capture these fields for every qualified prospect:

- name or handle
- source channel
- stack
- failure pattern
- qualification score
- chosen next step
- date of last contact
- outcome after 7 days

If the workflow reveals product gaps, route them into GitHub issues or Paperclip tasks instead of burying them in DMs.

## Weekly Operating Cadence

Monday:

- review last week's replies, DMs, issues, and discussion threads
- select 5 to 8 high-signal prospects for follow-up

Tuesday to Thursday:

- send or answer DMs
- run short design-partner calls
- push qualified builders toward setup completion

Friday:

- log results
- identify the best public proof point
- feed language and objections back into next week's content

## Scoreboard

Track weekly:

- high-signal replies
- DMs started
- qualified prospects
- calls booked
- calls completed
- setups started
- setups completed
- repeat-user signals
- public proof candidates

Track conversion by stage:

- `reply or DM -> qualified`
- `qualified -> call`
- `call -> setup started`
- `setup started -> meaningful result`

## Success Criteria

This engine is working if, within four weeks:

- Dan can name the top recurring objections from real builders
- ACE has at least 5 external workflow tests
- at least 2 real user stories can be turned into content or docs
- the next content cycle is based on actual adoption friction rather than internal guesses

## Dependencies

This system works better when these existing briefs are executed:

- `plans/2026-03-11-cmo-execution-briefs.md` for README, demo, example doc, proof artifact, and instrumentation
- `plans/2026-03-11-cmo-launch-copy-pack.md` for post and DM copy
- `plans/2026-03-11-cmo-channel-strategy.md` for where to source demand

## Recommended Next Assignment

If the CEO wants the fastest path from attention to adoption, the next governed work should be:

1. README positioning rewrite
2. growth instrumentation setup
3. design partner outreach execution against week-one replies and GitHub signals
