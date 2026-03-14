# ACE Week-One Live Capture Sheet

Date: 2026-03-13
Owner: CMO
Operator: Dan

## Purpose

Use this file as the raw working note during the live week-one launch window.

This is the fast capture layer that sits between the live X session and the
canonical scoreboard in `plans/2026-03-16-cmo-launch-operations-log.md`.

Capture here first when speed matters, then normalize the important entries
into the launch log before the day ends.

## Operating Rules

- Keep this file open during every publish and reply session.
- Copy the exact live post URL here within five minutes of publishing.
- Use exact post codes and follow-up codes from
  `plans/2026-03-16-cmo-launch-operations-log.md`.
- Use `plans/2026-03-13-cmo-week-one-response-bank.md` for exact public reply,
  DM, and qualification copy instead of improvising in the moment.
- Only record raw observations here; move final rows into the canonical launch
  log the same day.
- Capture profile-surface changes here first when the bio, website link, or
  pinned post changes so the same-day transfer into the launch log stays fast.
- If a reply or DM surfaces a repeated blocker, copy it into the friction
  section here before it gets lost.

## Capture Windows

Use these default checkpoints for each launch motion:

- `publish`: immediately after the post goes live
- `+30m`: first reaction and reply quality check
- `+3h`: first meaningful traffic and conversation check
- `EOD`: final same-day snapshot before logging closes

Default operating windows in Eastern Time:

| Day | Live window | Mid-session check | Final same-day checkpoint |
| --- | --- | --- | --- |
| Monday, March 16, 2026 | publish `11:00-11:30` ET | `+30m` and `+3h` after publish | `17:30` ET |
| Tuesday, March 17, 2026 | distribution pass `12:00-13:00` ET | `12:30` ET | `17:30` ET |
| Wednesday, March 18, 2026 | publish `11:00-11:30` ET | `+30m` and `+3h` after publish | `17:30` ET |
| Thursday, March 19, 2026 | follow-up pass `12:00-13:00` ET | `12:30` ET | `17:30` ET |
| Friday, March 20, 2026 | publish `10:30-11:00` ET | `+30m` and `+3h` after publish | `17:30` ET |

## Checkpoint Response Rules

Use these when checkpoint numbers are weak or unexpectedly strong.

These are the raw-capture counterparts to the `Checkpoint Decision Log` in
`plans/2026-03-16-cmo-launch-operations-log.md`.

| Motion | Checkpoint | If this is true | Required move |
| --- | --- | --- | --- |
| Monday anchor thread | `+30m` | engagement rate under `1.5%` or fewer than `2` meaningful builder replies | spend the next `30` minutes on thesis-clarifying public replies with `reply_w1_thread_readme`; do not change the tracked URL and do not add a new post |
| Monday anchor thread | `+3h` | README clicks under `10` | pin the thread if the live URL is verified, run `2-3` adjacent-conversation replies from the Tuesday watchlist using `reply_w1_thread_readme`, and note that Wednesday must sharpen the proof hook instead of repeating more category language |
| Tuesday distribution pass | EOD | fewer than `2` builders moved into DM or the design-partner pipeline | queue `2` proof-skeptic or workflow-fit targets for Wednesday from the watchlist and tighten Wednesday first replies around inspectability plus one concrete failure-pattern ask |
| Wednesday proof post | `+3h` | proof clicks under `5` or saves or bookmarks under `5` | use `reply_w1_proof_demo` on the strongest technical skeptic, DM the highest-fit responder for one workflow plus repeated failure pattern, and make Friday lead with workflow fit instead of more loop mechanics |
| Thursday follow-up pass | EOD | fewer than `2` qualified prospects have a dated next step | make Friday's first reply and first DM ask directly for one repeated failure pattern, default to `dm_w1_example_doc`, and push the best public-fit case toward `issue_w1_failure_pattern` |
| Friday adoption post | `+3h` or EOD | fewer than `2` setup-start signals or no public workflow report exists yet | spend the remaining window on `1:1` follow-up with the top `3` responders, route the best public-fit workflow into `issue_w1_failure_pattern`, and log the miss explicitly if no builder qualifies |

## Checkpoint Decision Scratchpad

Use this when a checkpoint changes the next move.

Record the raw trigger and the action here first, then move the final version
into the canonical `Checkpoint Decision Log` in
`plans/2026-03-16-cmo-launch-operations-log.md`.

| Date | Checkpoint | Trigger observed | Decision made | Action shipped | Transferred to launch log | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-03-16 | `+30m` anchor-thread review |  |  |  | pending | use only if Monday needs an immediate hook or reply adjustment |
| 2026-03-16 | `+3h` anchor-thread review |  |  |  | pending | use when Tuesday distribution or Wednesday proof positioning changes |
| 2026-03-17 | EOD distribution review |  |  |  | pending | use when Wednesday needs a sharper hook, CTA, or routing path |
| 2026-03-18 | `+3h` proof-post review |  |  |  | pending | use when Friday should emphasize setup fit, mechanism clarity, or public workflow capture |
| 2026-03-19 | EOD follow-up review |  |  |  | pending | use when Friday adoption routing needs to change before publish |
| 2026-03-20 | `+3h` or EOD adoption review |  |  |  | pending | use when week two needs an explicit messaging, docs, or routing fix |

## Launch Incident Scratchpad

Use this the moment a live publish or follow-up surface breaks.

This is the raw counterpart to the `Launch Contingency Log` in
`plans/2026-03-16-cmo-launch-operations-log.md`.

| Time | Motion | Failure mode | What broke | Immediate containment | Next check time | Transferred to launch log | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  | pending | use this for scheduler failures, broken tracked URLs, missing proof assets, or publish-window slips |

## Quick Routing Rules

Use this triage in the moment:

| Signal | Action | Code |
| --- | --- | --- |
| category curiosity but no setup intent | public reply or DM to README | `reply_w1_thread_readme` or `dm_w1_readme` |
| asks how to start | DM the worked example | `dm_w1_example_doc` |
| asks specifically about MCP | DM the MCP guide | `dm_w1_mcp_guide` |
| asks for implementation detail | DM the API reference | `dm_w1_api_ref` |
| adjacent conversation clearly matches ACE pain | public reply with the sharpest thesis or setup path, then log it as a distribution pass | `reply_w1_thread_readme` or `reply_w1_adoption_example` |
| can clearly document the workflow and repeated failure publicly | send the workflow issue template and log it as public intake | `issue_w1_failure_pattern` |
| shares a concrete repeated failure pattern | log in pipeline and friction, then qualify for call or setup help | use matching reply or DM code |

## Scoreboard Signal Tags

Use these raw tags in notes or next-step cells so the transfer into
`plans/2026-03-16-cmo-launch-operations-log.md` stays consistent.

| Tag | Use when | Maps to scoreboard |
| --- | --- | --- |
| `adoption_signal` | a builder takes a non-passive action such as starring the repo, opening an issue, asking for setup help, or confirming they tried ACE | `Adoption signals` |
| `serious_conversation` | the exchange is multi-turn and tied to a real workflow, repeated failure pattern, or concrete setup blocker | `Serious conversations` |
| `setup_start` | the builder asks for or begins a concrete setup path through the worked example, MCP guide, API route, or workflow issue | `Setup starts` |
| `setup_complete` | the builder explicitly confirms they completed a meaningful ACE setup step | `Setup completions` |
| `repeat_user_signal` | the same builder returns within `14` days with a new result, blocker, or follow-up question | `Repeat-user signals` |

Raw-tag rules:

- click metrics still come from tracked URLs, not these tags
- profile visits and new follows still come from captured platform snapshots,
  not from builder-signal tags
- count people once per tag per week unless the second touch qualifies as
  `repeat_user_signal`
- if the signal is unclear, capture the quote or action first and decide during
  same-day transfer

## Qualification Quickscore

Give one point for each:

- has a live agent workflow today
- can describe a repeated failure pattern
- has enough technical fluency to run ACE locally
- is willing to share logs, failures, or setup friction
- can test within 7 days
- has audience or team leverage beyond a single experiment

Route by score:

- `5-6`: move into the pipeline immediately and consider a short working session
- `3-4`: send the most relevant doc path, log the next step, and follow up
- `0-2`: answer briefly, log the objection if useful, and keep attention light

## Sunday, March 15, 2026 Preflight Scratchpad

Use this to confirm launch-week readiness before Monday.

Use this as the raw working surface, then transfer the final completion
timestamps and evidence into the `Sunday Preflight Completion Receipt` in
`plans/2026-03-16-cmo-launch-operations-log.md` before Sunday close.

| Check | Status | Notes |
| --- | --- | --- |
| `plans/2026-03-15-cmo-week-one-launch-calendar.ics` imported into live calendar | pending | verify the import created the Monday through Friday live windows plus the same-day reply and logging check blocks in Eastern Time |
| Monday publish window blocked on calendar (`11:00-11:30` ET) | pending | confirm the imported event matches this slot |
| Tuesday distribution pass blocked on calendar (`12:00-13:00` ET) | pending | confirm the imported event matches this slot |
| Wednesday publish window blocked on calendar (`11:00-11:30` ET) | pending | confirm the imported event matches this slot |
| Thursday follow-up pass blocked on calendar (`12:00-13:00` ET) | pending | confirm the imported event matches this slot |
| Friday publish window blocked on calendar (`10:30-11:00` ET) | pending | confirm the imported event matches this slot |
| week-one bio updated from `plans/2026-03-13-cmo-week-one-profile-surface.md` | pending | |
| X website field updated to `profile_w1_readme` | pending | |
| Monday anchor draft pasted into live X composer or scheduler as a connected `9`-post thread | pending | note the scheduler or composer surface used so Monday can be verified fast if the thread breaks |
| Wednesday proof draft pasted into live X composer or scheduler | pending | confirm the tracked URL stayed exact and `docs/assets/ace-proof-demo/still.svg` is still attached unless an approved replacement was logged |
| Friday adoption draft pasted into live X composer or scheduler | pending | confirm the tracked URL stayed exact and the worked-example CTA still points to `w1_adoption_example` |
| week-one response bank open and ready | pending | |
| `docs/assets/ace-proof-demo/still.svg` confirmed as proof visual | pending | |
| launch operations log open and ready | pending | |
| this live capture sheet open and ready | pending | |
| Tuesday distribution watchlist prepared | pending | base seed pool is loaded below; before Sunday close, swap in the strongest live Monday engagers first and keep at least 5 Tuesday targets |
| Thursday follow-up watchlist prepared | pending | before Sunday close, swap in the strongest live Wednesday engagers first and keep at least 5 Thursday targets |

## Sunday Go Or No-Go Scratchpad

Use this only on Sunday, March 15, 2026 after the preflight checklist above is
updated.

This is the raw note that feeds the canonical `Sunday Go Or No-Go Gate` in
`plans/2026-03-16-cmo-launch-operations-log.md`.

| Time checked | Gate status | Missing item or blocker | Next check time | Transferred to launch log | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-03-15 20:00 ET | pending |  |  | pending | mark `go` only if the governed profile, calendar, each scheduler draft, and each validated watchlist are all complete with evidence |

## Audience Baseline Snapshot

Capture this once on Sunday, March 15, 2026 after the governed week-one bio
and website link are live.

This is the raw note that feeds the `Audience Baseline Snapshot` in
`plans/2026-03-16-cmo-launch-operations-log.md`.

If X only shows a rolling or partial view, record the exact surface and
timestamp so later `Profile visits` and `New follows` comparisons are still
auditable.

| Captured at | Starting follower count | Starting profile-visits reading | Source surface | Transferred to launch log | Notes |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  | pending | fill this immediately after the Sunday profile update so Monday through Friday audience lift has a fixed starting point |

## Profile Surface Raw Capture

Use this table for the governed week-one bio, website link, and Monday pinning
motions before transferring the final details into the `Profile Surface Log` in
`plans/2026-03-16-cmo-launch-operations-log.md`.

| Date | Motion | Code | Live profile or pinned post URL | Profile visits snapshot | New follows snapshot | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-15 | apply week-one bio and website link | `profile_w1_readme` |  | baseline snapshot recorded above before Monday publish | baseline snapshot recorded above before Monday publish | pending | paste the live website-field URL here, note if the bio had to be shortened for X character limits, and make sure the audience baseline row above is filled |
| 2026-03-16 | pin the Monday anchor thread after URL verification | `profile_w1_readme` |  | `+3h` and `EOD` Monday profile visits | `+3h` and `EOD` Monday follows | pending | leave the README-first website link in place through Friday unless the launch log records an exception |

## Tuesday And Thursday Watchlist

Prepare this before Monday so distribution work starts from live targets instead
of a blank search tab.

Fill order:

1. start with qualified or nearly qualified people who already engaged with the
   Monday thread or Wednesday proof post
2. add technical builders Dan already follows or regularly interacts with when
   they are discussing prompt drift, agent-memory failures, MCP workflow
   friction, or eval regressions
3. use the search seeds below only after the first two pools are exhausted or
   too weak

Selection rules:

- prioritize builders or threads already talking about prompt drift, long system
  prompts, agent memory failures, MCP workflow design, eval regressions, or
  repeated setup friction
- prefer accounts with technical credibility, clear workflow context, or
  visible team leverage over generic AI chatter
- queue at least `5` Tuesday candidates and at least `5` Thursday candidates
- note the most likely first routing code before launch week starts so reply
  choices are faster in the moment
- replace stale or low-signal targets during the week, but keep the original
  notes so the watchlist becomes future distribution memory

## Watchlist Search Seeds

Use these search prompts to find adjacent conversations faster on X when the
watchlist needs more targets. Prefer posts from the last `24-72h` with clear
workflow pain over generic AI commentary.

| Day | Search seed | What qualifies the target | Default first code |
| --- | --- | --- | --- |
| Tuesday | `"prompt drift" OR "system prompt" ("agent" OR "coding agent")` | the post describes context bloat, brittle prompting, or repeated instructions | `reply_w1_thread_readme` |
| Tuesday | `("agent memory" OR "memory for agents") (broken OR failing OR context)` | the post describes retrieval misses, stale memory, or weak adaptation | `reply_w1_thread_readme` |
| Tuesday | `("MCP" OR "Model Context Protocol") (workflow OR server OR tooling)` | the post is about real tool orchestration or context handoff, not protocol news alone | `reply_w1_adoption_example` |
| Tuesday | `("eval regression" OR "agent evals") (prompt OR context OR memory)` | the post names recurring regressions or unstable agent behavior | `reply_w1_thread_readme` |
| Thursday | `notifications tab plus Monday and Wednesday engager list` | start with everyone who engaged on Monday or Wednesday and still looks qualified | `reply_w1_proof_demo` |
| Thursday | `"how do you test" ("coding agent" OR "agent workflow")` | the post asks for proof, measurement, or a concrete setup path | `reply_w1_proof_demo` |
| Thursday | `("repeated failure" OR "keeps failing") ("agent" OR "workflow")` | the post includes one failure pattern that could turn into a DM or issue intake | `reply_w1_adoption_example` |
| Thursday | `("MCP" OR "CLI") ("setup" OR "integration") ("agent" OR "workflow")` | the post asks operational fit questions rather than high-level architecture debate | `reply_w1_adoption_example` |

Watchlist status rule:

- use `queued` for promising targets not yet contacted
- use `contacted` after the public reply ships
- use `dm` after the next step moves private
- use `parked` when the target is low signal or no longer timely

Current seed-pool rule:

- this fallback pool was refreshed on Friday, March 13, 2026 from current
  public technical-builder accounts and recent adjacent-conversation surfaces
- before Thursday, replace any colder seed with warmer Monday or Wednesday
  engagers first; the seed pool exists so Dan does not start from a blank tab

| Day | Handle or post URL | Why this target fits ACE | Pain signal to match | Likely first code | Backup code | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tuesday | `@simonw` | practical coding-agent and agent-loop notes attract builders who care about inspectability and real tool behavior | agentic-loop safety, context drift, or tool reliability | `reply_w1_thread_readme` | `dm_w1_readme` | queued | prioritize when the thread already names failures, prompts, or long-lived agent state |
| Tuesday | `@swyx` | high-signal AI-engineering audience discussing MCP, tooling, and agent stacks can compound category reach fast | MCP workflow design or prompt-architecture drift | `reply_w1_adoption_example` | `dm_w1_example_doc` | queued | best when the conversation is operational rather than category-only |
| Tuesday | `@jxnlco` | practical evals and LLM-ops audience where failure-pattern language lands well | eval regressions, weak reliability, or repeated failures | `reply_w1_thread_readme` | `dm_w1_readme` | queued | use the thesis angle first, then move to DM if the workflow is real |
| Tuesday | `@HamelHusain` | deep evals and failure-analysis discussions often surface reusable workflow pain | repeated failure patterns or eval blind spots | `reply_w1_thread_readme` | `dm_w1_example_doc` | queued | good route when the thread already asks how to turn failures into product learning |
| Tuesday | `@latentspacepod` | AI-engineering media node surfaces adjacent builder debates Dan's audience already watches | coding-agent architecture, memory, or MCP tooling | `reply_w1_adoption_example` | `dm_w1_example_doc` | queued | use when a new episode, clip, or quote-post triggers implementation questions |
| Thursday | `@cursor_ai` | large pool of active coding-agent users who often ask workflow-fit questions | workflow fit, setup friction, or adoption intent | `reply_w1_adoption_example` | `dm_w1_example_doc` | queued | replace with warmer Monday or Wednesday engagers first if they appear |
| Thursday | `@leerob` | technical product audience discussing developer workflow changes and tool fit can turn proof into adoption intent | dev-workflow fit, proof skepticism, or adoption blockers | `reply_w1_proof_demo` | `dm_w1_example_doc` | queued | best after Wednesday proof when the objection is whether ACE fits a real stack |
| Thursday | `@never_settles_` | hands-on MCP and browser-workflow debugging threads sit close to ACE setup pain | MCP integration friction or repeated workflow failures | `reply_w1_adoption_example` | `dm_w1_mcp_guide` | queued | prioritize if the thread already includes toolchain details or debug logs |
| Thursday | `@willmcgugan` | terminal and agent-workflow audience has real implementation instincts and cares about inspectability | terminal agent UX, inspectability, or setup friction | `reply_w1_proof_demo` | `dm_w1_api_ref` | queued | useful for proof-first follow-up when README thesis is no longer enough |
| Thursday | `@amanrsanger` | coding-agent builder audience is likely to care about concrete setup and repeat failures | coding-agent workflow fit or repeated failure pattern | `reply_w1_adoption_example` | `dm_w1_example_doc` | queued | keep only if warmer Monday or Wednesday engagers do not fully populate Thursday |

## Monday, March 16, 2026 Raw Capture

Motion: anchor thread
Post code: `w1_thread_readme`

### Publish Facts

- publish time:
- live post URL:
- exact tracked URL used:
- first reply posted:
- next checks due: `+30m`, `+3h`, `17:30 ET`

### Metrics

| Window | Impressions | Engagements | Engagement rate | Link clicks | Profile visits | New follows | Bookmarks | Reposts | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| publish |  |  |  |  |  |  |  |  |  |
| +30m |  |  |  |  |  |  |  |  |  |
| +3h |  |  |  |  |  |  |  |  |  |
| EOD |  |  |  |  |  |  |  |  |  |

### High-Signal Replies

| Handle | Public reply needed | Failure pattern or question | Qual score 0-6 | Move to DM | Next step |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

### DMs Started

| Handle | Code used | What they want | Next action due | Status | Notes |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

### Friction Notes

- 

## Tuesday, March 17, 2026 Raw Capture

Motion: adjacent conversation distribution pass

### Distribution Passes

| Handle or post URL | Topic | Pain matched | Code used | Public reply sent | Move to DM | Qual score 0-6 | Next step |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |

### Metrics

| Window | Adjacent replies shipped | DMs started | Qualified interactions | README clicks influenced | Profile visits influenced | New follows influenced | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `12:30 ET` |  |  |  |  |  |  |  |
| EOD |  |  |  |  |  |  |  |

### Friction Notes

- 

## Wednesday, March 18, 2026 Raw Capture

Motion: proof post
Post code: `w1_proof_demo`
Visual: `docs/assets/ace-proof-demo/still.svg`

### Publish Facts

- publish time:
- live post URL:
- exact tracked URL used:
- visual attached:
- next checks due: `+30m`, `+3h`, `17:30 ET`

### Metrics

| Window | Impressions | Engagements | Engagement rate | Link clicks | Profile visits | New follows | Bookmarks | Reposts | Replies | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| publish |  |  |  |  |  |  |  |  |  |  |
| +30m |  |  |  |  |  |  |  |  |  |  |
| +3h |  |  |  |  |  |  |  |  |  |  |
| EOD |  |  |  |  |  |  |  |  |  |  |

### High-Signal Replies

| Handle | Technical question or objection | Public answer given | Qual score 0-6 | Move to DM | Next step |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

### DMs Started

| Handle | Code used | Workflow context | Next action due | Status | Notes |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

### Friction Notes

- 

## Thursday, March 19, 2026 Raw Capture

Motion: proof-to-adoption follow-up pass

### Follow-Ups

| Handle | Source motion | What they asked or signaled | Code used | Follow-up sent | Qual score 0-6 | Next step due | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |

### Metrics

| Window | Follow-ups sent | DMs advanced | Qualified prospects with dated next step | Profile visits influenced | New follows influenced | Strongest repeated objection | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `12:30 ET` |  |  |  |  |  |  |  |
| EOD |  |  |  |  |  |  |  |

### Friction Notes

- 

## Friday, March 20, 2026 Raw Capture

Motion: adoption post
Post code: `w1_adoption_example`

### Publish Facts

- publish time:
- live post URL:
- exact tracked URL used:
- first workflow reply:
- next checks due: `+30m`, `+3h`, `17:30 ET`

### Metrics

| Window | Impressions | Engagements | Engagement rate | Link clicks | Profile visits | New follows | Replies | Bookmarks | DMs started | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| publish |  |  |  |  |  |  |  |  |  |  |
| +30m |  |  |  |  |  |  |  |  |  |  |
| +3h |  |  |  |  |  |  |  |  |  |  |
| EOD |  |  |  |  |  |  |  |  |  |  |

### High-Signal Replies

| Handle | Workflow or failure pattern | Public answer given | Qual score 0-6 | Move to DM | Next step |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

### DMs Started

| Handle | Code used | Setup path | Next action due | Status | Notes |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

### Friction Notes

- 

## Public Workflow Intake

Use this table when a builder opens a GitHub issue or discussion from launch
traffic.

| Date | Handle or name | Source motion | Code used | GitHub URL | Stack | Failure pattern | Next step | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  | `issue_w1_failure_pattern` |  |  |  |  | pending | |

## Reusable Artifact Scratchpad

Capture raw content inputs here before turning them into the `Content Reuse
Queue` in `plans/2026-03-16-cmo-launch-operations-log.md`.

Use this when a post, reply burst, DM, or workflow issue produces something
worth reusing publicly next week.

| Date | Source motion | Artifact type | Raw evidence or handle | Why it matters | Candidate follow-up asset | Moved to launch log | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | metric jump or quote or objection or workflow issue |  |  |  | pending |  |

## End-Of-Day Transfer Checklist

Before closing each launch day:

1. move the live post URL and final same-day metrics into the `Execution Log`
2. move every Tuesday or Thursday distribution session into the
   `Distribution Pass Log`
3. add at least one `Engagement Follow-Up Log` row for each qualified reply
   burst, outbound DM, or inbound DM thread
4. move every public workflow issue or discussion into the `Workflow Intake Log`
5. move the most important checkpoint decision into the `Checkpoint Decision Log`
6. move any bio, website-link, or pinned-post change into the
   `Profile Surface Log`
7. move any launch incident into the `Launch Contingency Log`
8. move the best reusable metric jump, quote, objection, or workflow artifact
   into the `Content Reuse Queue`
9. add every qualified builder to the design-partner pipeline with a score and
   next step plus a real due date
10. copy repeated message, proof, or setup blockers into the friction log
11. roll up the day's profile visits and new follows into the launch log notes
    or scoreboard inputs while the platform snapshots are still easy to verify
12. mark any missing metrics or untracked links explicitly instead of leaving
   blanks
