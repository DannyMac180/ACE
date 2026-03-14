# ACE Week-One X Profile Surface

Date: 2026-03-13
Owner: CMO
Operator: Dan

## Purpose

Use this file to lock the X profile surface that week-one launch traffic sees
after clicking, tapping through replies, or opening Dan's profile from the
thread.

This file exists because launch posts can be governed while profile conversion
still leaks if the bio, website link, or pinned context are stale or generic.

Canonical launch logging still happens in
`plans/2026-03-16-cmo-launch-operations-log.md`.

## Non-Negotiables

- Do not change the website link mid-week unless the launch log is updated
  first.
- Do not use a generic AI-builder bio when the ACE category line is available.
- Do not pin a different launch post unless the reason is logged the same day.
- Do not change avatar, handle, or other high-recognition profile elements
  during week one unless a real account problem forces it.

## Exact Week-One Profile Link

Use this exact website link from Sunday, March 15, 2026 through Friday,
March 20, 2026 unless the launch log records a change first.

Code: `profile_w1_readme`

`https://github.com/DannyMac180/ACE/blob/main/README.md?utm_source=x&utm_medium=profile&utm_campaign=ace_launch&utm_content=profile_w1_readme`

Why this destination:

- the README is still the best category-conversion surface for profile traffic
- profile clicks often come from people who missed the full thread context and
  need the thesis first
- a dedicated profile code keeps profile traffic separate from thread, reply,
  and DM traffic

## Exact Week-One Bio

Paste this bio for week one unless a tighter version is approved and logged
first:

```text
Building ACE: the adaptive context layer for coding agents. Agentic Context Engineering for prompt drift, brittle prompts, and weak memory.
```

Bio rule:

- if the live account format forces a shorter version, cut only the ending
  phrase after `weak memory`; keep `ACE`, `adaptive context layer`, and
  `Agentic Context Engineering`

## Pinned Post Rule

Use this rule for the pinned post during launch week:

1. keep the existing pin in place until the Monday anchor thread is live
2. after the Monday thread is published and the live URL is verified, pin that
   Monday thread by end of day Monday
3. keep the Monday thread pinned through Friday close unless a different launch
   post materially outperforms it and the change is logged first in
   `plans/2026-03-16-cmo-launch-operations-log.md`

Materially outperforms means:

- clearly stronger qualified reply quality, or
- clearly stronger proof or setup intent than the pinned thread, not just more
  low-signal impressions

## Sunday, March 15, 2026 Preflight

Before launch week starts:

1. update the X profile website field to `profile_w1_readme`
2. update the X bio to the exact week-one bio above
3. confirm the profile surface now matches the README-first week-one launch
   thesis instead of a generic builder message
4. leave the current pinned post alone until the Monday anchor thread is live
5. note any deviation in `plans/2026-03-13-cmo-week-one-live-capture-sheet.md`
   before launch week starts, and record the baseline profile visits plus new
   follows in the `Profile Surface Raw Capture` table there

## Monday, March 16, 2026 Follow-Through

After the anchor thread publishes:

1. verify the live thread URL
2. pin the Monday anchor thread before end of day
3. record the live pinned-thread URL in the `Profile Surface Raw Capture` table
   in `plans/2026-03-13-cmo-week-one-live-capture-sheet.md`
4. record whether profile visits and new follows increased at `+3h` and `EOD`
   in the live capture sheet, then transfer the final snapshot into the
   `Profile Surface Log` in `plans/2026-03-16-cmo-launch-operations-log.md`
5. do not rotate the profile link away from `profile_w1_readme`

## Midweek Rules

- Wednesday proof traffic still lands on the proof asset from the post, but the
  profile link stays on the README so profile visitors keep getting the category
  thesis first
- Friday adoption traffic still lands on the worked example from the post, but
  do not rotate the profile link midweek unless the launch log records why
- if someone asks what ACE is from the profile alone, treat that as validation
  that the profile surface is doing category work; log the observation in the
  friction or notes fields instead of changing the link impulsively
