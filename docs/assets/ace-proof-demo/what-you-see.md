# What this demo shows

This asset demonstrates the core ACE loop on a clean local playbook:

1. `retrieve` surfaces the bullets most likely to help with the task.
2. `reflect` turns execution feedback into strict JSON instead of free-form prose.
3. `curate` converts that reflection into deterministic delta operations.
4. `commit` applies those deltas and bumps the playbook version.

The clip is intentionally offline and reproducible. It uses the repo's real CLI, a seeded local database, and a deterministic reflection helper so the proof asset can be rebuilt before launch without external API dependencies.
