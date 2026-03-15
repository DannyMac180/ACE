# ACE proof demo asset

This package is the reusable proof asset for launch posts scheduled the
week of Monday, March 16, 2026.

It captures a real ACE loop on a clean local playbook and keeps the
output stable enough to reuse across X posts, README callouts, and docs.

## Files

- [terminal-session.txt](docs/assets/ace-proof-demo/terminal-session.txt)
- [still.svg](docs/assets/ace-proof-demo/still.svg)
- [caption-script.md](docs/assets/ace-proof-demo/caption-script.md)
- [what-you-see.md](docs/assets/ace-proof-demo/what-you-see.md)
- [demo-task.json](docs/assets/ace-proof-demo/demo-task.json)
- [reflection.json](docs/assets/ace-proof-demo/reflection.json)
- [delta.json](docs/assets/ace-proof-demo/delta.json)

## Recording notes

- The terminal transcript is generated from a throwaway workspace.
- The demo forces `ACE_EMBEDDINGS=mock` so the offline path stays deterministic.
- Noisy model-load logs are trimmed from the published transcript for readability.
- The reflection step is deterministic so the asset can be rebuilt offline.
