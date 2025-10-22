```mermaid
flowchart LR
  %% STYLE
  classDef ext fill:#f7f7ff,stroke:#6b7fd7,stroke-width:1px
  classDef svc fill:#f1fbf5,stroke:#1e7f4f,stroke-width:1px
  classDef lib fill:#fff8f0,stroke:#a36500,stroke-width:1px
  classDef data fill:#fefbea,stroke:#b59f3b,stroke-width:1px
  classDef infra fill:#eef6ff,stroke:#2b6cb0,stroke-width:1px
  classDef sec fill:#fff,stroke:#e53e3e,stroke-dasharray:3

  subgraph UserSpace["Developer & Agent Host"]
    Dev["Dev (Dan)"]:::ext
    Host["MCP Host / Agent Runtime\n(e.g., Claude Desktop / Agents SDK / Cursor)"]:::infra
    Amp["Agent: Amp\n(planner/executor)"]:::ext
    Dev -->|tasks| Amp
    Amp --- Host
  end

  subgraph ACEServer[ACE MCP Server]
    MCP["ace_mcp_server\n(tools & resources)"]:::svc
  end

  subgraph ACELib[ACE Core Library]
    GEN["Generator\n(ReAct loop optional,\ntrajectory logger)"]:::lib
    REF["Reflector\n(structured critique → lessons JSON)"]:::lib
    CUR["Curator\n(delta ops: ADD/PATCH/\nDEPRECATE/INCR)"]:::lib
    MERGE["Delta Merger\n(deterministic, non‑LLM)"]:::lib
    RET["Retrieval\n(BM25 + Embeddings + Rerank)"]:::lib
    REFINE["Grow & Refine\n(dedup/merge/archive)"]:::lib
  end

  subgraph DataPlane["Storage & Models"]
    PB[("Playbook Store\nSQLite/Postgres")]:::data
    VEC[("Vector Index\nFAISS/pgvector")]:::data
    EMB["Embedding Model\n(local or API)"]:::infra
    LLM["LLM(s) for GEN/REF/CUR\n(Open/Hosted)"]:::infra
    LOG[("Logs/Events/Traces\n(JSONL / OTLP)")]:::data
  end

  subgraph Tooling[Dev Tooling]
    GIT[("Git/Repo")]:::infra
    CI[("CI/CD: tests, lint, typecheck")]:::infra
    SVCs[("External APIs/Tools\n(HTTP, DB, Filesystem)")]:::infra
  end

  %% WIRES
  Host -- MCP tools --> MCP
  MCP -- calls --> GEN
  MCP -- calls --> RET & REF & CUR & MERGE & REFINE
  GEN -- retrieve bullets --> RET
  RET -- topK IDs --> GEN
  GEN -- LLM calls --> LLM
  REF -- LLM calls --> LLM
  CUR -- "LLM optional (draft)" --> LLM
  RET -.embed.-> EMB
  RET -- index --> VEC
  GEN -- trajectories/env feedback --> REF
  REF -- lessons --> CUR
  CUR -- delta --> MERGE
  MERGE -- apply --> PB
  RET -- read bullets --> PB
  REFINE -- dedup/merge/archive --> PB
  AMP_TOOLS{{Amp Tools}}:::infra
  Amp -- read/write code --> GIT
  Amp -- run tests/build --> CI
  Amp -- call tools --> SVCs
  GEN --- AMP_TOOLS
  GEN -- exec feedback --> LOG
  MCP -- server logs --> LOG
  ACELib -- metrics --> LOG
  ```
