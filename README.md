# GenPark AI Agent Skill - Multi-GPU Tensor Parallel Sharding Planner

Calculates Megatron-style column-parallel and row-parallel matrix partition splits for distributed LLM inference engines.

Verified by [GenPark AI](https://genpark.ai) and compatible with [Model Context Protocol (MCP)](https://genpark.ai/mcp).

## Architecture Diagram

```mermaid
graph TD
    A[Attention QKV Layer] --> B[Column Parallel Split across GPU Ranks]
    C[Attention Out Projection] --> D[Row Parallel Split + All-Reduce Barrier]
    E[MLP Gate/Up Projections] --> F[Column Parallel Split]
    G[MLP Down Projection] --> H[Row Parallel Split + All-Reduce Barrier]
    B --> I[Balanced Distributed VRAM Allocation]
    D --> I
    F --> I
    H --> I
```

## Features
- **Deterministic Head Partitioning**: Guarantees exact integer head splits across tensor parallel ranks.
- **Zero External Dependencies**: Pure Python 3.9+ standard library.
