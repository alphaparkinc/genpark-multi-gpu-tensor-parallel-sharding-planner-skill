from typing import Dict, Any, List, Optional

class MultiGPUTensorParallelShardingPlanner:
    """
    Plans Megatron-style column-parallel and row-parallel matrix decompositions
    for multi-head attention (QKV projection) and feed-forward MLP layers across N GPUs.
    """
    def plan_sharding(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_heads: int,
        world_size: int,
        vram_per_gpu_gb: float = 24.0
    ) -> Dict[str, Any]:
        if num_heads % world_size != 0:
            return {
                "feasible": False,
                "error": f"num_heads ({num_heads}) must be divisible by world_size ({world_size})"
            }

        heads_per_gpu = num_heads // world_size
        head_dim = hidden_dim // num_heads

        # Attention QKV: Column Parallel split along output dimension
        qkv_out_dim_per_gpu = (hidden_dim * 3) // world_size
        qkv_weights_per_gpu = hidden_dim * qkv_out_dim_per_gpu

        # Attention Out Projection: Row Parallel split along input dimension
        out_proj_in_dim_per_gpu = hidden_dim // world_size
        out_proj_weights_per_gpu = out_proj_in_dim_per_gpu * hidden_dim

        # MLP Gate/Up: Column Parallel
        mlp_gate_per_gpu = hidden_dim * (intermediate_dim // world_size)
        # MLP Down: Row Parallel
        mlp_down_per_gpu = (intermediate_dim // world_size) * hidden_dim

        total_layer_weights_per_gpu = qkv_weights_per_gpu + out_proj_weights_per_gpu + (2 * mlp_gate_per_gpu) + mlp_down_per_gpu
        total_layer_mb_fp16 = (total_layer_weights_per_gpu * 2) / (1024 * 1024)

        gpu_allocations = []
        for rank in range(world_size):
            gpu_allocations.append({
                "rank": rank,
                "heads_assigned": heads_per_gpu,
                "qkv_column_slice": f"[0:{hidden_dim}, {rank * (qkv_out_dim_per_gpu)}:{(rank + 1) * qkv_out_dim_per_gpu}]",
                "out_row_slice": f"[{rank * out_proj_in_dim_per_gpu}:{(rank + 1) * out_proj_in_dim_per_gpu}, 0:{hidden_dim}]",
                "weights_per_layer_mb": round(total_layer_mb_fp16, 2)
            })

        return {
            "feasible": True,
            "world_size": world_size,
            "heads_per_gpu": heads_per_gpu,
            "head_dim": head_dim,
            "per_layer_mb_fp16": round(total_layer_mb_fp16, 2),
            "gpu_allocations": gpu_allocations
        }
