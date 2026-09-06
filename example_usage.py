import json
from client import MultiGPUTensorParallelShardingPlanner

def main():
    planner = MultiGPUTensorParallelShardingPlanner()
    # Plan Llama-style 70B layer across 4 GPUs (world_size=4)
    res = planner.plan_sharding(
        hidden_dim=8192,
        intermediate_dim=28672,
        num_heads=64,
        world_size=4
    )
    print("Tensor Parallel Plan:")
    print(json.dumps(res, indent=2))
    assert res["feasible"] is True
    assert res["heads_per_gpu"] == 16
    assert len(res["gpu_allocations"]) == 4
    print("Tensor parallel planner verification: PASS")

if __name__ == "__main__":
    main()
