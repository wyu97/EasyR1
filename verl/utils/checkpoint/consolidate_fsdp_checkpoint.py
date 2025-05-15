import os
import torch
#from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
#from torch.distributed.fsdp import StateDictType, ShardedStateDictConfig
from transformers import Qwen2_5_VLForConditionalGeneration, AutoConfig
import torch.distributed as dist
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"

def consolidate_fsdp_checkpoint(checkpoint_dir: str, output_path: str, world_size: int):
    """
    Consolidate sharded FSDP checkpoints into a single model file.
    
    Args:
        checkpoint_dir (str): Directory containing the sharded checkpoints.
        output_path (str): Path to save the consolidated model file.
        world_size (int): Number of ranks (shards) used during training.
    """
    dist.init_process_group(backend="gloo", init_method="env://", world_size=1, rank=0)
    # Load the Hugging Face config (saved by rank 0)
    hf_config_path = os.path.join(checkpoint_dir, "huggingface")
    config = AutoConfig.from_pretrained(hf_config_path)

    # Initialize the model (unwrapped)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained('/apdcephfs_sh2_300000800/share_300000800/models/Qwen2.5-VL-7B-Instruct')

    # Load all sharded state dictionaries
    sharded_state_dicts = []
    for rank in range(world_size):
        model_path = os.path.join(checkpoint_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing shard: {model_path}")
        shard = torch.load(model_path, map_location="cpu", weights_only=False)
        sharded_state_dicts.append(shard)
        print(f"Loaded shard from rank {rank}")


    # Group shards by parameter key
    param_shards = {}
    for rank, shard in enumerate(sharded_state_dicts):
        for key, value in shard.items():
            if key not in param_shards:
                param_shards[key] = [None] * world_size  # Initialize list for all ranks
            # Extract tensor from DTensor if necessary
            
            tensor_value = value.to_local()
            param_shards[key][rank] = tensor_value

    # Manually merge the sharded state dictionaries
    full_state_dict = model.state_dict()  # Get the structure
    for key in full_state_dict.keys():
        if key in param_shards:
            shards = param_shards[key]
            # Ensure all shards are present
            if any(shard is None for shard in shards):
                print(f"Warning: Incomplete shards for {key}, skipping")
                continue
            # Concatenate shards along the sharded dimension (assume dim 0 for now)
            #try:
            full_tensor = torch.cat(shards, dim=0)
            if full_tensor.shape == full_state_dict[key].shape:
                full_state_dict[key].copy_(full_tensor)
            else:
                print(f"Warning: Reconstructed shape {full_tensor.shape} for {key} does not match expected {full_state_dict[key].shape}")
                exit()
            # except RuntimeError as e:
            #     print(f"Error reconstructing {key}: {e}")
        else:
            print(f"Warning: Key {key} not found in any shard")

    # Save the consolidated state dictionary
    #torch.save(full_state_dict, output_path)
    #print(f"Consolidated model saved to {output_path}")

    # Optionally, save in Hugging Face format
    hf_output_dir = os.path.splitext(output_path)[0] + "_hf"
    os.makedirs(hf_output_dir, exist_ok=True)
    model.load_state_dict(full_state_dict)  # Load into the model for HF save
    model.save_pretrained(hf_output_dir)
    # unwrapped_model = model._fsdp_wrapped_module
    # unwrapped_model.load_state_dict(full_state_dict)
    # unwrapped_model.save_pretrained(hf_output_dir)
    print(f"Hugging Face-compatible model saved to {hf_output_dir}")

if __name__ == "__main__":
    checkpoint_dir = "/apdcephfs_gy2/share_302625455/user/kaixinma/gui_output/qwenvl_verl_test_hd_15steps/global_step_137/actor/"  # Replace with your checkpoint directory
    output_path = "/apdcephfs_gy2/share_302625455/user/kaixinma/gui_output/qwenvl_verl_test_hd_15steps/global_step_137/actor/huggingface"        # Output file name
    world_size = 8                               # Replace with your training world_size

    consolidate_fsdp_checkpoint(checkpoint_dir, output_path, world_size)