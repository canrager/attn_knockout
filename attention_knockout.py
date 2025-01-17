import os
import wget
import types
import json
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from attention_utils import AttentionEdge, AttentionPatcher
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

EPS = 1e-10
torch.set_grad_enabled(False)


def init_model(model_id):
    # TODO clarify handlling of BOS token. It is currently added.
    if "meta-llama" in model_id:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            cache_dir="/share/u/can/models",
        )
        model.family_name = "llama"
    elif "google/gemma" in model_id:
        # Adding BOS token is necessary for Gemma2
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            cache_dir="/share/u/can/models",
            low_cpu_mem_usage=True,
            attn_implementation='sdpa', # Does NOT work with `attn_implementation='eager'`
            # use_cache=False, # Do not use attention KV cache
        )
        model.family_name = "gemma"
    else:
        raise ValueError(f"Unsupported model: {model_id}. Only meta-llama and google/gemma models are supported.")
    
    # Save the original forward method for reuse
    model.original_forwards = {}
    for layer in range(model.config.num_hidden_layers):
        model.original_forwards[layer] = model.model.layers[layer].self_attn.forward
    return model


def edit_model(model, layer_window, cut_edges):
    for layer in range(model.config.num_hidden_layers):
        attn_block = model.model.layers[layer].self_attn
        if layer in layer_window:
            # Choose patcher based on model family
            if model.family_name in ["gemma", "llama"]:
                patcher = AttentionPatcher
            else:
                raise ValueError(f"Unsupported model: {model_id}. Only meta-llama and google/gemma models are supported.")

            attn_block.forward = types.MethodType(
                patcher(
                    block_name=f"layers.{layer}.self_attn",
                    cut_attn_edges=cut_edges,
                    save_attn_for=None,
                    attn_matrices=None,
                    attn_contributions=None,
                ),
                attn_block,
            )
        else:
            attn_block.forward = model.original_forwards[layer]
    return model


def find_subject_positions(tokenizer, text, subject):
    """
    Find all positions where the subject appears in the tokenized text.
    Returns both the positions and debug info.
    """
    full_tokens = tokenizer.encode(text, add_special_tokens=True) # Gemma2 does not work for add_special_tokens=False
    subject_tokens = tokenizer.encode(subject, add_special_tokens=False)

    full_token_strings = [tokenizer.decode([t]) for t in full_tokens]
    subject_token_strings = [tokenizer.decode([t]) for t in subject_tokens]

    subject_length = len(subject_token_strings)
    subject_positions = []
    
    for i in range(len(full_token_strings) - subject_length + 1):
        window = full_token_strings[i : i + subject_length]
        if ''.join(window).strip() == ''.join(subject_token_strings).strip():
            subject_positions.append((i, i + subject_length - 1))  # Store start and end positions

    if not subject_positions:
        return None
    
    return subject_positions


def process_subject_position(tokenizer, knowns_df):
    subject_positions = []
    
    for idx, item in enumerate(knowns_df.itertuples()):
        prompt = item.prompt
        subject = item.subject
        positions = find_subject_positions(tokenizer, prompt, subject) # Can be None if not found
        subject_positions.append(positions)
    
    return subject_positions


def get_correct_pred_id(tokenizer, knowns_df):
    correct_pred_ids = []
    for sample in knowns_df.itertuples():
        correct_pred_str = " " + sample.attribute
        correct_pred_id = tokenizer.encode(correct_pred_str, add_special_tokens=False)[0] # First token if multiple tokens
        correct_pred_ids.append(correct_pred_id)
    return correct_pred_ids


def run_exp(model_id, layer_window_size, num_samples=None, verbose=False, cut_all_subject_edges=False):
    # Load model and tokenizer
    model = init_model(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    num_layers = model.config.num_hidden_layers

    # Load dataset
    json_path = "known_1000.json"
    if not os.path.exists(json_path):
        wget.download("https://rome.baulab.info/data/dsets/known_1000.json")
    knowns_df = pd.read_json("known_1000.json")
    if num_samples is None:
        num_samples = len(knowns_df)
    else:
        knowns_df = knowns_df.head(num_samples)

    # Process subject positions
    subject_positions = process_subject_position(tokenizer, knowns_df)
    correct_pred_ids = get_correct_pred_id(tokenizer, knowns_df)

    # Define layer windows
    wing = layer_window_size // 2
    windows = []
    for i in range(num_layers):
        start = max(0, i - wing)
        end = min(i + wing, num_layers - 1)
        window = list(range(start, end + 1))
        windows.append(window)

    # Run the experiment
    base_prob = torch.zeros(num_samples, device=model.device)
    patch_probs = torch.zeros((num_layers, num_samples), device=model.device)

    for sample_idx, sample in tqdm(knowns_df.iterrows(), total=num_samples):
        if subject_positions[sample_idx] is None:
            print(f"Warning: Could not find subject position for sample {sample_idx}. Skipping...")
            continue
       
        correct_pred_id = correct_pred_ids[sample_idx]

        # Tokenize
        input_ids = tokenizer.encode(
            sample["prompt"], return_tensors="pt", add_special_tokens=True
        )
        input_ids = input_ids.to(model.device)

        # Get base prediction
        model = edit_model(model, [], {})
        logits = model(input_ids).logits[0, :, :]
        logits = logits[-1, :]
        probs = torch.softmax(logits, dim=-1)

        base_prob[sample_idx] = probs[correct_pred_id]

        # Define ablation
        q_idx = len(input_ids[0]) - 1
        edges = []
        
        if cut_all_subject_edges:
            # Cut edges from final token to all subject token positions
            for start_pos, end_pos in subject_positions[sample_idx]:
                for k_idx in range(start_pos, end_pos + 1):
                    edges.append(AttentionEdge(q_idx=q_idx, k_idx=k_idx))
        else:
            # Original behavior: only cut edge to final subject position
            k_idx = subject_positions[sample_idx][-1][1]  # Use end position of last occurrence
            edges = [AttentionEdge(q_idx=q_idx, k_idx=k_idx)]

        cut_edges = {i: edges for i in range(model.config.num_attention_heads)}

        # Evaluate all windows
        for l, layer_window in enumerate(windows):
            model = edit_model(model, layer_window, cut_edges)
            logits = model(input_ids).logits[0, -1, :]
                
            probs = torch.softmax(logits, dim=-1)
            patch_probs[l, sample_idx] = probs[correct_pred_id]

    # Calculate relative difference with safety checks
    relative_diff = (patch_probs - base_prob) * 100.0 / (base_prob + EPS)

    # Add logging for debugging
    num_nans = torch.isnan(relative_diff).sum().item()
    if num_nans > 0:
        print(f"Warning: {num_nans} NaN values in results. This can occur if too many edges are cut.")

    # Convert tensors to numpy arrays for JSON serialization
    results = {
        "model_id": model_id,
        "model_name": model_id.split("/")[-1],
        "num_layers": num_layers,
        "num_samples": num_samples,
        "layer_window_size": layer_window_size,
        "relative_diff": relative_diff.cpu().numpy().tolist(),
        "base_prob": base_prob.cpu().numpy().tolist(),
        "patch_probs": patch_probs.cpu().numpy().tolist(),
    }

    return results


def plot_results(results_dict, output_path=None):
    plt.figure(figsize=(9, 5)) 
    
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'legend.title_fontsize': 14,
        'font.family': 'serif',  # Use serif fonts
        'font.serif': 'Times New Roman'  # Set to Times New Roman
    })
    
    sns.set_theme(style="whitegrid")
    
    num_models = len(results_dict)
    palette = sns.color_palette("husl", num_models)  # More vibrant colors
    
    # Create DataFrame for all models
    dfs = []
    model_colors = {}
    for i, (model_name, data) in enumerate(results_dict.items()):
        relative_diff = np.array(data["relative_diff"])
        num_layers = data["num_layers"]
        layer_window_size = data["layer_window_size"]
        num_samples = data["num_samples"]
        
        layer_percentages = np.linspace(0, 100, num_layers)
        model_label = f"{model_name} ({num_layers} layers)\n   {layer_window_size} layer window, {num_samples} samples"
        
        df = pd.DataFrame()
        df["layer_percent"] = np.repeat(layer_percentages, relative_diff.shape[1])
        df["value"] = relative_diff.flatten()
        df["model"] = model_label
        dfs.append(df)
        model_colors[model_label] = palette[i]
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    sns.lineplot(
        data=combined_df,
        x="layer_percent",
        y="value",
        hue="model",
        palette=model_colors,
        errorbar=("ci", 68),
        linewidth=2
    )
    
    plt.xlabel("Layer depth (%)", fontweight='bold')
    plt.ylabel("% change in prediction probability", fontweight='bold')
    plt.title("Attention Knockout Experiment", fontweight='bold', pad=20)
    plt.xlim(0, 100)
    
    plt.grid(True, linewidth=0.8, alpha=0.7)
    plt.axhline(0, color="black", linewidth=1.5)
    
    plt.legend(
        title="Models",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=True,
        facecolor='white',
        edgecolor='black'
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)  # Higher resolution
    else:
        plt.show()


def get_model_name(model_id: str) -> str:
    """Extract a clean model name from model id."""
    return model_id.split('/')[-1].lower()


if __name__ == "__main__":
    # Set experiment parameters here
    model_ids = [
        "meta-llama/Llama-3.2-1B",
        # "google/gemma-2-9b",
        "google/gemma-2-2b",
    ]
    layer_window_sizes = [4]*2
    num_samples = [30]*2
    verbose = True
    cut_all_subject_edges = True  # New parameter to control edge cutting behavior

    assert (
        len(model_ids) == len(layer_window_sizes) == len(num_samples)
    ), "Lengths of model_ids, layer_window_sizes, and num_samples must match"

    # Dictionary to store results for all models
    all_results = {}

    # Run experiments for each model
    start_time = time.time()
    for model_id, layer_window_size, num_sample in zip(model_ids, layer_window_sizes, num_samples):
        print(f"\nProcessing {model_id}...")
        results = run_exp(
            model_id, 
            layer_window_size=layer_window_size, 
            num_samples=num_sample, 
            verbose=verbose,
            cut_all_subject_edges=cut_all_subject_edges
        )
        all_results[results["model_name"]] = results
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f}s")


    # Create clean model names string
    model_names = "_".join([get_model_name(m) for m in model_ids])
    
    # Save results to JSON
    results_filename = f"attn_knockout_{model_names}_win{layer_window_sizes[0]}_n{num_samples[0]}.json"
    
    # Create output directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", results_filename)
    
    if all_results:
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
    else:
        all_results = json.load(open(results_path))

    # Plot results
    plot_filename = results_path.replace(".json", ".png")
    plot_results(all_results, output_path=plot_filename)

    print(f"\nResults have been saved to '{results_path}'")
    print(f"Plot has been saved to '{plot_filename}'")