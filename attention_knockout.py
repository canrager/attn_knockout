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
from llama_attention import LlamaAttentionPatcher, AttentionEdge
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

torch.set_grad_enabled(False)


def init_model(model_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        cache_dir="/share/u/can/models",
    )
    # Save the original forward method for reuse
    model.original_forwards = {}
    for layer in range(model.config.num_hidden_layers):
        model.original_forwards[layer] = model.model.layers[layer].self_attn.forward
    return model


def edit_model(model, layer_window, cut_edges):
    for layer in range(model.config.num_hidden_layers):
        attn_block = model.model.layers[layer].self_attn
        if layer in layer_window:
            attn_block.forward = types.MethodType(
                LlamaAttentionPatcher(
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


def find_subject_end_position(tokenizer, text, subject):
    """
    Find the position of the last token of the subject in the tokenized text.
    """
    full_tokens = tokenizer.encode(text, add_special_tokens=False)
    subject_tokens = tokenizer.encode(subject, add_special_tokens=False)

    full_token_strings = [tokenizer.decode([t]) for t in full_tokens]
    subject_token_strings = [tokenizer.decode([t]) for t in subject_tokens]

    subject_length = len(subject_token_strings)
    for i in range(len(full_token_strings) - subject_length + 1):
        if full_token_strings[i : i + subject_length] == subject_token_strings:
            return i + subject_length - 1

    return None


def process_subject_position(model_id, knowns_df):

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    subject_end_pos = []
    for item in knowns_df.itertuples():
        prompt = item.prompt
        subject = item.subject
        subject_end_pos.append(find_subject_end_position(tokenizer, prompt, subject))

    return subject_end_pos


def run_llama(model_id, layer_window_size, num_samples=None):
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
    subject_end_pos = process_subject_position(model_id, knowns_df)

    # Define layer windows
    wing = layer_window_size // 2
    windows = []
    for i in range(num_layers):
        start = max(0, i - wing)
        end = min(i + wing, num_layers - 1)
        window = list(range(start, end + 1))
        windows.append(window)
    print(f"Selected layer windows:\n{windows}")

    # Run the experiment
    base_prob = torch.zeros(num_samples, device=model.device)
    patch_probs = torch.zeros((num_layers, num_samples), device=model.device)

    for sample_idx, sample in tqdm(knowns_df.iterrows(), total=num_samples):
        # Tokenize
        input_ids = tokenizer.encode(
            sample["prompt"], return_tensors="pt", add_special_tokens=False
        )
        input_ids = input_ids.to(model.device)

        # Get base prediction
        model = edit_model(model, [], {})
        logits = model(input_ids).logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs)
        base_prob[sample_idx] = probs[pred_id]

        # Define ablation
        k_idx = subject_end_pos[sample_idx]
        q_idx = len(input_ids[0]) - 1
        edges = [AttentionEdge(q_idx=q_idx, k_idx=k_idx)]
        cut_edges = {i: edges for i in range(model.config.num_attention_heads)}

        # Evaluate all windows
        for l, layer_window in enumerate(windows):
            model = edit_model(model, layer_window, cut_edges)
            logits = model(input_ids).logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            patch_probs[l, sample_idx] = probs[pred_id]

    relative_diff = (patch_probs - base_prob) * 100.0 / base_prob

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



if __name__ == "__main__":
    # Set experiment parameters here
    model_ids = [
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.1-8B",
    ]
    layer_window_sizes = [8, 12, 14]
    num_samples = [20]*3

    assert (
        len(model_ids) == len(layer_window_sizes) == len(num_samples)
    ), "Lengths of model_ids, layer_window_sizes, and num_samples must match"

    # Dictionary to store results for all models
    all_results = {}

    # Run experiments for each model
    start_time = time.time()
    for model_id, layer_window_size, num_sample in zip(model_ids, layer_window_sizes, num_samples):
        print(f"\nProcessing {model_id}...")
        results = run_llama(model_id, layer_window_size=layer_window_size, num_samples=num_sample)
        all_results[results["model_name"]] = results
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f}s")

    # Save results to JSON
    if all_results:
        with open("model_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
    else:
        all_results = json.load(open("model_results.json"))


    # Plot results
    plot_results(all_results, output_path="attention_knockout_comparison.png")

    print("\nResults have been saved to 'model_results.json'")
    print("Plot has been saved to 'attention_knockout_comparison.png'")
