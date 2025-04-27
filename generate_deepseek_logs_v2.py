import os
import torch
import sys
import re
import csv
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

def main():
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "prompts.txt"
   
    skip_until_prompt = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    checkpoint = "deepseek-ai/DeepSeek-V2-Lite-Chat"

    # Load model
    print(f"Loading model {checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        load_in_8bit=True,
        device_map="auto"
    ).eval()
    print("Model loaded.")

    # Read and parse input file
    try:
        with open(input_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    # Extract prompts with pattern: PROMPT_START: <num> ... PROMPT_END: <num>
    prompt_pattern = re.compile(
        r'>>>>>>>>>>>PROMPT_START: (\d+)\n(.*?)\n>>>>>>>>>>>>PROMPT_END: \1',
        re.DOTALL
    )
    prompts = prompt_pattern.findall(content)
    
    prompts = [
        (num, text) for num, text in prompts
        if int(num) >= skip_until_prompt
    ]

    if not prompts:
        print("No valid prompts found in input file.")
        return

    print(f"Processing {len(prompts)} prompts...")
    
    # Process all prompts

    output_dir = os.path.expanduser("~/ngavhane-fs/dataset_csvs")
    os.makedirs(output_dir, exist_ok=True)

    for 100, prompt_text in prompts:
        prompt_text = prompt_text.strip()
        output_file = os.path.join(output_dir, f"prompt_{prompt_num}_data.csv")
        print(f"Processing prompt {prompt_num} and saving to {output_file}")

        # Process and write results to CSV file for each prompt
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write header without prompt number and prompt text
            csv_writer.writerow([
                "Layer ID", "Batch Number", "Token", "Activated Expert IDs", "Token Embedding Vector"
            ])
            
            process_prompt(prompt_text, csv_writer, tokenizer, model)
            gc.collect()
            torch.cuda.empty_cache()

    print(f"All prompts processed.")

def process_prompt(text, csv_writer, tokenizer, model):
    expert_logs = defaultdict(list)

    # Hook factory
    def make_hook(layer_idx):
        def _hook(module, inputs, output):
            topk_idx, topk_weight, _ = output
            hidden_states = inputs[0]
            B, T, _ = hidden_states.shape
            K = topk_idx.size(-1)

            ids = topk_idx.view(B, T, K).detach().cpu()
            weights = topk_weight.view(B, T, K).detach().cpu()

            for b in range(B):
                for t in range(T):
                    expert_logs[layer_idx].append({
                        "batch": b,
                        "token_pos": t,
                        "expert_ids": ids[b, t].tolist(),
                        "expert_weights": weights[b, t].tolist(),
                        "token_embedding": hidden_states[b, t].detach().cpu().tolist(),
                    })
        return _hook

    # Register hooks
    hooks = []
    for idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if hasattr(mlp, "gate"):
            hook = mlp.gate.register_forward_hook(make_hook(idx))
            hooks.append(hook)

    try:
        # Process prompt
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)

        # Write results for this prompt to CSV
        for layer_idx, events in sorted(expert_logs.items()):
            for evt in events:
                b, t = evt["batch"], evt["token_pos"]
                ids = evt["expert_ids"]
                tok = tokenizer.convert_ids_to_tokens(int(inputs.input_ids[b, t]))
                embedding = evt["token_embedding"]

                # Write row to CSV without prompt number or prompt text
                csv_writer.writerow([
                    layer_idx, b, tok, ids, embedding
                ])

    finally:
        # Cleanup hooks
        for hook in hooks:
            hook.remove()

if __name__ == "__main__":
    main()
