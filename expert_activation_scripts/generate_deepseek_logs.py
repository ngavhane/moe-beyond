import torch
import sys
import re
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "prompts.txt"
    
    checkpoint = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    output_file = "structured_prompts_results.txt"

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
    
    if not prompts:
        print("No valid prompts found in input file.")
        return

    print(f"Processing {len(prompts)} prompts...")
    
    # Process all prompts and write to output file
    with open(output_file, 'w') as out_f:
        for prompt_num, prompt_text in prompts:
            prompt_text = prompt_text.strip()
            out_f.write(f"\n>>>>>>>>>>PROMPT_START: {prompt_num}\n")
            process_prompt(prompt_text, out_f, tokenizer, model)
            out_f.write(f"\n>>>>>>>>>>PROMPT_END: {prompt_num}\n")
            print(f"Processed prompt {prompt_num}")

    print(f"All results written to {output_file}")

def process_prompt(text, output_file, tokenizer, model):
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

        # Write results
        output_file.write(f"\nAnalysis for prompt:\n{text}\n")
        for layer_idx, events in sorted(expert_logs.items()):
            output_file.write(f"\n=== Layer {layer_idx} ===\n")
            for evt in events:
                b, t = evt["batch"], evt["token_pos"]
                ids, ws = evt["expert_ids"], evt["expert_weights"]
                tok = tokenizer.convert_ids_to_tokens(int(inputs.input_ids[b, t]))
                embedding = evt["token_embedding"]
                output_file.write(f"batch={b}, token_pos={t:02d} ({tok!r}) → experts={ids}, weights={ws}, embeddings={embedding}\n")
    finally:
        # Cleanup hooks
        for hook in hooks:
            hook.remove()

if __name__ == "__main__":
    main()
