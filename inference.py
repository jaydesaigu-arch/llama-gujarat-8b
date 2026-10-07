"""Load standalone full BF16 weights, or optionally the original LoRA adapter."""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = 'meta-llama/Llama-3.1-8B-Instruct'
REVISION = '0e9e39f249a16976918f6564b8830bc894c89659'
ADAPTER = 'DesaiJayM/Llama-Gujarat-8B'
MODEL_REVISION = '731b42cd6d78f8388dafaa08953dd1e8c5490007'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt')
    parser.add_argument('--model', default=ADAPTER)
    parser.add_argument('--revision', default=MODEL_REVISION)
    parser.add_argument('--adapter-mode', action='store_true',
                        help='Load pinned Meta base and optional adapter/')
    parser.add_argument('--adapter', default=ADAPTER)
    parser.add_argument('--adapter-revision',
                        help='Legacy adapter-only release commit; implies adapter mode')
    parser.add_argument('--max-new-tokens', type=int, default=192)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error('--max-new-tokens must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('This example requires a compatible CUDA GPU')
    if args.adapter_mode or args.adapter_revision:
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
        model = AutoModelForCausalLM.from_pretrained(
            BASE, revision=REVISION, device_map={'': 0},
            torch_dtype=torch.bfloat16, attn_implementation='sdpa',
        )
        options = {} if args.adapter_revision else {'subfolder': 'adapter'}
        model = PeftModel.from_pretrained(
            model, args.adapter, revision=args.adapter_revision or args.revision,
            **options,
        ).eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, device_map={'': 0},
            torch_dtype=torch.bfloat16, attn_implementation='sdpa',
        ).eval()
    inputs = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': args.prompt}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors='pt',
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    print(tokenizer.decode(
        output[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True,
    ))

if __name__ == '__main__':
    main()
