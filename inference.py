"""Run the released adapter with its pinned Llama 3.1 base on a CUDA GPU."""
import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = 'meta-llama/Llama-3.1-8B-Instruct'
REVISION = '0e9e39f249a16976918f6564b8830bc894c89659'
ADAPTER = 'DesaiJayM/Llama-Gujarat-8B'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt')
    parser.add_argument('--adapter', default=ADAPTER)
    parser.add_argument('--adapter-revision', required=True,
                        help='Verified release commit from the model card')
    parser.add_argument('--max-new-tokens', type=int, default=384)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error('--max-new-tokens must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('This example requires a compatible CUDA GPU')
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, revision=REVISION, device_map={'': 0},
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
    )
    model = PeftModel.from_pretrained(
        model, args.adapter, revision=args.adapter_revision,
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
