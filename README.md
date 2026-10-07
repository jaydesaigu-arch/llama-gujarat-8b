# Llama Gujarat 8B

**Bringing Gujarati into the AI conversation.** Created by **Dr. Jay Desai**. Built with Llama, using Meta Llama 3.1 8B Instruct.

A standalone Gujarati language research model for instruction following, translation and supplied-context questions. Full BF16 weights and the original LoRA adapter are both available. Explore the model, inspect its answer examples and help improve Gujarati AI.

Model: https://huggingface.co/DesaiJayM/Llama-Gujarat-8B
Code: https://github.com/jaydesaigu-arch/llama-gujarat-8b
Project: https://www.leoai.in/models/llama-gujarat-8b/

## Gujarati language results worth exploring

**16/16 structured translations preserved the meaning and every requested key detail. All 8 supplied-context answers returned both requested habitat facts.** In this 48-question development review, the model carried source, object, action, recipient and numerical details across Gujarati–English translations and extracted information from fictional supplied passages. One habitat answer also repeated an extra sentence from its source, giving 7/8 strict habitat-only answers.

These encouraging results make this release a practical starting point for exploring Gujarati translation, instruction following and context-grounded answers. They describe these authored exercises; broader language, grammar and factual performance remains for future evaluation.

## Measured results and scope

| Final development task | Correct / tested |
| --- | ---: |
| Structured translation meaning/slots |16/16|
| Supplied habitat fields |8/8 core;7/8 requested-only|
| Tiny English retention |6/6 normalized|

Final training loss:0.0178173711. Final validation loss (122-example development fixture):0.2046867821; matched base:0.4505630165. Separate 48-example development loss:0.0742802848; matched base:0.7192907749. Native BF16 completion-token-weighted causal NLL including EOT, same base/collator, batch4; measured in a separate zero-optimizer A10080GB evaluation of unchanged final weights.

Final adapter evaluated on 48 authored development prompts:12 Gujarati arithmetic,12 English arithmetic,16 translations and8 fictional supplied-context habitat questions, with a separate6-example English retention fixture. Full fresh and historical suites were not generated after the original development stop. This release is authorized with disclosed arithmetic limitations, not a claim of full qualification.

The original test outcomes remain recorded, including failed screens. This release proceeds with disclosed arithmetic limitations. These small authored, template-sharing and inspected tests are not independent headline benchmarks. No broad factual, mathematical or bilingual accuracy claim follows from them. Loss measures reference fit.

## Training

Seven stages: clean:1epoch at2e-5; refinement:2epochs at5e-5; bridge:1epoch at3e-5; terminology:1epoch at1e-5; coverage:1epoch at2e-5; transfer:1epoch at1e-5; Gujarati arithmetic:1epoch at1e-5. The first five used NF4 QLoRA; the last two continued the saved adapter with native BF16 and fresh optimizers. The final 12,394-row stage added Gujarati carry/borrow/column multiplication/percentage explanations and numeral conversion, with English and non-math replay. Final 775 optimizer steps/full epoch. Counts include replay and are not unique-example totals. Groundnut is correctly both a legume and an oilseed crop.

The code repository records the recipe. Private input bundles, credentials, runtime/account logs and earlier unsuccessful adapters are excluded from this public package.

## Download and use full weights

The root of this repository contains the complete standalone BF16 model: four weight shards, model configuration and tokenizer. Download size is about **16.1 GB**. No separate base-model download or PEFT adapter loader is needed for this version. The original adapter remains in `adapter/`; its previous release is also preserved at commit `ed38be6643a3f7d5f94cee9b12c09aa54cddee51`.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "DesaiJayM/Llama-Gujarat-8B"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(
    repo, torch_dtype=torch.bfloat16, device_map="auto",
    attn_implementation="sdpa",
).eval()
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "ગુજરાતીમાં ટૂંકો જવાબ આપો."}],
    add_generation_prompt=True, return_dict=True, return_tensors="pt",
).to(model.device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=192, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

For reproducible downloads, pin the verified release commit recorded in the code repository's `release.json`. Use the pinned requirements from that repository and a compatible BF16 GPU. A CPU/offload deployment needs sufficient memory and may be slow. Quantized inference can change results.

### Conversion provenance

These full weights merge the exact published LoRA adapter into Meta Llama 3.1 8B Instruct with official PEFT 0.17.1 `merge_and_unload(safe_merge=True)`. This is a **merged LoRA checkpoint**, not a new full-parameter training run. No optimizer steps were taken during conversion. Base revision: `0e9e39f249a16976918f6564b8830bc894c89659`. Original adapter SHA256: `150c261d6b05e2f40efe01b131b29283ed527278d5cdcdb56f19f6a3e90081e2`.

All 291 expected model tensors are present, all 224 LoRA linear layers were merged, and saved tensors were checked for finite values. The saved standalone model was reloaded and three translation/context probes were compared with the separate base-plus-adapter configuration. See `merge-provenance.json`, `merge-comparison.json` and `SHA256SUMS`. BF16 merging involves rounding; the earlier 48-answer scores and validation losses below describe the separate base-plus-adapter configuration, rather than a new full evaluation of the merged checkpoint.

To use only the compact adapter, load the pinned Meta base and `PeftModel.from_pretrained(base, repo, subfolder="adapter")`; base access and a separate base download are required for that option. Follow Meta's Llama 3.1 licence and acceptable use policy for both downloads.

## Arithmetic in context

Like its Meta Llama 3.1 8B Instruct base, Llama Gujarat 8B can make numerical mistakes. [Meta reports](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct#instruction-tuned-models)84.5% on GSM8K with 8-shot chain-of-thought prompting and 51.9% on MATH with 0-shot chain-of-thought prompting. These are different English benchmarks; they do not establish the same error rate for this adapter or attribute every observed error to the base model.

This release focuses on Gujarati language adaptation. Its available examples show faithful structured translations and supplied-context fact extraction. For exact calculations, validate the quantities and operation, execute a calculator or code, and check the result. A calculator/agent integration is not included or tested.

## Evaluation scope and use

- Arithmetic on 48 authored development prompts: Gujarati8/12 and English8/12. Eight numerical answers were incorrect; the original screen required11/12 in each language and its outcome remains recorded.
- All16 narrowly structured development translations preserved meaning and slots. This does not establish broad translation quality.
- All 8 fictional supplied-context habitat answers preserved both requested fields; one repeated an extra supplied sentence, so strict habitat-only compliance was7/8.
- No answer hit the192-token development cap. A tiny six-example English retention fixture passed after normalizing capitalization and terminal punctuation; it is not a broad English benchmark.
- The frozen100 fresh generations,657 historical regression generations,100 paired base generations were not run because the development screen stopped the original pipeline.
- These authored prompts share task families with training. Current weights have not been requalified on the full historical suite. Prior-candidate metrics are not results for this adapter.
- A calculator or code-execution agent is not included or tested. Validate the operation and quantities and check executed results for exact calculations.
- A separate zero-optimizer A10080GB evaluation measured reference-token loss on 122 development examples and the 48 observed development examples. Training and generation used H200; this is not a controlled cross-hardware accuracy comparison or an independent benchmark.

For exact arithmetic, an application can validate the quantities and operation, execute a calculator or generated code, and check the result before presenting it. This package does not include or claim a tested calculator/agent integration. The model can misunderstand a problem or generate incorrect code. For factual applications, supply authoritative context and verify the answer.

## Licence

The merged weights, adapter and tokenizer follow the Llama 3.1 Community License and Acceptable Use Policy. Retain LICENSE/NOTICE/USE_POLICY.md. Original project code uses MIT; Meta materials are in model-license/ in the code repository. No institutional or source-publisher endorsement is implied.

Pinned full-model revision: `731b42cd6d78f8388dafaa08953dd1e8c5490007`. Run `python inference.py 'ગુજરાતીમાં ટૂંકો જવાબ આપો.'` for the standalone model. Use `--adapter-mode` for the compact adapter or `--adapter-revision ed38be6643a3f7d5f94cee9b12c09aa54cddee51` for the original release.
