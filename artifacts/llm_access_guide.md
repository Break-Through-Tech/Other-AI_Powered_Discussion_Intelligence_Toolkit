# LLM Access Guide

Reference for fellows on the **LLM synthesis stage** of the toolkit. Use an LLM only after the connector, schema, retrieval, and evidence-selection steps are working. **Default to local open-weight models** running on Colab's free T4. Use API fallbacks only when you need faster iteration during development or the local model is too slow for the experiment.

## Why local first

- No API keys to manage, no rate limits, no quota anxiety
- Fully reproducible — the model weights are content-addressed on Hugging Face
- The toolkit ships with the model choice baked in; no external dependencies post-program
- Cost: zero

## Model choices

Three viable options for the T4's 16 GB VRAM at int4 quantization. Pick one and stick with it for a given run — switching mid-experiment makes comparisons meaningless.

| Model | Params | VRAM at int4 | Tokens/sec on T4 | Notes |
|---|---|---|---|---|
| **Phi-3-mini-4k-instruct** | 3.8 B | ~3 GB | 35–50 | Fastest, decent quality, 4K context |
| **Qwen2.5-7B-Instruct** | 7.6 B | ~5 GB | 18–25 | Best quality of the three, 32K context |
| **Llama-3.1-8B-Instruct** | 8.0 B | ~5 GB | 15–22 | Strong reasoning, 8K context |

Start with **Qwen2.5-7B-Instruct** — best quality/speed trade-off for the synthesis task. Drop to Phi-3-mini if you need throughput; move to Llama-3.1-8B if you specifically need reasoning chains.

## Loading a model (Colab)

```python
# One-time install
# !pip install -q transformers accelerate bitsandbytes

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
```

Loading takes 60–120 seconds on first call (weights download). Cache them to Drive to skip future waits:

```python
# Cache to mounted Drive
import os
os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
```

## Running inference

```python
def generate(prompt: str, max_new_tokens: int = 512, temperature: float = 0.3) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
```

Wrap this in a small **synthesis** function or class that takes retrieved thread evidence and emits structured JSON or another constrained report format. Do **not** treat the synthesis model as a `TextClassifier`; classification and generation should stay separate in the toolkit design.

## Fallback: free-tier API access

Use an API only when local inference is too slow for the iteration loop or when you need a second comparison model. Store keys in environment variables or Colab secrets. Never commit them to the repo, paste them into notebooks, or assume a shared credential will be available.

### Groq — fastest free tier

- 30 req/min on free tier, 14k tokens/min
- Llama-3.1-8B, Llama-3.3-70B, Qwen-32B all available
- Best for fast iteration during development

```python
from groq import Groq
client = Groq(api_key=GROQ_API_KEY)
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}],
)
```

### Gemini — highest free quota

- 1500 req/day on free tier for Gemini 2.0 Flash
- Long context (1M tokens) — useful for whole-thread synthesis
- Better for batch eval runs than interactive iteration

```python
from google import genai
client = genai.Client(api_key=GEMINI_API_KEY)
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt,
)
```

### OpenRouter — model variety

- Pay-per-use, but several models have free tiers
- Useful when comparing across model families
- Don't make this your default — use one of the above for the bulk of work

## Prompt engineering for discussion analysis

Three things matter more than prompt cleverness:

1. **Structure the input.** A thread is not a wall of text. Format it as:
   ```
   [u/alice, depth 0, score 42]: <root post>
     [u/bob, depth 1, score 10]: <reply>
       [u/alice, depth 2, score 3]: <reply to reply>
   ```
   Models reason much better about threading when the structure is visible.

2. **Anchor outputs in evidence.** Don't ask "summarize this thread." Ask "list the claims being made, and for each claim, cite the utterance ID that contains it." This is how you keep hallucination rate low.

3. **Constrain output format.** JSON or structured prose. For JSON, give the schema in the prompt. Models follow schemas reliably; they invent fields when left to free-form.

```python
SYNTHESIS_PROMPT = """Below is a Reddit discussion thread. Produce a structured analysis.

Output JSON with this schema:
{
  "claims": [
    {"text": "<claim>", "evidence_utterance_ids": ["<id1>", "<id2>"]}
  ],
  "topics": ["<topic1>", "<topic2>"],
  "summary": "<2-3 sentences>"
}

Cite only utterance IDs that actually appear in the thread. If you cannot find evidence for a claim, do not include it.

Thread:
{formatted_thread}
"""
```

## When to escalate

If the local model produces consistently poor output even with structured prompts and evidence anchoring, **the problem is rarely the model**. In order:

1. Check the input structure — is threading visible?
2. Check the prompt — is the schema explicit?
3. Try one prompt change at a time, measure on the held-out set
4. Only then consider switching models

Switching models is the easiest debugging step to *reach for* and the rarely the one that actually helps.
