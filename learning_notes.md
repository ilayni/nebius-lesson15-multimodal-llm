# Learning Points — Multimodal Models (HW4)

## What This Exercise Was About

Building a multimodal pipeline that combines vision and language: generating image captions, answering visual questions, and doing identity recognition — all using pretrained models from HuggingFace.

---

## Models Used

| Model | Purpose | Source |
|---|---|---|
| `Salesforce/blip-image-captioning-base` | Image captioning (Task 1) | HuggingFace |
| `Salesforce/blip-vqa-base` | Visual Question Answering (Task 2) | HuggingFace |
| `openai/clip-vit-base-patch32` | Image embeddings for face recognition (Task 3) | HuggingFace |

**BLIP** (Bootstrapped Language-Image Pretraining) — encoder-decoder model that jointly learns vision and language. Separate fine-tuned variants exist for captioning vs. VQA.

**CLIP** (Contrastive Language-Image Pretraining) — OpenAI model trained to align images and text in a shared embedding space using contrastive loss. Excellent for zero-shot retrieval and similarity-based matching.

---

## Python Libraries Used

| Library | Role |
|---|---|
| `transformers` | Load and run BLIP & CLIP models |
| `torch` | Tensor ops, GPU inference |
| `Pillow (PIL)` | Image loading and conversion |
| `requests` | Download images from URLs |
| `numpy` | Array manipulation, dataset splits |
| `matplotlib` | Visualising images, gallery figures, confusion matrices |
| `pandas` | Summary tables, CSV export |
| `scikit-learn` | `fetch_lfw_people` — loads the LFW face dataset |
| `google-api-python-client` | Google Drive upload (Colab-specific) |

---

## Datasets & Data Sources

| Dataset | Used For | URL |
|---|---|---|
| COCO val2017 | Images for captioning & VQA (Tasks 1 & 2) | http://images.cocodataset.org/val2017/ |
| LFW (Labeled Faces in the Wild) | Celebrity face recognition (Task 3) | https://scikit-learn.org/stable/datasets/real_world.html#labeled-faces-in-the-wild-dataset |

LFW was pulled directly via `sklearn.datasets.fetch_lfw_people`. No manual download needed.

---

## Key Concepts Learned

### 1. Image Captioning
- A pretrained vision-language model reads an image and generates a free-text description.
- BLIP uses a visual encoder (ViT) + a language decoder (BERT-based) trained end-to-end.
- Generation is autoregressive: tokens are sampled one at a time conditioned on the image.

### 2. Visual Question Answering (VQA)
- Same architecture as captioning, but the question is passed as a text prompt alongside the image.
- The model produces a short free-form answer.
- VQA and captioning use different fine-tuned heads (the output later of the model) even when the backbone is the same.

### 3. CLIP Embeddings for Identity Recognition
- CLIP encodes images into a 512-dimensional vector. Similar images end up close in this space.
- Identity recognition = cosine similarity between a query embedding and per-identity centroid embeddings.
- Centroid = mean of multiple reference embeddings per person (smooths out pose/lighting variation).
- This is a **zero-shot** retrieval approach — no fine-tuning needed.

### 4. Two-Model Pipeline (CLIP + BLIP)
- CLIP answers "who is this?" via similarity search.
- BLIP-VQA answers "what are they wearing / doing?" as a separate inference.
- The two models share no state — BLIP doesn't know the identity CLIP predicted.
- A more integrated system (e.g., a single multimodal LLM) would do better at attribute questions about a named person.

---

## Concept Deep-Dives

### What is zero-shot retrieval?
"Zero-shot" means the model was **never trained or fine-tuned on the task you're asking it to do** — no labeled examples, no gradient updates. It works purely from the representations learned during pretraining.

In the context of CLIP identity recognition:
- We never showed CLIP any labeled face-recognition examples.
- We just embedded reference images for each person, then embedded a query image, and asked "which reference is most similar?"
- CLIP's pretraining already gave it a rich enough embedding space that this works — without any task-specific training.

Contrast with a "few-shot" approach (give the model a handful of examples) or "fine-tuned" (retrain the model on your specific task data).

---

### What is contrastive loss?
Contrastive loss is the training objective that teaches CLIP to **pull matching pairs together and push non-matching pairs apart** in the shared embedding space.

During training, CLIP receives batches of (image, text) pairs. For each batch of N pairs:
- The diagonal entries are **positives** (the image and its matching caption).
- All off-diagonal entries are **negatives** (image paired with a wrong caption, or caption paired with a wrong image).

The loss maximizes the cosine similarity of matching pairs while minimizing it for all mismatched pairs — simultaneously in both directions (image→text and text→image). The result is an embedding space where "a photo of a dog" lands near images of dogs, and far from images of cats.

In plain terms: *the model learns by being told "these two things go together, those two things don't."*

---

### What does the centroid represent — image or text?
In this exercise, the centroid is a **pure image embedding** — it represents images only, no text.

For each identity (person) in the reference set, we took several photos of that person, ran each through CLIP's image encoder, and averaged the resulting vectors. That average is the centroid.

The centroid is a single point in CLIP's embedding space that approximates "the typical appearance of this person." At query time, a new photo is embedded and compared (cosine similarity) to each identity's centroid. The nearest centroid wins.

Text plays no role in the recognition step — CLIP's shared embedding space just makes it possible to also compare images against text if needed, but here we're doing image-to-image only.

---

### Is CLIP focused on identity of people? On object recognition?
Neither specifically. **CLIP is a general-purpose vision-language alignment model**, not specialized for faces or objects.

- It was trained on ~400 million (image, alt-text) pairs scraped from the web — covering an enormous variety of scenes, objects, styles, and concepts.
- It has no special face-recognition head and no object-detection head.
- It doesn't inherently know who a person *is* by name — it just embeds visual appearance into a vector.

Using CLIP for face recognition (as in Task 3) is repurposing a general model: we exploit the fact that photos of the same person tend to embed near each other, but CLIP itself doesn't "know" it's doing face recognition.

Object recognition is something CLIP handles well via text prompts (e.g. cosine similarity against "a photo of a cat" vs. "a photo of a dog"), but again this is emergent from general training, not a targeted capability.

---

### What is CLIP good at — in simple words?
CLIP is good at **matching things by meaning across image and text**.

Give it an image and a list of text descriptions — it can tell you which description fits best, with no task-specific training. Give it a collection of images — it can find the one most similar to a query image or a query phrase.

Simple examples of what CLIP can do:
- "Which of these 10 images best matches the caption 'a sunset over the ocean'?" → picks the right one.
- "Is this photo more like a cat or a dog?" → correct most of the time.
- "Find the image in this database that looks most like this query photo." → nearest-neighbor retrieval.

What it is **not** good at: reading fine text in images, precise counting, identifying specific individuals by name (it has no identity database), or detecting exact bounding boxes.

---

### Full pipeline flow — CLIP + BLIP together

**Important upfront:** BLIP does NOT receive CLIP's output. They run independently on the same image. The pipeline is two parallel branches, not a chain.

```
                        ┌─────────────────────┐
                        │   Input: test photo  │
                        │   (e.g. a celebrity) │
                        └────────┬────────────┘
                                 │
               ┌─────────────────┴──────────────────┐
               │                                    │
               ▼                                    ▼
        ┌─────────────┐                    ┌──────────────────┐
        │    CLIP      │                   │   BLIP-VQA       │
        │ image encoder│                   │                  │
        └──────┬───────┘                   │ Input: image +   │
               │                           │ text question    │
               │ 512-dim                   │ e.g. "What is    │
               │ embedding                 │ this person       │
               │ vector                    │ wearing?"        │
               ▼                           └───────┬──────────┘
  Compare cosine similarity                        │
  against reference centroids                      │ text answer
  (one centroid per identity,                      │ e.g. "a suit"
   each centroid = average of                      ▼
   reference photo embeddings)           ┌──────────────────┐
               │                         │  Attribute output │
               │ nearest centroid wins   │  (free-form text) │
               ▼                         └──────────────────┘
        ┌──────────────────────────────────────┐
        │  Our code looks up the label         │
        │  attached to the winning centroid    │
        │  in our reference dictionary         │
        │  e.g. centroid #3 → "Colin Powell"  │
        │                                      │
        │  ⚠ CLIP never output this text.      │
        │  The name came from LFW metadata     │
        │  that we loaded before running CLIP. │
        └──────────────────────────────────────┘
```

**What each model receives and produces:**

| | CLIP | BLIP-VQA |
|---|---|---|
| Input | test photo (pixels) | same test photo + a text question |
| Output | 512-dim float vector | free-text answer |
| How output is used | compared to reference centroids → identity label | shown directly as the answer |

**How CLIP produces a name — step by step (CLIP never outputs text):**

1. **Before running CLIP**, our code loaded the LFW dataset which includes name labels (e.g. "Colin Powell"). We ran CLIP's image encoder on several reference photos per person and averaged the resulting vectors into one centroid per person. Each centroid was stored in a dictionary: `{ "Colin Powell": [0.12, -0.34, ...], "George Bush": [...], ... }`. The names are just Python strings from the dataset — CLIP has no idea what they say.

2. **At query time**, we fed the test photo into CLIP's image encoder. CLIP output a 512-dim float vector — nothing else. No text, no name, just numbers like `[0.07, 0.21, -0.11, ...]`.

3. **Our code** computed cosine similarity between that vector and every centroid in the dictionary, found the highest score (say, centroid for "Colin Powell"), and printed the key — `"Colin Powell"`. That string came entirely from the dictionary key we built in step 1. CLIP's job ended at step 2.

**Why they don't share state:**
BLIP gets the image and a generic question like "what is this person wearing?" — it has no idea CLIP identified the person as Colin Powell. If you wanted BLIP to answer "what is Colin Powell wearing?", you would inject CLIP's identity prediction into the question text yourself — but BLIP still wouldn't *know* it's Colin Powell, it would just see that text as part of the prompt.

---

### BLIP vs. CLIP — what each model is good at

| Capability | BLIP | CLIP |
|---|---|---|
| Generate a caption for an image | Yes — this is its primary task | No — produces embeddings, not text |
| Answer a question about an image | Yes (VQA variant) | No |
| Match an image to a text description | No | Yes — core strength |
| Find the most similar image to a query image | No | Yes — cosine similarity on embeddings |
| *(query image = the input photo you're searching with; the database images are what you search through)* | | |
| Classify an image into categories (zero-shot) | No | Yes — compare image to label text |
| Identity / face recognition | No | Usable (similarity-based, not trained for it) |
| Works without any fine-tuning on the new task | Partial — needs task-specific variant | Yes — fully zero-shot |
| Outputs human-readable text | Yes | No |
| Outputs a fixed-size embedding vector | No | Yes (512-dim) |

**One-line summary:**
- BLIP = *"describe or answer questions about what you see"*
- CLIP = *"find things that match each other across image and text"*

---

## Failure Cases & Why They Happen

### Captioning failures (BLIP)
- **Missing people**: BLIP described a living room's furniture but ignored the people in it. Dominant background objects attract more visual attention than smaller foreground subjects.
- **Hallucination**: BLIP generated "a stop sign on a door" when no door existed. Training data bias — stop signs often appear near building entrances, so the model learned a spurious association.
- **Undercounting**: "a sign" (singular) vs. 3 signs actually present. Captioning models default to the most common phrasing rather than counting explicitly.

### VQA failures (BLIP)
- **Wrong color**: "brown" for gray/black cats. Fine-grained color distinctions (gray vs. brown vs. black) are lossy in image embeddings.
- **Undercounting people**: Counted 1 instead of 2. Partially obscured or ambiguous figures (e.g., reflections) are ignored — the model attends to the most salient subject.

### Recognition failures (CLIP)
- **Confused Powell with Rumsfeld**: Both appear in nearly identical press-conference settings (suits, podiums, similar backgrounds). CLIP embeds scene context, not just faces. At LFW's ~62×47 px resolution, facial geometry is largely lost.
- **General failure conditions**: low resolution, similar professional context between identities, unusual pose/lighting not represented in the reference set.

---

## Interesting Finding

When explicitly asked a counting question in VQA mode ("How many signs?"), BLIP correctly answered 3. But in captioning mode it wrote "a sign" — singular. The two tasks use different output heads, and the captioning decoder defaults to the most fluent phrasing rather than precise enumeration. **Knowing this distinction helps: use VQA for precise attributes, captioning for scene summaries.**

---

## Conclusions

- Pretrained multimodal models (BLIP, CLIP) are powerful out of the box with no fine-tuning.
- CLIP cosine-similarity retrieval achieves 91.7% face recognition accuracy on LFW with just 6 reference images per identity.
- Failure modes are predictable: visual dominance bias, hallucination from training co-occurrences, poor fine-grained color/count, and context leakage into embeddings.
- A two-model pipeline (CLIP for identity + BLIP for attributes) works, but the models don't share state — a unified multimodal LLM would handle identity-conditioned attribute questions better.
- Image resolution matters a lot for recognition: LFW's tiny crops hurt accuracy precisely because CLIP can't distinguish faces it can barely see.
