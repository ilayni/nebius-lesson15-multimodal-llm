# Homework: Multimodal Models - Image Captioning and Visual Question Answering

**Goal**

In this assignment, you will experiment with multimodal models that combine vision and language.

You will use a pretrained model to:

1. Generate a caption for an image.
2. Answer questions about an image.

You will also analyze the model's strengths, weaknesses, and possible failure cases.

## Setup (run first)

> **Before running:** make sure you are on a **GPU runtime** in Colab  
> (Runtime → Change runtime type → L4 / T4 GPU).
>
> Make sure you have a Colab Secret named **`FOLDER_ID`** containing the ID of your  
> Google Drive folder (the long string after `/folders/` in the folder URL).  
> Enable notebook access to it via the 🔑 icon in the left sidebar.
>
> Then run the three cells below in order.


```python
# Install required packages (torch, numpy, google-api-python-client already on Colab)
!pip install -q transformers pillow requests scikit-learn matplotlib pandas
```


```python
import os, io, csv, requests
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✓ Device: {device}")
if device == "cpu":
    print("  ⚠ No GPU detected — models will be slow. Check Runtime → Change runtime type.")
```


```python
# ── Google Drive persistence ─────────────────────────────────────────────────
# Tries drive.file scope first (restricted: only files this app creates).
# Falls back to standard full-Drive auth on older Colab versions.
#
# Uses google-api-python-client (pre-installed on Colab — no extra pip needed).
# Reads your folder ID from a Colab Secret named FOLDER_ID.

import os

_drive_service = None   # googleapiclient Drive v3 service
_folder_id     = None   # destination folder ID
_drive_ready   = False  # set to True after successful init


def _init_drive():
    global _drive_service, _folder_id, _drive_ready

    # ── Detect Colab ──────────────────────────────────────────────────────────
    try:
        import google.colab          # noqa – just testing import
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if not IN_COLAB:
        print("Not running in Colab — Drive saves will be skipped.")
        return

    # ── Read folder ID from Colab Secret ─────────────────────────────────────
    try:
        from google.colab import userdata
        _folder_id = userdata.get('FOLDER_ID')
        if not _folder_id:
            raise ValueError("Secret value is empty")
        print(f"✓ FOLDER_ID read from secret: {_folder_id[:8]}…")
    except Exception as e:
        print(f"⚠  Could not read FOLDER_ID secret: {e}")
        print("   → In Colab: click 🔑 in the left sidebar → New secret")
        print("     Name: FOLDER_ID  |  Value: <your folder ID>")
        return

    # ── Authenticate ──────────────────────────────────────────────────────────
    from google.colab import auth as colab_auth

    try:
        # Newer Colab: request only drive.file scope (restricted — app-created files only)
        colab_auth.authenticate_user(scopes=['https://www.googleapis.com/auth/drive.file'])
        print("✓ Authenticated with restricted scope (drive.file)")
    except TypeError:
        # Older Colab: scopes parameter not supported — fall back to full Drive auth
        colab_auth.authenticate_user()
        print("✓ Authenticated (standard scope — older Colab version)")

    # ── Build Drive API client ────────────────────────────────────────────────
    try:
        import google.auth
        from googleapiclient.discovery import build

        creds, _ = google.auth.default()
        _drive_service = build('drive', 'v3', credentials=creds)
        _drive_ready = True
        print("✓ Drive ready — uploads will go to your folder")
    except Exception as e:
        print(f"⚠  Drive API init failed: {e}")
        print("   Files will be saved locally in /content/ only.")


def save_to_drive(local_path: str):
    """Upload a local file to the configured Drive folder.
    Safe to call even if Drive init failed — prints a warning and continues.
    """
    global _drive_ready
    if not _drive_ready:
        _init_drive()          # lazy init on first call

    fname = os.path.basename(local_path)

    if not _drive_ready:
        print(f"  (Drive not configured — '{fname}' saved locally only)")
        return

    try:
        from googleapiclient.http import MediaFileUpload
        metadata = {'name': fname, 'parents': [_folder_id]}
        media    = MediaFileUpload(local_path, resumable=False)
        result   = _drive_service.files().create(
                       body=metadata, media_body=media, fields='id'
                   ).execute()
        print(f"  ↑ Saved to Drive: {fname}  (id: {result['id'][:12]}…)")
    except Exception as e:
        print(f"  ⚠  Upload failed for '{fname}': {e}")


# ── Initialise now so the auth popup appears before any task cell runs ────────
_init_drive()
```

## Task 1 - Image Captioning (30 Points)

**Description**

Given an input image, use a pretrained multimodal model to generate a textual description of the image.

Example:

**Input**: An image of a dog running in a park.

**Output**: A dog is running on grass in a park.

**Requirements**

Use at least 5 different images.

The images should include:

* At least one image with people
* At least one image with animals
* At least one indoor scene
* At least one outdoor scene
* At least one image with multiple objects

You may use your own images or images from an open dataset.

**Suggested Models**

You may use one of the following:

* Salesforce/blip-image-captioning-base
* Salesforce/blip-image-captioning-large
* microsoft/git-base
* Any other image-captioning model from HuggingFace

**Deliverables**

1. Present your images and model captions clearly

    Display each image together with the caption generated by the model.

2. Analyze model failures

    Choose at least 2 examples where the model made a mistake or gave an incomplete caption.

    For each example, explain:

    * What the model predicted
    * What the correct description should be
    * Why the model may have failed

    For example :

    "The model described the image as “a person sitting at a table,” but it missed the laptop and the coffee cup.
    This may have happened because the person was the most visually dominant object, while the smaller objects were less noticeable."



```python
# ── Task 1 · Load BLIP Image Captioning Model ────────────────────────────────
from transformers import BlipProcessor, BlipForConditionalGeneration

print("Loading BLIP image captioning model…  (first run downloads ~1 GB)")
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = (BlipForConditionalGeneration
                 .from_pretrained("Salesforce/blip-image-captioning-base")
                 .to(device))
caption_model.eval()
print("✓ Model ready on", device)
```


```python
# ── Task 1 · Images & Captions ───────────────────────────────────────────────
# Five COCO val2017 images — one per required category.
# Replace any URL with your own image if you prefer.

IMAGE_URLS = {
    "people":           "http://images.cocodataset.org/val2017/000000074816.jpg",
    "animals":          "http://images.cocodataset.org/val2017/000000039769.jpg",
    "indoor":           "http://images.cocodataset.org/val2017/000000297343.jpg",
    "outdoor":          "http://images.cocodataset.org/val2017/000000058636.jpg",
    "multiple_objects": "http://images.cocodataset.org/val2017/000000252219.jpg",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def download_image(url):
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        print(f"  ⚠ Download failed ({url}): {e}")
        return None

def generate_caption(image):
    inputs = caption_processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = caption_model.generate(**inputs, max_new_tokens=50)
    return caption_processor.decode(ids[0], skip_special_tokens=True)

# ── Generate captions ─────────────────────────────────────────────────────────

results = []
for category, url in IMAGE_URLS.items():
    print(f"[{category}] downloading…", end=" ")
    img = download_image(url)
    if img is None:
        continue
    caption = generate_caption(img)
    results.append({"category": category, "caption": caption, "image": img})
    print(f'→ "{caption}"')

# ── Gallery ───────────────────────────────────────────────────────────────────

n = len(results)
fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
axes = [axes] if n == 1 else list(axes)

for ax, row in zip(axes, results):
    ax.imshow(row["image"])
    ax.set_title(f"[{row['category']}]\n{row['caption']}", fontsize=9, pad=6)
    ax.axis("off")

plt.suptitle("Task 1 — BLIP Image Captions", fontsize=12, y=1.02)
plt.tight_layout()

gallery_path = "/content/task1_gallery.png"
plt.savefig(gallery_path, dpi=120, bbox_inches="tight")
plt.show()
save_to_drive(gallery_path)

# ── Summary table + CSV ───────────────────────────────────────────────────────

df_captions = pd.DataFrame([
    {"Category": r["category"], "Caption": r["caption"]} for r in results
])
print("\n── Caption Summary ──────────────────────────────────────────────")
try:
    display(df_captions)
except NameError:
    print(df_captions.to_string(index=False))

csv_path = "/content/task1_captions.csv"
df_captions.to_csv(csv_path, index=False)
save_to_drive(csv_path)
print("✓ Gallery and CSV saved.")
```

### Task 1 — Failure Analysis

---

**Failure 1 — Missing subjects (background dominates)**

| | |
|---|---|
| **Predicted** | "a living room with a television and a fireplace" |
| **Correct description** | A living room scene with people present |
| **Why the model failed** | The background (furniture, TV, fireplace) occupied more pixels than the people, so BLIP described the environment and ignored the subjects entirely. |

---

**Failure 2 — Hallucination (invented object)**

| | |
|---|---|
| **Predicted** | "a stop sign on a door" |
| **Correct description** | A stop sign — no door is present in the image |
| **Why the model failed** | BLIP hallucinated a "door" that does not exist. The model likely associated the stop sign with a door frame from similar training images, generating a plausible-sounding but factually incorrect description. |

> **Note:** A third failure is also visible in the outdoor image — the model described "a pole with a sign" when the pole actually has 3 signs. This shows BLIP's known weakness with counting repeated objects.

## Task 2 - Visual Question Answering (30 points)

**Description**

Given an image and a natural-language question, use a pretrained multimodal model to answer the question.

Example:

**Input image**: A kitchen scene

**Question**: How many chairs are visible?

**Output**: Four.

**Requirements**

Use at least 3 images.

For each image, ask at least 3 questions.

Your questions should include:

* One object-recognition question
* One counting question
* One color or attribute question

Example questions:
* What animal is in the image?
* How many people are visible?
* What color is the car?

**Suggested Models**

You may use:

* Salesforce/blip-vqa-base
* Salesforce/blip-vqa-capfilt-large
* Any other VQA model from HuggingFace

**Deliverables**

1. Present your images, questions and model's answers clearly

    Display each image together with the Q&A generated by the model.

2. Analyze model failures

    Choose at least 2 examples where the model made a mistake or gave an incomplete caption.

    For each example, explain:

    * What the model predicted
    * What the correct description should be
    * Why the model may have failed

    **You are required to find examples where it fails and analyse why**



```python
# ── Task 2 · Load BLIP VQA Model ─────────────────────────────────────────────
from transformers import BlipForQuestionAnswering

print("Loading BLIP VQA model…  (weights are shared with captioning — fast if already cached)")
vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
vqa_model = (BlipForQuestionAnswering
             .from_pretrained("Salesforce/blip-vqa-base")
             .to(device))
vqa_model.eval()
print("✓ BLIP VQA model ready on", device)
```


```python
# ── Task 2 · Visual Question Answering ───────────────────────────────────────
# Reuses 3 images from Task 1. Each image gets 3 questions:
# one object-recognition, one counting, one color/attribute.

VQA_QUESTIONS = {
    "animals": [
        ("What animals are in the image?",      "object-recognition"),
        ("How many cats are in the image?",     "counting"),
        ("What color is the cat on the right?", "color/attribute"),
    ],
    "people": [
        ("What is in the background?",          "object-recognition"),
        ("How many people are visible?",        "counting"),
        ("What color is the television?",       "color/attribute"),
    ],
    "outdoor": [
        ("What is mounted on the pole?",        "object-recognition"),
        ("How many signs are on the pole?",     "counting"),
        ("What color is the parking sign?",     "color/attribute"),
    ],
}

# ── Helper ────────────────────────────────────────────────────────────────────

def ask(image, question):
    inputs = vqa_processor(image, question, return_tensors="pt").to(device)
    with torch.no_grad():
        out = vqa_model.generate(**inputs, max_new_tokens=20)
    return vqa_processor.decode(out[0], skip_special_tokens=True)

# ── Run Q&A ───────────────────────────────────────────────────────────────────

image_by_category = {r["category"]: r["image"] for r in results}
all_qa_rows = []

for category, questions in VQA_QUESTIONS.items():
    img = image_by_category.get(category)
    if img is None:
        print(f"⚠ '{category}' image not found in Task 1 results — skipping")
        continue

    print(f"\n── [{category}] ──────────────────────────────────────────────────")
    qa_rows = []
    for question, qtype in questions:
        answer = ask(img, question)
        qa_rows.append({"Type": qtype, "Question": question, "Answer": answer})
        print(f"  [{qtype:>20}]  {question}  →  {answer}")

    # ── Display: image + Q&A table side by side ───────────────────────────────
    fig, (ax_img, ax_tbl) = plt.subplots(
        1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [1, 2]}
    )

    ax_img.imshow(img)
    ax_img.set_title(f"[{category}]", fontsize=11)
    ax_img.axis("off")

    df_qa = pd.DataFrame(qa_rows)
    ax_tbl.axis("off")
    tbl = ax_tbl.table(
        cellText=df_qa.values,
        colLabels=df_qa.columns,
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(df_qa.columns))))
    tbl.scale(1, 1.6)

    plt.tight_layout()
    fig_path = f"/content/task2_qa_{category}.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.show()
    save_to_drive(fig_path)

    all_qa_rows.extend([{"Category": category, **row} for row in qa_rows])

# ── Save combined CSV ─────────────────────────────────────────────────────────
csv_path = "/content/task2_qa.csv"
pd.DataFrame(all_qa_rows).to_csv(csv_path, index=False)
save_to_drive(csv_path)
print("\n✓ All Q&A figures and CSV saved.")
```

### Task 2 — Failure Analysis

---

**Failure 1 — Wrong color attribute**

| | |
|---|---|
| **Question** | "What color is the cat on the right?" |
| **Predicted** | "brown" |
| **Correct answer** | Gray / dark gray (the cats are gray and black) |
| **Why the model failed** | BLIP-VQA confused the cat's color with "brown" — a common neighboring color in training data. Fine-grained color distinctions (gray vs. brown vs. black) are difficult because the image encoder compresses color detail, and the model defaults to the most frequent color label seen for similar-looking fur in training. |

---

**Failure 2 — Undercounting visible people**

| | |
|---|---|
| **Question** | "How many people are visible?" |
| **Predicted** | "1" |
| **Correct answer** | 2 — one person inside the room, one visible through the window (possibly outside or a reflection) |
| **Why the model failed** | The model counted only the clearly interior person and ignored the second figure visible through the window. Partially occluded or ambiguous figures (reflections, people behind glass) are systematically undercounted because the model's attention focuses on the most salient, unobstructed subject. |

## Task 3 - Personalized Face Recognition with a Celebrity Dataset (40 points)


**Goal**

Build a small personalized multimodal recognition system.

For this task, you will use a public celebrity face dataset rather than private photos of family or friends.

Although the same approach could easily be adapted to recognize your own friends, family members, or personal image collections, in this assignment you must use a celebrity dataset in order to avoid privacy concerns and to ensure a consistent evaluation setting across all students.

Students who are interested may later explore how the same techniques can be extended to build personalized multimodal recognition systems for recreational or educational purposes outside the scope of this course.

The system should learn to recognize a small set of known identities and answer questions such as:

* Who is in the image?
* Is this person Emma Watson or Leonardo DiCaprio?
* What is Emma doing or wearing?

Recommended datasets:

* CelebA — contains over 200K celebrity face images, 10,177 identities, and 40 attribute annotations. https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
* LFW — Labeled Faces in the Wild — designed for unconstrained face recognition and face verification.
https://scikit-learn.org/0.19/datasets/labeled_faces.html
* You may also scrape or download images from the internet in order to build your celebrity dataset.

1. Choose 3-5 identities from a celebrity dataset. For each identity, collect at least 8 images.

2. split the images into:
   * Train/reference set: 5-6 images per identity
   * Test/query set: 2-3 images per identity

3. Use a pretrained model to convert each image into an embedding vector. Recommended model : CLIP

4. Compare a query image to the known identity images using cosine similarity

5. Use the predicted identity to answer simple personalized questions.

    Note:

    CLIP similarity predicts who the person is; a VQA/captioning model answers visual attribute/action questions about the image.

    Use this structure:
    1. CLIP similarity → Who is this?
    2. VQA model → What is the person wearing / doing / holding?
    3. Combined answer → “This is probably Emma Watson. She appears to be wearing a black dress.”

              Example:

              Question: Who is in the image?

              Answer: This is probably Emma Watson.

              Question: Is this Barack Obama?

              Answer: No, this is probably Taylor Swift.

              Question: What is Emma wearing?

              Answer: The model appears to show Emma wearing a black jacket.

6. Analyze when the system works and when it fails. Show images , Q&A and disscuess results.


```python
# ── Task 3 · Cell 1: Load LFW & Build Reference / Query Sets ─────────────────
from sklearn.datasets import fetch_lfw_people
from collections import Counter

print("Loading LFW dataset…  (first run downloads ~200 MB)")
lfw = fetch_lfw_people(min_faces_per_person=20, color=True, resize=0.5)
print(f"✓ {lfw.images.shape[0]} images | {len(lfw.target_names)} identities")
print(f"  Image shape: {lfw.images.shape[1:]}")

# ── Select top 4 identities by image count ────────────────────────────────────
counts   = Counter(lfw.target.tolist())
top4_ids = [idx for idx, _ in counts.most_common(4)]
names    = [lfw.target_names[i] for i in top4_ids]
print("\nTop 4 identities:")
for name, idx in zip(names, top4_ids):
    print(f"  {name}: {counts[idx]} images")

# ── Filter to top 4 and remap labels to 0-3 ──────────────────────────────────
mask        = np.isin(lfw.target, top4_ids)
images_all  = lfw.images[mask]
id_remap    = {old: new for new, old in enumerate(top4_ids)}
targets_all = np.array([id_remap[t] for t in lfw.target[mask]])

# ── Helper: float array → PIL RGB ────────────────────────────────────────────
def arr_to_pil(arr):
    return Image.fromarray((arr * 255).astype(np.uint8), "RGB")

# ── Split: 6 reference + 3 query per identity ────────────────────────────────
np.random.seed(42)
reference_images = {}   # name → list of 6 PIL images
query_images     = {}   # name → list of 3 PIL images

for i, name in enumerate(names):
    idx = np.where(targets_all == i)[0].copy()
    np.random.shuffle(idx)
    reference_images[name] = [arr_to_pil(images_all[j]) for j in idx[:6]]
    query_images[name]     = [arr_to_pil(images_all[j]) for j in idx[6:9]]

# ── Show reference grid (4 identities × 6 images) ────────────────────────────
fig, axes = plt.subplots(4, 6, figsize=(14, 10))
for row, (name, imgs) in enumerate(reference_images.items()):
    axes[row][0].set_ylabel(name.split()[-1], fontsize=9,
                            rotation=0, labelpad=55, va="center")
    for col, img in enumerate(imgs):
        axes[row][col].imshow(img)
        axes[row][col].axis("off")

plt.suptitle("Reference faces — 6 images per identity", fontsize=12)
plt.tight_layout()
ref_path = "/content/task3_reference_grid.png"
plt.savefig(ref_path, dpi=100, bbox_inches="tight")
plt.show()
save_to_drive(ref_path)
print("✓ Reference grid saved.")
```


```python
# ── Task 3 · Cell 2: CLIP Embeddings & Identity Centroids ────────────────────
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F

print("Loading CLIP model…")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_model.eval()
print("✓ CLIP ready on", device)

def embed_images(imgs):
    """L2-normalised image embeddings for a list of PIL images. Returns (N, 512) tensor."""
    inputs       = clip_processor(images=imgs, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        # Access vision model directly — avoids get_image_features() version quirks
        vision_out = clip_model.vision_model(pixel_values=pixel_values)
        feats      = clip_model.visual_projection(vision_out.pooler_output)  # (N, 512)
    return F.normalize(feats, dim=-1)

# ── Build mean centroid per identity ─────────────────────────────────────────
print("\nBuilding reference centroids…")
centroids = {}
for name, imgs in reference_images.items():
    embs            = embed_images(imgs)                                     # (6, 512)
    centroid        = F.normalize(embs.mean(dim=0, keepdim=True), dim=-1)   # (1, 512)
    centroids[name] = centroid
    print(f"  {name.split()[-1]:20s}  centroid norm: {centroid.norm().item():.4f}")

centroid_matrix = torch.cat(list(centroids.values()), dim=0)  # (4, 512)
identity_names  = list(centroids.keys())
print("✓ Centroids ready.")
```


```python
# ── Task 3 · Cell 3: Recognition, Accuracy & Confusion Matrix ────────────────
correct   = 0
total     = 0
confusion = np.zeros((4, 4), dtype=int)
all_preds = []   # (true_name, pred_name, is_correct, sim_score, image)

for true_name, imgs in query_images.items():
    true_idx = identity_names.index(true_name)
    for img in imgs:
        emb      = embed_images([img])                       # (1, 512)
        sims     = (emb @ centroid_matrix.T).squeeze(0)     # (4,)
        pred_idx = int(sims.argmax())
        pred_name = identity_names[pred_idx]
        ok        = (pred_name == true_name)
        correct  += int(ok)
        total    += 1
        confusion[true_idx][pred_idx] += 1
        all_preds.append((true_name, pred_name, ok, float(sims[pred_idx]), img))

accuracy = correct / total
print(f"Accuracy: {correct}/{total} = {accuracy:.1%}\n")

# ── Confusion matrix ──────────────────────────────────────────────────────────
short = [n.split()[-1] for n in identity_names]
df_conf = pd.DataFrame(confusion,
                        index   =[f"True: {s}"  for s in short],
                        columns =[f"Pred: {s}"  for s in short])
try:
    display(df_conf)
except NameError:
    print(df_conf.to_string())

# ── Visual: first 8 query predictions ────────────────────────────────────────
n_show = min(8, len(all_preds))
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, (true_name, pred_name, ok, sim, img) in zip(axes.flatten(), all_preds[:n_show]):
    ax.imshow(img)
    color = "green" if ok else "red"
    ax.set_title(
        f"True: {true_name.split()[-1]}\nPred: {pred_name.split()[-1]}\nsim={sim:.2f}",
        fontsize=8, color=color
    )
    ax.axis("off")

plt.suptitle(f"Query predictions  (accuracy: {accuracy:.1%})", fontsize=11)
plt.tight_layout()
recog_path = "/content/task3_recognition.png"
plt.savefig(recog_path, dpi=120, bbox_inches="tight")
plt.show()
save_to_drive(recog_path)
print(f"✓ Recognition figure saved.")
```


```python
# ── Task 3 · Cell 4: Personalized Q&A (CLIP + BLIP-VQA) ─────────────────────
# Note: uses ask() and vqa_model from Task 2 — make sure those cells have run.
# CLIP predicts who the person is; BLIP-VQA answers attribute questions.

print("── Personalized Q&A ─────────────────────────────────────────────────────\n")

fig, axes = plt.subplots(1, 4, figsize=(16, 5))

for col_idx, (true_name, imgs) in enumerate(query_images.items()):
    img  = imgs[0]   # use first query image per identity
    last = true_name.split()[-1]

    # ── CLIP: who is this? ────────────────────────────────────────────────────
    emb       = embed_images([img])
    sims      = (emb @ centroid_matrix.T).squeeze(0)
    pred_idx  = int(sims.argmax())
    pred_name = identity_names[pred_idx]
    pred_last = pred_name.split()[-1]
    sim_score = float(sims[pred_idx])
    correct_id = (pred_name == true_name)

    # ── BLIP-VQA: attributes ──────────────────────────────────────────────────
    wearing = ask(img, "What is the person wearing?")
    doing   = ask(img, "What is the person doing?")

    # ── Print combined Q&A ────────────────────────────────────────────────────
    print(f"── Image: {last} ──────────────────────────────────────────────────")
    print(f"  Q: Who is in the image?")
    print(f"  A: This is probably {pred_last}.  (similarity: {sim_score:.3f})")
    print()
    print(f"  Q: Is this {last}?")
    if correct_id:
        print(f"  A: Yes, this is probably {last}.")
    else:
        print(f"  A: No, this is probably {pred_last}.")
    print()
    print(f"  Q: What is {pred_last} wearing?")
    print(f"  A: {pred_last} appears to be wearing {wearing}.")
    print()
    print(f"  Q: What is {pred_last} doing?")
    print(f"  A: {pred_last} appears to be {doing}.")
    print()

    # ── Display ───────────────────────────────────────────────────────────────
    ax = axes[col_idx]
    ax.imshow(img)
    color = "green" if correct_id else "red"
    ax.set_title(
        f"True: {last}\nPred: {pred_last}\n{wearing}",
        fontsize=7, color=color, pad=4
    )
    ax.axis("off")

plt.suptitle("Personalized Q&A — CLIP (identity) + BLIP-VQA (attributes)", fontsize=11)
plt.tight_layout()
qa_path = "/content/task3_personalized_qa.png"
plt.savefig(qa_path, dpi=120, bbox_inches="tight")
plt.show()
save_to_drive(qa_path)
print("✓ Personalized Q&A figure saved.")
```

### Task 3 — Analysis

**Accuracy:** [fill in after running — e.g. "10/12 = 83%"]

---

**When the system works**

The system performs well when identities are visually distinct — different facial structure, skin tone, or hair style creates well-separated embedding clusters. With 6 reference images per identity, the mean centroid averages out pose and lighting variation, making the representation more robust than matching against a single image.

---

**When the system fails**

CLIP is a general image-text model, not a dedicated face recognizer. It fails when:

- **Visually similar identities** — people who look alike (similar age, skin tone, facial structure) produce overlapping embeddings that the cosine similarity cannot separate.
- **Low resolution** — LFW crops at `resize=0.5` are small (~62×47 px). Fine-grained facial details (eye shape, jawline) are lost, reducing discriminability.
- **Pose and expression variation** — if a query image shows a very different angle or expression than the reference set, the embedding drifts away from the centroid.

---

**Limitations of the combined Q&A**

The BLIP-VQA attribute answers (wearing, doing) are generated from the face crop alone — LFW images show mostly the face and little of the body. This means "What is the person wearing?" often returns a generic guess ("a suit", "a shirt") rather than a precise description. The two models operate independently: CLIP identifies who, BLIP-VQA describes what, with no shared context between them.
