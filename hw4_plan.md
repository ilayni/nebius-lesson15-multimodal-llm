# Plan: Implement HW4 — Multimodal Models (Captioning, VQA, Face Recognition)

## Context
`LLM_Architecture_HW4.ipynb` is a Nebius Academy assignment with three tasks and three empty
`# code goes here` cells. The goal is to fill those cells with a complete, runnable solution that
**strictly follows the assignment's stated requirements and order**.

Decisions locked with the user:
- **Runtime: Google Colab** (free GPU). All heavy models run there.
- **Models: the assignment's free HuggingFace models** — BLIP for captioning, BLIP-VQA for VQA,
  CLIP for Task 3. **Nebius API is dropped** (user only had credits; free path is better and matches the rubric).
- **Task 3 dataset: LFW via `sklearn.datasets.fetch_lfw_people`** — the assignment lists CelebA *and*
  LFW as "Recommended datasets," and LFW auto-downloads with no manual/Kaggle step.
- **Persistence: PyDrive2** writes result artifacts to a specific Google Drive folder (folder name is a
  config variable the user sets).

### Important honesty note (read before approving)
I cannot execute the notebook here (no GPU/torch locally; it's built for Colab), so I cannot observe the
*actual* captions/answers. The **code will be complete and runnable**. The **failure-analysis prose**
(Task 1 ≥2, Task 2 ≥2, Task 3 works/fails) will be written as guided, well-structured templates pre-filled
with the documented, common failure modes of these models (miscounting, missing small objects, color
errors, CLIP confusing similar faces). After the user runs it in Colab, they pick the 2 real failures and
confirm/adjust the prose against their actual outputs.

To keep this from grading as generic boilerplate, every pre-filled analysis cell is wrapped in a
**unmistakable draft marker**: `> **DRAFT — verify against your actual run and edit before submitting:**`.
And each task's code prints a results table **sorted to surface likely failures first** (shortest captions,
single-word VQA answers, lowest-similarity recognitions) so the student finds real failures in seconds.

## Notebook structure changes
Keep all existing markdown cells untouched (assignment text). Add a Setup section after the intro, and
replace each `# code goes here` cell, splitting each task into a few cells for readability. Cell indices
shift as cells are inserted, so edits are described by content/location, not fixed indices.

### A. Setup section (insert after the "Goal" intro markdown, before Task 1)
1. **Markdown** `## Setup (run first)` — short note: Colab + GPU runtime, what gets installed, Drive folder.
2. **Code — installs**: `pip install -q transformers pillow requests scikit-learn matplotlib pydrive2`
   (torch + numpy already on Colab).
3. **Code — imports + device**: torch, PIL, requests, matplotlib, numpy; `device = "cuda" if available`.
4. **Code — Google Drive persistence (PyDrive2 primary + mount fallback)**:
   - `DRIVE_FOLDER_NAME = "HW4_multimodal_outputs"` (user-editable config var, clearly flagged).
   - **Primary (PyDrive2, as requested):** `from google.colab import auth; auth.authenticate_user()` →
     `GoogleAuth()` with `GoogleCredentials.get_application_default()` → `GoogleDrive(gauth)`;
     `get_or_create_folder` (query title+folder mimeType, create if missing) + `save_to_drive(local_path)`
     (`CreateFile({'parents':[{'id':folder_id}]})`, `SetContentFile`, `Upload()`).
   - **Fallback:** `oauth2client` is EOL and this auth path is intermittently broken on current Colab, so
     wrap PyDrive2 init in try/except. On failure, fall back to
     `from google.colab import drive; drive.mount('/content/drive')` and make `save_to_drive` copy files
     into `/content/drive/MyDrive/<DRIVE_FOLDER_NAME>/`. Same end result, much higher reliability.
   - A single `save_to_drive(local_path)` interface so task cells don't care which backend is active;
     also a no-op guard so the notebook still runs (skipping saves) if executed outside Colab.

### B. Task 1 — Image Captioning (replace cell-7 region)
1. **Code — load model**: `Salesforce/blip-image-captioning-base` via `BlipProcessor` + `BlipForConditionalGeneration`
   (comment: swap to `-large` for quality). Move to `device`.
2. **Code — image set**: `IMAGE_URLS` = editable dict mapping each required category to a stable URL,
   covering all five required cases: **people, animals, indoor, outdoor, multiple-objects** (≥5 total).
   Use stable sources (COCO `images.cocodataset.org/val2017/...` + Wikimedia Commons). `download_image(url)`
   helper with a fallback message. Category tag stored per image so requirement coverage is visible.
3. **Code — caption + display loop**: for each image, run BLIP, render with matplotlib (image + caption as
   title), collect `(category, caption)` rows; render a summary table (pandas) of all captions; save the
   gallery figure + a captions CSV to Drive.
4. **Markdown — failure analysis** (≥2): template with **Predicted / Correct / Why** for each, pre-filled
   with the most likely BLIP failure modes for the chosen images; instruct the user to confirm against their run.

### C. Task 2 — Visual Question Answering (replace cell-13 region)
1. **Code — load model**: `Salesforce/blip-vqa-base` via `BlipProcessor` + `BlipForQuestionAnswering`.
2. **Code — Q&A definitions**: reuse ≥3 of Task 1's images; per image a list of ≥3 questions covering the
   three required types: **object-recognition, counting, color/attribute**. Stored as a structured list.
3. **Code — answer + display loop**: for each image, display it and a table of (question, answer);
   `ask(image, question)` helper. Save the per-image Q&A figures + a Q&A CSV to Drive.
4. **Markdown — failure analysis** (≥2): same Predicted/Correct/Why template; lead with BLIP-VQA's
   well-known counting weakness and attribute errors; user confirms with real answers.

### D. Task 3 — Personalized Face Recognition with CLIP (replace cell-18 region)
1. **Code — load LFW**: `fetch_lfw_people(min_faces_per_person=20, color=True, resize=0.5)`; select the
   top **4 identities** by image count, **≥9 images each**; convert each float array → uint8 → `PIL.Image` (RGB).
2. **Code — split**: per identity, **6 reference + 3 query = 9 images** (within the assignment's "5–6
   reference / 2–3 query" ranges **and** safely above the "≥8 images per identity" floor — note 5+2=7 would
   violate it, so the split is pinned, never the lower bounds); show a small grid of reference faces per identity.
3. **Code — CLIP embeddings**: `openai/clip-vit-base-patch32` via `CLIPModel` + `CLIPProcessor`;
   `embed(images)` → `get_image_features`, L2-normalize. Build a **mean reference embedding (centroid)** per identity.
4. **Code — recognition**: for each query, cosine similarity to each identity centroid → predicted identity;
   compute **accuracy** + a small **confusion matrix**; display example queries with predicted vs true label.
5. **Code — personalized Q&A** (assignment's exact structure): combine **CLIP (who)** + **BLIP-VQA (attributes)**:
   - "Who is in the image?" → `"This is probably <name>."`
   - "Is this <X>?" → yes/no from CLIP prediction.
   - "What is <name> wearing/doing?" → BLIP-VQA answer on the query image.
   - Print the combined sentence as in the assignment example. Save result figures to Drive.
6. **Markdown — analysis**: when it works vs fails. Frame the key point honestly — CLIP is a general
   image-text model, **not** a dedicated face recognizer, so it works on visually distinct identities and
   fails on similar-looking ones / low-res LFW crops. Include accuracy number placeholder for the user to fill.

## Files modified
- `LLM_Architecture_HW4.ipynb` — only file edited (via NotebookEdit: insert Setup cells; replace the three
  `# code goes here` cells, splitting into the cells above).

## Verification
Cannot run locally. Verification is in Colab by the user:
1. Open the notebook in Colab, set Runtime → GPU, set `DRIVE_FOLDER_NAME`.
2. Run Setup → authorize Drive (PyDrive2 popup, or mount fallback).
3. Run Task 1 → 5+ captioned images display, captions CSV + gallery land in the Drive folder.
4. Run Task 2 → 3 images × 3 Q&A tables display, Q&A CSV saved.
5. Run Task 3 → LFW loads, accuracy + confusion matrix print, example recognitions + combined Q&A display, figures saved.
6. Fill in / confirm the failure-analysis markdown cells from the real outputs.
Also a static review: confirm every cell's imports resolve in order, no cell depends on a later cell, and
requirement coverage (5 categories, 3 question types, 3–5 identities) is explicit.
