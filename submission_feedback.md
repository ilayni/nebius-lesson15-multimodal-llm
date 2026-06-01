# Submission Feedback — `LLM_Architecture_HW4_colab_ilay_nisim_boris_kogan.ipynb`

Overall the notebook is in good shape: all cells ran end-to-end, all required deliverables are present, figures and CSVs uploaded to Drive, and the analyses are written in a clear, student-like voice. Below are the issues I'd fix or double-check **before submitting**, listed by severity.

---

## 📊 Grade (my assessment)

This is **not** an official rubric score — I'm inferring deductions from the stated point values and what I can see in your outputs and images. Treat it as a strong estimate.

| Task | Points | My estimate | Why |
|---|---|---|---|
| **Task 1 — Captioning** | 30 | **~29** | All 5 required categories present (people/animals/indoor/outdoor/multiple-objects), captions displayed in gallery + table + CSV, 2 solid failure analyses. The earlier indoor/outdoor mislabel (#1) is now **fixed** — Failure 2 correctly says "indoor image." |
| **Task 2 — VQA** | 30 | **~27** | 3 images × 3 questions covering all three required types (object / counting / color), displayed clearly, 2 failure analyses. The undercount failure is solid; the cat-color failure is debatable and one explanation detail is geometrically off (see #4). |
| **Task 3 — Face Recognition** | 40 | **~38** | Complete and strong: 4 identities, 6 reference + 3 query each, CLIP centroids, cosine similarity, **91.7% accuracy**, confusion matrix, personalized Q&A in the exact template, and a genuinely insightful works/fails analysis (Powell→Rumsfeld via shared press-conference context + low res). **−** minor grammar (see #5). |
| **Total** | **100** | **~94** | Solid A. The only objective error (#1) is already fixed; remaining notes are polish. |

---

## 🔴 Must-fix before submission

### 1. Task 1, Failure 2 — wrong image attributed ✅ FIXED

This was the one blocker. Your earlier draft attributed the *"a stop sign on a door"* caption to the **outdoor** image, but it was generated for the **indoor** image (`000000297343.jpg`). The current notebook now correctly reads:

> "For the **indoor** image, BLIP returned: *'a stop sign on a door'*"

Verified against the regenerated md (Task 1 Failure 2). Nothing more to do here — the hallucination analysis is sound and the attribution now matches the visible output.

### 2. Task 1, footnote about counting — partially undermined by Task 2

Your Task 1 note says BLIP "struggles with counting" because it captioned the outdoor pole as having "a sign" (singular) when there are 3. That's still true for the *captioning* model. **But** in Task 2, you asked the VQA model "How many signs are on the pole?" and it answered `3` — correctly!

**Background — captioning vs. VQA:**
BLIP is trained on two separate tasks with two separate output heads (a "head" is the final layer that produces the output):

- The **captioning head** was trained to produce natural-sounding sentences that match how humans typically describe images. Humans tend to write *"a pole with a sign"* rather than *"a pole with 3 signs"* — so the model learned that phrasing as the default, even when there are multiple objects.
- The **VQA head** was trained to answer specific questions accurately. When the question is *"how many?"*, the model is forced to count rather than just generate the most common phrase.

So the difference isn't really about the model's ability to count — it *can* count, but only when forced to. Captioning optimizes for fluent, natural description and takes the easy/common phrasing shortcut; VQA must actually look and answer the specific question.

**✅ This is now in the notebook** under the "Interesting finding" heading after Task 2 Failure 2:

> "When explicitly asked in Task 2, BLIP-VQA correctly counted 3 signs — so the issue isn't that the model can't count, but that when asking a general question like in task1, the model routes to captioning output which defaults to the most common phrasing ('a sign') rather than another output layer which is used when checking quantity explicitly."

That ties Tasks 1 and 2 together exactly as intended — good. Two micro-nits only: typo **"explictily" → "explicitly"**, and **"Intersting" → "Interesting"** in the heading. Content is spot-on.

---

## 🟡 Worth checking / minor

### 3. The "people" image was changed (verified — you're fine)

The original notebook used COCO `000000074816.jpg` for the "people" category. The submission uses `000000000139.jpg`. I looked at the image: there **are** people in it — at least one or two small figures standing near the dining table in the background. So your Failure 1 framing ("BLIP ignored the people") holds: the people are genuinely present but small, and BLIP described the dominant furniture/TV instead. No change needed — just be aware the people are background-scale, so keep the framing as "BLIP ignored the *small/background* people" rather than implying they're prominent.

### 4. Task 2 failure analyses — two small accuracy notes

**(a) Cat-color (Failure 1).** Your analysis says the cats in `000000039769.jpg` are "gray and black" and that "brown" is wrong. I looked at the actual image — these are **tabby** cats, and the one on the right reads as brown/tan to me as much as gray. "brown," "gray," and "tabby" are all defensible here, so a grader who looks at the image may not agree that BLIP was *wrong*. The failure is still presentable (BLIP commits to a single flat color label), but hedge so you're not over-claiming:

> "The cat on the right looks more gray/tabby than brown to me, though color naming on a tabby is admittedly subjective."

**(b) "Through the window" (Failure 2) — you've hedged this, mostly fine.** Your undercount failure is solid (BLIP answered `1`, there are ≥2 people), and you already acknowledge you can't tell exactly what the second figure is. Good — that honesty covers it. One thing to **double-check is saved**: the version in this `.ipynb` still reads *"a second figure visible through the window (either a reflection or someone outside)"* and hedges *"hard to say if this was a reflection or a person outside."* It does **not** yet list "a person standing near the table" as an option. That third option is the one that best matches what I see in the image, so if you added it, make sure it's saved to the submitted notebook (it may be unsaved in Colab). Suggested final wording:

> "…a second figure further back in the room — hard to say at this resolution whether it's a reflection, someone standing near the table, or a person outside the window. Either way BLIP only counted the obvious foreground person."

### 5. Task 3 analysis — small grammar fix

In the "When it works" section:

> "I also think the tie on bush is not seen. the suit are barely seen, and powell seems to me more talking than staring but it is hard to say."

A few small issues:
- "the suit are" → "the suits are"
- "bush" → "Bush" (capitalize, since you capitalize Powell etc.)
- Could be one cleaner sentence:

> "Looking at the images, I don't actually see Bush's tie, the suits are barely visible, and Powell looks more like he's talking than staring — but at this resolution it's hard to be sure."

The candor is great — just tighten the wording.

### 6. Drive upload — one failure on first run

In the Task 1 cell, `task1_gallery.png` failed to upload the first time:

```
⚠  Upload failed for 'task1_gallery.png': EOF occurred in violation of protocol (_ssl.c:2427)
```

You manually re-ran `save_to_drive("/content/task1_gallery.png")` in the next cell and it succeeded. **Verify** the Drive folder actually has `task1_gallery.png` — the retry worked from the log, but if the grader is checking Drive uploads, confirm the file is there.

---

## 🟢 What's good (keep as is)

- **All required deliverables present** — 5 captioned images, 3 images × 3 questions Q&A, 4 identities × (6 reference + 3 query) recognition, personalized Q&A, accuracy, confusion matrix, failure analyses.
- **Task 3 accuracy of 11/12 (91.7%)** with one Powell→Rumsfeld misclassification — that's a really clean result and the analysis correctly explains it via shared press-conference context + low resolution.
- **Confusion matrix matches the textual analysis** (Powell row shows the single misclassification).
- **Personalized Q&A output is structured exactly per the assignment template** ("This is probably X. / Is this Y?" / wearing / doing).
- **Analyses are written in the first person** with hedging and genuine reflection — they read like a student wrote them, which is what you want.
- **All figures + CSVs uploaded to Drive** with file IDs printed (good evidence trail).

---

## Summary

**Estimated grade: ~94/100 (solid A).** Complete, all deliverables present, Task 3 is excellent.

**Blocker:** None — the one objective error (#1, indoor/outdoor mislabel) is **already fixed**.

**Already addressed (verified in the current notebook):** #1 (fixed), #2 (the captioning-vs-VQA tie-in is in under "Interesting finding"), #4(b) (you've hedged the second-figure uncertainty).

**Remaining — all minor polish, none blocking:**
- **#4(b) save-check** — confirm your "standing near the table" wording is actually saved to the submitted `.ipynb` (on disk it still says only "through the window / reflection / outside").
- **#2 typos** — "explictily" → "explicitly", "Intersting" → "Interesting".
- **#4(a)** hedge the cat color, **#5** grammar in Task 3 "when it works", **#6** confirm the Drive upload.
- **#3** — already verified for you (people image is fine).

You're in great shape. The remaining items are spelling/polish, not substance.
