# Task 2 — Visual Question Answering: Results & Insights

## Model
**Salesforce/blip-vqa-base** — same vision encoder as BLIP captioning, fine-tuned for question answering.

---

## Results

### animals (two cats on a couch)
| Type | Question | Answer | Correct? |
|---|---|---|---|
| object-recognition | What animals are in the image? | cats | ✅ |
| counting | How many cats are in the image? | 2 | ✅ |
| color/attribute | What color is the cat on the right? | brown | ❌ (cats are gray/black) |

### people (living room with TV, fireplace, 2 visible figures)
| Type | Question | Answer | Correct? |
|---|---|---|---|
| object-recognition | What is in the background? | windows | ✅ |
| counting | How many people are visible? | 1 | ❌ (2 visible — one inside, one through window) |
| color/attribute | What color is the television? | black | ✅ |

### outdoor (pole with 3 street signs: Gladis, Parking, Detroit)
| Type | Question | Answer | Correct? |
|---|---|---|---|
| object-recognition | What is mounted on the pole? | signs | ✅ |
| counting | How many signs are on the pole? | 3 | ✅ |
| color/attribute | What color is the parking sign? | green | ✅ |

---

## Failure Analysis

### Failure 1 — Wrong color attribute
- **Question:** "What color is the cat on the right?"
- **Predicted:** "brown"
- **Correct:** Gray / dark gray (the cats are gray and black)
- **Why:** BLIP-VQA confused the cat's color with "brown" — a common neighboring color in training data. Fine-grained color distinctions (gray vs. brown vs. black) are difficult because the image encoder compresses color detail, and the model defaults to the most frequent color label for similar-looking fur.

### Failure 2 — Undercounting visible people
- **Question:** "How many people are visible?"
- **Predicted:** "1"
- **Correct:** 2 — one person inside the room, one visible through the window (possibly outside or a reflection)
- **Why:** The model counted only the clearly interior person and ignored the second figure visible through the window. Partially occluded or ambiguous figures are systematically undercounted because the model's attention focuses on the most salient, unobstructed subject.

---

## Key Insights

1. **Counting worked well for objects** — the model correctly counted 2 cats and 3 signs on a pole. Counting fails more when subjects are partially obscured or ambiguous (people through glass).
2. **Color precision is unreliable** — the model answered "brown" for gray/black cats. Similar-hue colors (brown, gray, dark) are frequently confused, likely because training labels are noisy for fine color distinctions.
3. **VQA is more robust than captioning for direct questions** — asking "How many cats?" got the right answer ("2") even though captioning described the same scene as "two cats sleeping on a couch" (which was also correct). Targeted questions can surface information that open-ended captioning misses.
4. **Consistent bias toward giving an answer** — BLIP-VQA always commits to a concrete answer even when uncertain (e.g., "1" person instead of "I can't tell"). The model was trained to minimize abstention, which inflates confident but wrong answers.
