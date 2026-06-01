# Task 1 — Image Captioning: Results & Insights

## Model
**Salesforce/blip-image-captioning-base** — encoder-decoder vision-language model.

---

## Results

| Category | Caption |
|---|---|
| people | "a living room with a television and a fireplace" |
| animals | "two cats sleeping on a couch" |
| indoor | "a stop sign on a door" |
| outdoor | "a pole with a sign" |
| multiple_objects | "a woman walking down the street" |

---

## Failure Analysis

### Failure 1 — Missing subjects (background dominates)
- **Predicted:** "a living room with a television and a fireplace"
- **Correct:** A living room scene with people present
- **Why:** The background (furniture, TV, fireplace) occupied more pixels than the people, so BLIP described the environment and ignored the subjects entirely.

### Failure 2 — Hallucination (invented object)
- **Predicted:** "a stop sign on a door"
- **Correct:** A stop sign — no door is present in the image
- **Why:** BLIP hallucinated a "door" that does not exist. The model likely associated the stop sign with a door frame from similar training images, generating a plausible-sounding but factually incorrect description.

### Bonus Observation — Undercounting
- **Predicted:** "a pole with a sign" (singular)
- **Correct:** A pole with 3 signs
- **Why:** BLIP-base has a known weakness with counting repeated similar objects. It defaults to describing the category ("a sign") without enumerating instances.

---

## Key Insights

1. **Scene bias** — BLIP tends to describe the dominant visual region. If background elements are large, they overshadow smaller or less salient subjects (people, objects).
2. **Hallucination** — the model can generate plausible but factually wrong objects, likely by completing common visual patterns seen in training (stop sign → doorframe → door).
3. **Counting** — repeated instances of the same object are systematically undercounted or collapsed into a singular description.
4. **Generally fluent** — even when wrong, captions are grammatically correct and describe *something* real in the image. Failures are errors of omission or association, not random nonsense.
