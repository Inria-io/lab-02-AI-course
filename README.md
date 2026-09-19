# Lab 02 — Open the box

Week 2 · Inside the model · Narxoz University

Lecture 3 made claims about what happens inside a language model. You will check three of them on a real
model, with your own hands: that the output is a list of 50,257 probabilities and not a word; that temperature
reshapes that list without being able to change which token is first; and that attention is a table of weights
whose every row sums to one. The model is GPT-2 small (124 million parameters). It is weak on purpose — you can
see what it does wrong.

**No API key. No installation. No cost.**

## Setup

Open the notebook in Google Colab — one click:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unreal-kz/lab-02-AI-course/blob/main/lab02_inside_the_model.ipynb)

Run **only the first code cell** and let it finish. It downloads the model (about 0.5 GB, fetched by Google's
servers, not by your laptop). Do **not** use *Runtime → Run all*: the lab's rule is that you write a prediction
down *before* you see the answer, and Run all shows you every answer first.

Colab is the only environment this lab was prepared for. No GPU is needed, and the first cell installs
anything Colab happens to be missing.

## Two departures from Lab 01

Both are deliberate; a later edit should not "fix" them.

- **Hugging Face is used here, and Lab 01 rejected it.** Lab 01 ran locally, where a gated licence or a
  multi-hundred-megabyte download would stall a classroom. `openai-community/gpt2` is not gated, and in Colab the
  download happens on Google's machine.
- **A notebook instead of `.py` scripts.** An attention heatmap needs inline plots, and a local `torch`
  environment on every laptop would not come up inside a 50-minute class.

## Part 0 — a token is not a word

GPT-2's tokenizer is the seventh in the lecture's comparison, and the smallest (50,257 entries). Predict how
many tokens `бөлімшеңізде` takes, then measure it, and compare `bank` with `банк`.

## Part 1 — the output is 50,257 numbers

The next-token distribution for one prompt: every token's probability, the top ten, and how much of the total
they hold. Then a temperature sweep (0.25, 1, 2, 5) with the entropy of each distribution, then top-p, then ten
samples from each setting. The difference between "reshape" and "delete" has to come from your own numbers.

## Part 2 — attention is a table of weights

Twelve layers times twelve heads is 144 tables per prompt. You get a map of all 144 (which heads look at the
previous token, which look at token 0), pick one head, and read its heatmap. Row = the token doing the looking,
column = the token being looked at.

## Part 3 — the same model, in Kazakh

Tokens per character in English, Russian and Kazakh; the model's most likely next token for a Russian and a
Kazakh prompt; and the slide-14 loop written out in six lines — pick the top token, append it, repeat.

## What you hand in

Download the notebook (*File → Download → .ipynb*) **after** you have run every cell and filled in every ✏️,
and upload it to Canvas.

1. The four predictions (Part 0, Part 1, temperature, Part 3), written before you ran the cell, with the
   measured value next to each.
2. One sentence on temperature and one on top-p, in your own words.
3. The (layer, head) you found for the previous-token pattern, and the (layer, head) of the most extreme
   "token 0" head.
4. Your Part 3 answer.

## Extension tasks

Optional. Each task names the exact edit, what to hand in, what to expect, and the mistake that produces a
plausible but wrong conclusion. Every number under "Expect" comes from the reference run, not from memory. It
was produced with `transformers` 5.17.0 and repeated on 4.57.3 with identical output, on one machine. On
Colab the last digit of a probability may differ; the ranking and the order of magnitude should not.

### Core

**1. Flip the answer.**
Do: in `PROMPT`, replace `Astana` with `Berlin`, then with `Rome`. Re-run `show_next_token(PROMPT)` each time.
Hand in: the #1 token and its probability for all three cities.
Expect: with `Astana` the #1 token is ` Ast` (22.87%) and ` Paris` is second (22.08%). With `Berlin` it is
` Paris` (41.78%) and ` Berlin` gets 8.53%. With `Rome` it is ` Paris` (49.73%) and ` Rome` gets 3.67%.
Trap: "the model just copies what the context says." If it copied, ` Berlin` would lead in the second run,
and it does not. Nothing in this lab shows *why* ` Ast` wins the first prompt; a 0.8-point gap is a near-tie,
and a near-tie flips easily.

**2. Is the previous-token head stable?**
Do: set `TEXT` to three other texts — one Russian, one code, one English prose — and re-run the Part 2 cells.
Hand in: for each text, the head with the highest previous-token score, and the head with the highest
token-0 score.
Expect: layer 4, head 11 wins the previous-token score every time — 1.000 on
`The quick brown fox jumps over the lazy dog because the dog was sleeping`, 0.994 on
`Столица Казахстана — Астана. Столица Франции — Париж.`, 0.939 on `def add(a, b):` + newline +
`    return a + b`. The most extreme token-0 head changes with the text: (5, 1) at 0.991 on the fox sentence,
(9, 1) at 0.988 on the Russian one, (7, 2) at 0.985 on the code, and (7, 10) at 0.964 on the lab's own prompt.
Trap: reporting one head as "the" attention sink. The phenomenon is stable; which head shows it most is not.

**3. Locate the Kazakh premium, at letter level.**
Do: in a new cell, run `len(tok.encode(c))` for each of the nine Kazakh-only letters `ә ғ қ ң ө ұ ү һ і`, for the
33 lowercase Russian letters, and for the same 33 in uppercase.
Hand in: how many letters cost one token and how many cost two, in each group.
Expect: all 9 Kazakh-only letters cost 2 tokens. Of the 33 lowercase Russian letters, 17 cost 1 token and
16 cost 2. All 33 uppercase Russian letters cost 2.
Trap: settling "is Kazakh worse than Russian on GPT-2?" from one sentence pair. In Part 3 the totals say yes
(34 tokens against 31, +10%); per character they say no (1.10 against 1.11), because the Kazakh sentence is
simply longer (31 characters against 28). Both sentences are translations written for this lab, not a controlled
pair. Say which measure you use, and compare with lecture 2's ranges (Kazakh 1.68–4.83× English, Russian
1.23–2.92×, across six tokenizers) before you claim anything. Where the cost comes from is the letters:
Kazakh-only letters have no merge in this table, so each is two byte-tokens.

### Advanced

**4. Attention is not an explanation.**
Do: for the three prompts of task 1, print how much attention the last token gives to the city token
(position 5 in all three) in layer 5, heads 0 and 1: `atts[5][0, 0, -1, 5]` and `atts[5][0, 1, -1, 5]`.
Hand in: the six numbers, and one paragraph on what you can and cannot conclude from them.
Expect: layer 5, head 0 — 0.564 (Astana), 0.490 (Berlin), 0.396 (Rome). Layer 5, head 1 — 0.515, 0.303, 0.205.
Trap: "the heads look at ` Ast` more when the answer is wrong, so attention caused the wrong answer." Head 0
looks at the city token in all three prompts, including both where the answer is right, and three prompts
cannot separate cause from coincidence. Testing a cause needs an intervention — switch a head off and see
whether the answer changes — which this lab does not do.

## Files

| File | What it is |
|---|---|
| `lab02_inside_the_model.ipynb` | The lab: 27 cells, Parts 0–3, no outputs saved |
| `README.md` | This file |
