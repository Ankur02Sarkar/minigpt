# Phase 4 — Generation Samples

Verbatim, unedited outputs from the final Phase 4 checkpoints. Generated on the T4 VM
(CPU, FP32) on **2026-08-15** with `scripts/generate_samples.py`, seed 42,
`--max-new-tokens 128`. Raw captures: `logs/phase4-artifacts/` (gitignored).

**Reading guide:** these samples corroborate the eval metrics in
[`EVAL.md`](EVAL.md) — topical vocabulary in broken word order, plus repetition-loop
collapse under greedy decoding. `ï¿½` is a byte-level BPE fallback artifact (malformed
UTF-8 bytes decoded), itself a diagnostic signal for task 4.8.

---

## minigpt-low — `checkpoints/minigpt-low/best.pt` (step 1,000 · best_val_loss 9.599)

Trained on TinyStories only. Note: `best.pt` is the step-1,000 checkpoint because the
WikiText val loss worsened as TinyStories training progressed (see EVAL.md).

### Sampling (temperature 0.8, top-k 50)

**Prompt:** `Once upon a time` (+128 tokens)

> Once upon a time there a girl Sam was and little. had special and was to at. loved play his with toys play One, went the to park play He a for walk the. saw big, tree the, the was a girl Sue She scared help. asked mom it time go the. told mom she help, they a dog a. boy the girl Mia her said ", are. you't to. are so when am to with. is and like play with." was excited asked " want help wait play you the?"I the said ",. can have toys Lily. I help see and, can go the. has very. we

**Prompt:** `One day, a little girl named Lily` (+109 tokens, hit `</s>`)

> One day, a little girl named Lily home a toy The had idea Lily she't to her and mom with friends She some to a on floor the., went the and her had great, the. looked the and, saw big on floor Lily hole Lily. looked Tom and, was. wanted go the, Lily mom said ",,'s.'s!" didn mean be, Lily Max her. wanted learn be. she't to Lily, they at the and for. saw's saw friend Lily Lily her and,,, said ", is for. love too Lily</s>

**Prompt:** `In a small village, there lived a brave knight` (+123 tokens, hit `</s>`)

> In a small village, there lived a brave knight The was, would up the and to the and them with best. day a girl the was to the and little and little came to. he for walk the and the, little called "! am three old!" the said. bird a girl ",," the said. little looked at and's said "'s,. I have big and will you.'m happy you me The is." little and mom the girl amazed They on way and a smile They back her and the girl the. little thanked kind for and the girl the. little smiled she, she her and, knew would have to the girl</s>

### Greedy (temperature 0)

**Prompt:** `Once upon a time` (+128 tokens)

> Once upon a time there a girl Lily She to park her. day she to park her and a girl a. day she to park her and a. saw big and, was. wanted play the, she a girl Lily She her and, she to her. asked mom ", you a!" mom. mom and mom the girl the girl the. was happy see and. said ", are a girl Lily I a. you to with and. you to with and. you to with and. you to with and. you to with and. you to with and. you to with and. you to with and. you to with and. you

---

## minigpt-high — `checkpoints/minigpt-high/best.pt` (step 100,000 · best_val_loss 4.714)

Trained on TinyStories + WikiText-103 + FineWeb-Edu (500M).

### Sampling (temperature 0.8, top-k 50)

**Prompt:** `Once upon a time` (+128 tokens)

> Once upon a time there an farmer a egg It a egg a, one a, a one The was very and very. farmer the was and wanted organize egg all. put egg the in box Then egg to egg egg hen egg egg egg egg egg egg egg egg egg. egg egg egg egg egg egg egg The egg egg
>  egg egg egg hen egg egg egg egg the egg egg: egg egg In egg egg egg egg The egg egg have egg egg egg When egg egg the egg the egg I egg The egg egg the egg we as egg egg egg "" egg egg for and you egg egg The egg of egg egg She egg. egg egg egg The

**Prompt:** `The history of the Roman Empire` (+128 tokens)

> The history of the Roman Empireï¿½ history not by of or the of- but by and of history as as in other of Roman.The of, the of a which in had or been as been, a which been a and and is as the of. Roman and Roman. is by, the of;, as in; the; in places in or the;; in in ways than or, the; in and. Romans more, in and, in ways or of, in.. in are any.In or, in,,; in
>  in or in or in, which,, in,. in, in of
>  as.,,

**Prompt:** `Scientists have discovered that the brain` (+128 tokens)

> Scientists have discovered that the brain a of with, a with, a,,,,,,, and are,,,, people We know this of,,,, and,,, other. is for of..Ln at1 The of ()
> : the we of? the
> . it is the that the which be,The of ( or,).,, be (
> rom,,),,.,,,.The, of or,,, you know
>  the (),,,,.,, are.,,,,,,,, . ,,..(,),.

### Greedy (temperature 0)

**Prompt:** `Once upon a time` (+128 tokens)

> Once upon a time there a girl Lucy a girl Lucy a girl a. was years and was years. day Lucy to to to. was three old Lucy to. was excited go the day. Lucy to park her. Lucy to park She to park She to park She to park to. Lucy to park to park to. Lucy to park to to to to to to to to to to to to. Lucy to to to to to. Lucy to to to to to. Lucy to to to to to. Lucy to to to to to. Lucy to to to to. Lucy to to to to. Lucy to to to to. Lucy to to. Lucy

**Prompt:** `The history of the Roman Empire` (+128 tokens)

> The history of the Roman Empire the of Roman was by the of Roman. Roman was by of importance the of Roman,, the of, the,,,,,,,,,,,,,,,,,, people the of,,,,,,,, people of,,, people,, people,,.,,,.,,,.,.,,,..,,,...,,,.. is a of,,,..,,,..,,,. is a of,,,..,,,. is of,,...,,

---

## Reproduce

```bash
# on azure-train (VM must be started first)
cd /opt/minigpt_llm && source .venv/bin/activate
python -m scripts.generate_samples \
  --checkpoint checkpoints/minigpt-high/best.pt \
  --tokenizer-dir tokenizer \
  --max-new-tokens 128 --temperature 0.8 --top-k 50 \
  --prompts "Once upon a time"
```
