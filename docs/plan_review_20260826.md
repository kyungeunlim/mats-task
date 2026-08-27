# Review generated against plan.md
2026-08-26. Asked Claude in a new chat, with no prior context, for brutal feedback:
what would a skeptical mech interp reviewer say is wrong or missing. The text below
is the response, unedited.

Responses are marked [rev] in plan.md. Items not addressed there are carried into
the write-up's limitations (T9).

**1. The probe target is the whole experiment and it's deferred to a 30-minute ticket.**

The candidate target, correct versus distractor at the answer position, is close to the worst possible choice for this comparison. Circuit-breaker-style methods work by rerouting representations at the point where the harmful output would be produced. The answer position is exactly where the intervention is designed to bite. So a probe there is measuring the thing that was explicitly trained away, and a null result on CB would be the expected outcome of either suppression or removal. The interesting positions are earlier: the question and context tokens where the relevant knowledge would be retrieved, before the rerouting fires. If the residual at position of the pathogen name carries target-specific information in CB but not in the filtered model, that is a claim about retained knowledge. At the answer position it is closer to a restatement of the logits.

Related: probing only items the base model answers correctly conditions on behavior and induces selection. Probing all items dilutes the base curve. Neither choice is stated.

**2. The error bars measure the wrong thing, and the plan says so without fixing it.**

Bootstrap over prompts estimates eval-set sampling error. The quantity that matters for "is this gap real" is training-run variance, and there is no seed pair anywhere in the design. A reviewer will note that base, filtered, and CB differ in three different ways at once, and that any CB-versus-filtered gap smaller than typical seed-to-seed spread is uninterpretable. Naming the substitution in the limitations does not rescue the inference. One cheap partial fix is to bootstrap over probe training subsamples as well, which at least separates probe-fit variance from prompt variance.

**3. The expected result is nearly guaranteed a priori and does not mean what the framing implies.**

CB was fine-tuned from base. The filtered model never saw the data. Of course CB retains more linearly decodable structure. That follows from initialization plus a light-touch fine-tune, and it does not require the knowledge to be functionally accessible. Without a recovery arm, the strongest honest claim from this design is "the residual stream at layer L retains linearly decodable target information in CB and not in filtered," which is a fact about decodability, not about suppression. The quantization-recovery test from the scoping work would be the cheap thing that makes the internal number mean something, and it is absent here.

**4. TransformerLens will silently apply a per-model transform.**

`from_pretrained` defaults to `fold_ln=True`, plus weight and unembed centering. Folding layer norm is a transform parameterized by *that checkpoint's* LN weights, so it is not the same transform across the three models. That could manufacture or erase differences in the residual stream you are comparing. `from_pretrained_no_processing`, or at minimum logging the flags and checking that a fixed prompt gives matching hidden states against the HF forward pass, would close this. Worth doing in T4 before caching anything.

**5. Only one outcome has an interpretation attached.**

Fail-fast 3 covers "base and filtered don't separate." It does not cover the likelier case: base and filtered separate cleanly, CB sits on top of base at every layer, and the result reduces to "a small fine-tune barely moved the residual stream." A reviewer would want all three orderings pre-registered with what each would license.

**6. Analysis decisions made after seeing the data.**

T4 says decide on position-0 exclusion after inspecting norms, on the same data the headline comes from. Small, but it is a forking path, and the exclusion rule could be pre-committed in T2 at no cost. Same for layer selection if any layer gets picked as "the" layer post hoc.

**7. Power is never calculated.**

1076 cloze-compatible items split into train and held-out gives a held-out probe accuracy interval of maybe four or five points. If the CB-versus-filtered gap being hunted is that size, the experiment cannot see it. That calculation costs nothing and belongs before activation caching. Also unspecified: probe regularization strength and how it is chosen. Selecting it on the held-out split is leakage.

**8. MMLU as retain check is doing less work than it appears.**

A few hundred items across 57 subjects is roughly five per subject, so "MMLU near base" is not really testable at that n, and no effect size worth detecting is stated. MMLU also contains college biology and virology, which overlap the forget set. Splitting bio-adjacent subjects out would make the retain claim sharper.

**9. The behavioral anchor for CB is a training-set number.**

The plan says this once, honestly. But it means the framing "on these benchmarks CB looks like filtered" is weaker than it reads. CB was selected to look like filtered on that metric. The internal measurement is then being motivated against a benchmark the checkpoint was optimized against.

**On the budget:** the stated 1.5x factor on new code does not appear to be applied to T4 or T5. First-time TL hook verification on a 7B NeoX plus three model loads in 2h is optimistic, and 1.5h of buffer against roughly 10h of new-code tickets is thin.

The plan structure itself, gates and pre-committed deliverables and a red-team ticket, is the part a reviewer would not complain about.
