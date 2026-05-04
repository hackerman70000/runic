# Method reference

## Soft watermark — Kirchenbauer Algorithm 2

For each generation step `t`:

1. Forward pass produces logits `l^(t)` over the vocabulary `V`.
2. Hash the previous token `s^(t-1)` together with the secret `hash_key`
   to seed a PRNG.
3. The PRNG produces a deterministic random permutation of `V`. The
   first `floor(gamma * |V|)` indices form the green list `G`; the rest
   form the red list `R`.
4. Add a constant bias `delta` to logits of all `k in G`:

   ```
   l_k^(t) <- l_k^(t) + delta · 1[k in G]
   ```

5. Apply softmax to the modified logits and sample `s^(t)` as usual.

When entropy is high (the model is unsure), the bias significantly
shifts probability mass toward the green list and a green token is
sampled. When entropy is low (the model is nearly deterministic), the
bias has little effect and quality is preserved — but the watermark
also leaves no signal on those tokens.

## Hard variant — Algorithm 1

`delta = inf` corresponds to the hard variant: red tokens are masked to
`-inf` and never sampled. Easier to analyse but quality drops on
low-entropy text.

## Detection (model-free)

A verifier tokenises the candidate text and recomputes the green list
for each transition `s^(t-1) -> s^(t)` using the same hash. Let
`|s|_G` be the number of green hits and `T = len(tokens) - 1`.

One-proportion z-statistic:

```
z = (|s|_G - gamma * T) / sqrt(T * gamma * (1 - gamma))
```

H_0: tokens are produced without knowledge of the green list, so
`E[|s|_G] = gamma * T`. Reject H_0 when `z >= z_threshold` (paper uses
4.0, p ≈ 3e-5). Detection requires only the tokenizer and the
`hash_key`.

## Spike entropy

The soft watermark only marks high-entropy tokens. The paper introduces
the *spike entropy with modulus z*:

```
S(p, z) = sum_k  p_k / (1 + z * p_k)
```

Theorem 4.2: a watermarked sequence with average spike entropy at least
`S*` produces at least

```
E[|s|_G] >= gamma * alpha * T / (1 + (alpha - 1) * gamma)
```

green tokens, where `alpha = exp(delta)`.

For `gamma = 0.5`, `delta = ln(2) ≈ 0.7` this simplifies to
`E[|s|_G] >= (2/3) * T * S*`.

## Empirical settings

Paper headline numbers (OPT-1.3B, C4 RealNewsLike, T = 200 tokens, 500
generations):

| sampling     | gamma | delta | TPR @ z=4 | FPR  |
|--------------|-------|-------|-----------|------|
| multinomial  | 0.25  | 2.0   | 99.4%     | 0%   |
| 8-beam       | 0.25  | 2.0   | 99.4%     | 0%   |
| 8-beam       | 0.25  | 5.0   | 100.0%    | 0%   |

## T5 span-replacement attack (paper §7.1)

Adversary masks single words and asks T5 to refill until a fraction
`epsilon` of tokens is replaced. `epsilon = 0.1` barely dents detection
(AUC drop ≈ 0.01). `epsilon = 0.5` reduces AUC to ~0.84 but adds 22 PPL
points — usually unacceptable to the attacker.

## References

* Kirchenbauer et al. *A Watermark for Large Language Models.* ICML 2023.
