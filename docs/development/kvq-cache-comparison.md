# Two-tier KV cache against plain f16, q8_0 and q4_0 caches (Qwen3.8-27B UD-IQ3_S target, DFlash2 drafter), 2026-11-17

Machine: RTX 5090 32 GB, Ryzen 9 9950X3D, Windows 11, CUDA 12.9.
One binary for all four caches: ours (JetBrains-Hardware, `kvq-two-tier-cache`) at 2758a0a64. The cache design is in `kvq-two-tier-cache.md`, the launch line and the memory harness in `dflash2-vram-baseline.md`. Raw tables and logs are outside the repo in `../kvq-scratch/mem/comb` (`final-compare.txt`).

| Name | Cache |
|---|---|
| A | plain f16 K/V (the reference of every part) |
| B | plain q8_0 K/V, default attention rotation, as the server runs it |
| C | `--kvq` with defaults: q8_0 precise tier in host RAM, 8192 resident slots, whole union of the verify batch, precise rows in blocks of 256 |
| D | plain q4_0 K/V, default attention rotation |
| Bn | plain q8_0 without the rotation; only an extra reference for C, because the earlier kvq numbers were measured against this teacher |

Contexts: 24k (24,183-token prompt), 50k (49,737), 100k (99,395).

## Result

- Against f16, `--kvq` is as close as plain q8_0 (the intervals overlap at all three contexts) and q4_0 is 7 - 10x further.
- `--kvq` needs 4,221 MiB less VRAM than f16 and 1,543 MiB less than q8_0 in generation at n_ctx 102400, the same as q4_0 (57 MiB more).
- f16 is the fastest cache at every context. `--kvq` is the slowest of the four at 24k and 50k, and faster than q8_0 and q4_0 at 100k.

## 1. Quality

kvq-eval, IQ3_S target, teacher forced on the recorded reply of each prompt, `-b 2048 -ub 2048`. The teacher is A at one token per decode. KL(teacher || cache) with the 95% bootstrap interval, then top-1 flips out of the scored positions.

### 24k (245 positions)

| Cache | Against A at 1 per decode | Against Bn at 1 per decode |
|---|---|---|
| A f16, 8 per batch (floor) | 6.20e-5 [4.28e-5, 8.45e-5], 0 | |
| B q8_0, 1 per decode | 1.30e-4 [7.84e-5, 1.91e-4], 0 | |
| B q8_0, 8 per batch | 9.34e-5 [6.51e-5, 1.25e-4], 0 | |
| Bn q8_0 no rotation, 1 per decode | 1.02e-4 [6.70e-5, 1.41e-4], 0 | |
| Bn q8_0 no rotation, 8 per batch | 1.14e-4 [7.56e-5, 1.57e-4], 0 | 4.81e-5 [3.21e-5, 6.64e-5], 0 |
| C --kvq, 1 per decode | 1.07e-4 [7.04e-5, 1.50e-4], 0 | 4.53e-5 [3.04e-5, 6.31e-5], 0 |
| C --kvq, 8 per batch | 1.07e-4 [7.27e-5, 1.45e-4], 0 | 3.70e-5 [2.55e-5, 5.12e-5], 0 |
| D q4_0, 1 per decode | 8.91e-4 [5.87e-4, 1.27e-3], 2 | |
| D q4_0, 8 per batch | 1.02e-3 [6.71e-4, 1.48e-3], 2 | |

### 50k (289 positions)

| Cache | Against A at 1 per decode | Against Bn at 1 per decode |
|---|---|---|
| A f16, 8 per batch (floor) | 6.48e-5 [4.30e-5, 9.11e-5], 1 | |
| B q8_0, 1 per decode | 1.77e-4 [1.26e-4, 2.43e-4], 2 | |
| B q8_0, 8 per batch | 1.84e-4 [1.29e-4, 2.45e-4], 3 | |
| Bn q8_0 no rotation, 1 per decode | 1.93e-4 [1.31e-4, 2.68e-4], 4 | |
| Bn q8_0 no rotation, 8 per batch | 1.85e-4 [1.33e-4, 2.41e-4], 2 | 6.29e-5 [4.69e-5, 8.03e-5], 2 |
| C --kvq, 1 per decode | 2.30e-4 [1.68e-4, 2.96e-4], 3 | 1.17e-4 [7.54e-5, 1.67e-4], 1 |
| C --kvq, 8 per batch | 2.27e-4 [1.60e-4, 2.98e-4], 3 | 1.09e-4 [7.91e-5, 1.42e-4], 1 |
| D q4_0, 1 per decode | 1.73e-3 [1.18e-3, 2.36e-3], 4 | |
| D q4_0, 8 per batch | 1.81e-3 [1.27e-3, 2.41e-3], 4 | |

### 100k (471 positions)

| Cache | Against A at 1 per decode | Against Bn at 1 per decode |
|---|---|---|
| A f16, 8 per batch (floor) | 6.74e-5 [5.24e-5, 8.37e-5], 0 | |
| B q8_0, 1 per decode | 1.77e-4 [1.08e-4, 3.06e-4], 1 | |
| B q8_0, 8 per batch | 1.78e-4 [1.03e-4, 3.20e-4], 1 | |
| Bn q8_0 no rotation, 1 per decode | 1.22e-4 [9.34e-5, 1.56e-4], 1 | |
| Bn q8_0 no rotation, 8 per batch | 1.13e-4 [8.49e-5, 1.49e-4], 2 | 6.49e-5 [5.04e-5, 8.08e-5], 1 |
| C --kvq, 1 per decode | 1.23e-4 [9.66e-5, 1.51e-4], 1 | 8.14e-5 [6.27e-5, 1.03e-4], 0 |
| C --kvq, 8 per batch | 1.34e-4 [1.07e-4, 1.64e-4], 1 | 8.86e-5 [6.91e-5, 1.11e-4], 0 |
| D q4_0, 1 per decode | 1.01e-3 [7.45e-4, 1.32e-3], 2 | |
| D q4_0, 8 per batch | 1.02e-3 [7.77e-4, 1.34e-3], 2 | |

Notes:

- The floor (f16 at 8 per batch against f16 at 1 per decode) is 6.2 - 6.7e-5, so every KL below about 1e-4 is near the rounding floor of the model.
- C at 50k is the weakest cell of C: 2.30e-4 against 1.77e-4 for B (the intervals overlap), and 1.17e-4 against Bn, higher than at 24k and 100k.
- B with the rotation is further from f16 than Bn at 24k and 100k and slightly closer at 50k; at 100k it has a few large positions (max KL 2.1e-2 against 2.6e-3). The intervals overlap.

## 2. Speed

llama-server with the DFlash2 drafter (incoai Q4_K_M), the launch line of `dflash2-vram-baseline.md`, n_ctx 102400, real prefill, 300 tokens, two sessions each. The prompt is the prefix + 3 tokens (ends with the assistant header). Times from `LLAMA_SERVER_GEN_TIMES`. n_max 7 gives verify batches of 8 tokens, n_max 3 of 4 tokens.

Decode, ms per verify batch (mean of the two sessions; they differ by at most 0.2 ms). B, C and D are differences to A.

| Context | n_max | A f16 | B q8_0 | C --kvq | D q4_0 |
|---|---|---|---|---|---|
| 24k | 7 | 16.57 | +1.16 | +4.61 | +1.56 |
| 24k | 3 | 14.01 | +1.03 | +3.38 | +1.48 |
| 50k | 7 | 17.53 | +3.04 | +5.03 | +3.14 |
| 50k | 3 | 15.05 | +2.89 | +3.09 | +3.17 |
| 100k | 7 | 19.64 | +6.68 | +6.03 | +6.39 |
| 100k | 3 | 17.17 | +6.45 | +2.81 | +6.36 |

Prefill, seconds (n_max 7 sessions; the n_max 3 sessions are within 0.05 s).

| Context | A f16 | B q8_0 | C --kvq | D q4_0 |
|---|---|---|---|---|
| 24k | 6.53 | 6.53 | 7.36 | 6.51 |
| 50k | 15.09 | 15.27 | 16.92 | 15.24 |
| 100k | 40.00 | 42.24 | 43.53 | 42.22 |

Tokens per second and the reply of each cell (generated tokens, drafted / accepted; the two sessions of a cell give the same counts).

| Context | n_max | A f16 | B q8_0 | C --kvq | D q4_0 |
|---|---|---|---|---|---|
| 24k | 7 | 224.4 t/s, 124, 168 / 99 | 206.3, 124, 175 / 98 | 186.4, 124, 182 / 97 | 203.2, 124, 175 / 98 |
| 24k | 3 | 157.4, 134, 135 / 88 | 174.3, 124, 105 / 88 | 161.6, 124, 108 / 87 | 162.5, 124, 111 / 86 |
| 50k | 7 | 156.5, 298, 606 / 210 | 166.9, 270, 448 / 206 | 130.6, 287, 581 / 203 | 144.2, 276, 532 / 199 |
| 50k | 3 | 151.1, 298, 309 / 195 | 130.2, 298, 312 / 194 | 135.8, 262, 264 / 173 | 124.8, 276, 297 / 177 |
| 100k | 7 | 161.3, 191, 336 / 142 | 122.5, 192, 350 / 141 | 150.0, 300, 463 / 232 | 139.4, 300, 489 / 229 |
| 100k | 3 | 154.2, 300, 271 / 208 | 104.7, 192, 195 / 126 | 141.8, 300, 265 / 210 | 114.5, 300, 280 / 205 |

Notes:

- The caches give different replies in most cells (other rounding; only B and D at 24k with n_max 7 have the same counts), so the t/s of a cell follows its reply and is not a comparison of the caches. The ms per verify batch is the comparable number. The acceptance rate of the caches was not measured here (it needs the 12-request set of `dflash2-vram-baseline.md`).
- f16 is the fastest at every context and batch size: it has no dequantize before the flash attention. q8_0 and q4_0 cost about the same (q4_0 0.1 - 0.5 ms more at 24k and 50k) and grow with the context.
- C has a cost per round that depends little on the context (+4.6 ms at 24k, +6.0 ms at 100k with batches of 8), so it is the slowest at 24k and 50k and ahead of q8_0 and q4_0 at 100k.

## 3. Memory

Dedicated usage of the process (`mem\dfl\memsample.ps1`), MiB. The idle device baseline (desktop, 721 MiB that day) is not included.

| | A f16 | B q8_0 | C --kvq | D q4_0 |
|---|---|---|---|---|
| n_ctx 102400, 100k prompt: after load | 21,838.6 | 19,160.6 | 17,561.8 | 17,560.5 |
| n_ctx 102400, 100k prompt: prefill peak | 21,974.6 | 19,296.9 | 17,726.0 | 17,696.5 |
| n_ctx 102400, 100k prompt: generation | 22,022.6 | 19,344.6 | 17,802.0 | 17,744.6 |
| Generation against A | 0 | -2,678.0 | -4,220.6 | -4,278.0 |
| Generation against B | +2,678.0 | 0 | -1,542.6 | -1,600.0 |
| n_ctx 51200, load only: after load | 18,538.4 | 17,160.3 | 16,788.3 | 16,360.3 |
| n_ctx 204800, load only: after load | 28,439.0 | 23,161.0 | 19,106.9 | 19,961.0 |
| Per 1000 positions of n_ctx (51200 to 204800) | 64.46 | 39.06 | 15.10 | 23.44 |
| Extrapolated to n_ctx 0 (computed from the two load-only rows) | 15,238 | 15,160 | 16,015 | 15,160 |

Notes:

- C pins the q8_0 store in host RAM: shared usage 4,266 MiB at n_ctx 102400 and 7,778 MiB at 204800 after load (A, B and D: 470 - 670 MiB).
- The part of C that does not depend on n_ctx is about 855 MiB above the plain caches (ring, resident slots, staging buffer of the union, tables). With the slopes above, C needs less VRAM than q8_0 from about 36k positions of n_ctx, less than f16 from about 16k and less than q4_0 from about 102k. These three figures are computed from the load-only rows, not measured.
- C against D, the other way to shrink the cache: the same VRAM at n_ctx 102400, 854 MiB less at 204800, KL against f16 7 - 8x lower.
