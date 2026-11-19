#pragma once

// packed rows and kernel tables of the coarse K/V copy for ggml_kvq_attn (layout: see ggml.h)
//
// the components of a table are stored in packed order: bits descending, ties by codec component
// the transform basis is arbitrary, so the permutation is applied to the tables and costs nothing at run time

#include "llama-kvq-codec.h"

#include <cstdint>
#include <vector>

// perm[c] = codec component at packed position c
LLAMA_API std::vector<int32_t> llama_kvq_pack_perm(const llama_kvq_table & t);

// words of the codes of a packed row, ceil(sum(bits)/32); a V row in vc has one more word for the gain
LLAMA_API int32_t llama_kvq_pack_words(const llama_kvq_table & t);

// one table in the layout of ggml_kvq_attn: inv [t.n*t.n] i8 (one i2 slice of tab_inv), aux [3*t.n] f32 (one i2 slice of tab_aux)
// row inv[c][:] is stored as int8 with one scale per component, the stored step holds the scale
LLAMA_API void llama_kvq_pack_table(const llama_kvq_table & t, int8_t * inv, float * aux);
