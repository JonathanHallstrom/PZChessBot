#pragma once

#include "../arch.hpp"

// Define constants for each target
#if defined(TARGET_ARM_NEON)

using ivec = uint8x16_t;
using fvec = float32x4_t;
#define VEC_SIZE 128

#define L1_UNROLL 4
#define L2_UNROLL 4
#define L3_UNROLL 2

#elif defined(TARGET_X86_AVX512)

#include <immintrin.h>

using ivec = __m512i;
using fvec = __m512;
#define VEC_SIZE 512

#define L1_UNROLL 4
#define L2_UNROLL 2
#define L3_UNROLL 1

#elif defined(TARGET_X86_AVX2)

#include <immintrin.h>

using ivec = __m256i;
using fvec = __m256;
#define VEC_SIZE 256

#define L1_UNROLL 4
#define L2_UNROLL 4
#define L3_UNROLL 2

#else
#error "Unsupported architecture"
#endif

#define BYTES_PER_VEC (VEC_SIZE / 8)
#define SHORTS_PER_VEC (VEC_SIZE / 16)
#define FLOATS_PER_VEC (VEC_SIZE / 32)

namespace simd {
	ivec setzero_ivec();
	fvec setzero_fvec();

	ivec broadcast_i16(int16_t x);
	ivec broadcast_i32(int32_t x);
	fvec broadcast_f32(float x);

	ivec load_ivec(const ivec *p);
	fvec load_fvec(const float *p);

	ivec clamp_i16(ivec x, ivec lo, ivec hi);
	fvec clamp_f32(fvec x, fvec lo, fvec hi);

	ivec shift_mulhi(ivec a, ivec b);

	ivec dpbusd(ivec sum, ivec u, ivec i);
	ivec add_i32(ivec a, ivec b);

	fvec cvt_i32_f32(ivec v);

	fvec fma_f32(fvec a, fvec b, fvec c);
	fvec mul_f32(fvec a, fvec b);
	fvec add_f32(fvec a, fvec b);

	void store_f32(float *p, fvec v);
	void store_i32(int32_t *p, ivec v);

	void store_u16_u8(uint8_t *p, ivec v);
	float reduce_add_ps(fvec v);
};
