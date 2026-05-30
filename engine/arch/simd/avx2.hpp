#pragma once

#include "../arch.hpp"

#if defined(TARGET_X86_AVX2)

namespace simd {
	ivec setzero_ivec() {
		return _mm256_setzero_si256();
	}

	fvec setzero_fvec() {
		return _mm256_setzero_ps();
	}

	ivec broadcast_i16(int16_t x) {
		return _mm256_set1_epi16(x);
	}

	ivec broadcast_i32(int32_t x) {
		return _mm256_set1_epi32(x);
	}

	fvec broadcast_f32(float x) {
		return _mm256_set1_ps(x);
	}

	ivec load_ivec(const ivec *p) {
		return _mm256_loadu_si256(p);
	}

	fvec load_fvec(const float *p) {
		return _mm256_loadu_ps(p);
	}

	ivec clamp_i16(ivec x, ivec lo, ivec hi) {
		x = _mm256_max_epi16(x, lo);
		x = _mm256_min_epi16(x, hi);
		return x;
	}

	fvec clamp_f32(fvec x, fvec lo, fvec hi) {
		x = _mm256_max_ps(x, lo);
		x = _mm256_min_ps(x, hi);
		return x;
	}

	ivec shift_mulhi(ivec a, ivec b) {
		a = _mm256_slli_epi16(a, 7);
		return _mm256_mulhrs_epi16(a, b);
	}

	ivec dpbusdx2(ivec sum, ivec u0, ivec i0, ivec u1, ivec i1) {
		ivec prod0 = _mm256_maddubs_epi16(u0, i0);
		ivec prod1 = _mm256_maddubs_epi16(u1, i1);
		ivec prod = _mm256_madd_epi16(_mm256_add_epi16(prod0, prod1), _mm256_set1_epi16(1));
		return _mm256_add_epi32(sum, prod);
	}

	ivec dpbusd(ivec sum, ivec u, ivec i) {
		ivec prod = _mm256_maddubs_epi16(u, i);
		prod = _mm256_madd_epi16(prod, _mm256_set1_epi16(1));
		return _mm256_add_epi32(sum, prod);
	}

	ivec add_i32(ivec a, ivec b) {
		return _mm256_add_epi32(a, b);
	}

	uint32_t nonzero_mask(ivec v) {
		ivec is_zero = _mm256_cmpeq_epi32(v, _mm256_setzero_si256());
		return (~_mm256_movemask_ps(_mm256_castsi256_ps(is_zero))) & 0xFF;
	}

	fvec cvt_i32_f32(ivec v) {
		return _mm256_cvtepi32_ps(v);
	}

	fvec fma_f32(fvec a, fvec b, fvec c) {
		return _mm256_fmadd_ps(a, b, c);
	}

	fvec mul_f32(fvec a, fvec b) {
		return _mm256_mul_ps(a, b);
	}

	fvec add_f32(fvec a, fvec b) {
		return _mm256_add_ps(a, b);
	}

	void store_f32(float *p, fvec v) {
		_mm256_storeu_ps(p, v);
	}

	void store_i32(int32_t *p, ivec v) {
		_mm256_storeu_si256((ivec *)p, v);
	}

	void store_u16_u8(uint8_t *p, ivec v) {
		__m256i res = _mm256_packus_epi16(v, v);
		res = _mm256_permute4x64_epi64(res, _MM_PERM_DCCA);

		_mm_storeu_si128((__m128i *)p, _mm256_castsi256_si128(res));
	}

	float reduce_add_ps(fvec v) {
		__m128 sum = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
		sum = _mm_add_ps(sum, _mm_movehdup_ps(sum));
		sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));

		return _mm_cvtss_f32(sum);
	}
}

#endif
