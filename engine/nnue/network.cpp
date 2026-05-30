#include "network.hpp"

#include "incbin.h"
#include <bit>
#include <cstdint>

extern "C" {
	INCBIN(network_weights, NNUE_PATH);
}

void Network::load() {
	char *ptr = (char *)gnetwork_weightsData;

	memcpy(accumulator_weights, ptr, sizeof(accumulator_weights));
	ptr += sizeof(accumulator_weights);

	memcpy(accumulator_biases, ptr, sizeof(accumulator_biases));
	ptr += sizeof(accumulator_biases);

	for (int bucket = 0; bucket < NBUCKETS; bucket++)
		for (int chunk = 0; chunk < L1_SIZE / 4; chunk++)
			for (int out = 0; out < L2_SIZE; out++)
				for (int k = 0; k < 4; k++)
					l1_weights[bucket][chunk * L2_SIZE * 4 + out * 4 + k] = ptr[(bucket * L2_SIZE + out) * L1_SIZE + chunk * 4 + k];

	ptr += sizeof(l1_weights);

	memcpy(l1_biases, ptr, sizeof(l1_biases));
	ptr += sizeof(l1_biases);

	for (int i = 0; i < NBUCKETS; i++) {
		for (int j = 0; j < L3_SIZE; j++) {
			for (int k = 0; k < L2_SIZE; k++) {
				memcpy(&l2_weights[i][k][j], ptr, 4);
				ptr += 4;
			}
		}
	}

	memcpy(l2_biases, ptr, sizeof(l2_biases));
	ptr += sizeof(l2_biases);

	memcpy(output_weights, ptr, sizeof(output_weights));
	ptr += sizeof(output_weights);

	memcpy(&output_biases, ptr, sizeof(output_biases));
}

int calculate_index(Square sq, PieceType pt, bool side, bool perspective, int nbucket) {
	if (nbucket & 1) {
		// Flip the square
		sq = (Square)(sq ^ 7);
	}
	nbucket /= 2;
	if (perspective) {
		side = !side;
		sq = (Square)(sq ^ 56);
	}
	return nbucket * INPUT_SIZE + side * 64 * 6 + pt * 64 + sq;
}


using u16x8 = uint16_t __attribute__((vector_size(16)));
static constexpr auto NONZERO_INDICES = [] {
	std::array<std::array<uint16_t, 8>, 256> table{};
	for (int mask = 0; mask < 256; mask++) {
		int n = 0;
		for (int bit = 0; bit < 8; bit++)
			if (mask & (1 << bit))
				table[mask][n++] = bit;
	}
	return std::bit_cast<std::array<u16x8, 256>>(table);
}();

static int findNonZeroIndices(const uint8_t *l1, uint16_t *indices) {
	int count = 0;
	u16x8 base = {};
	for (int i = 0; i < L1_SIZE; i += BYTES_PER_VEC) {
		uint32_t mask = simd::nonzero_mask(simd::load_ivec((const ivec *)&l1[i]));
		for (int b = 0; b < FLOATS_PER_VEC; b += 8) {
			uint8_t byte = (mask >> b) & 0xFF;
			u16x8 actual = NONZERO_INDICES[byte] + base;
			memcpy(&indices[count], &actual, sizeof(actual));
			count += arch::popcnt(byte);
			base += 8;
		}
	}
	return count;
}

int32_t nnue_eval(const Network &net, const Accumulator &stm, const Accumulator &ntm, uint8_t nbucket) {
	const ivec zero = simd::setzero_ivec();
	const ivec clip = simd::broadcast_i16(QA);
	const fvec f_zero = simd::setzero_fvec();
	const fvec f_clip = simd::broadcast_f32(1.0f);
	const fvec div = simd::broadcast_f32(1.0f / (QA * QA * QB / 256.0f));

	alignas(64) uint8_t l1[L1_SIZE];
	alignas(64) int32_t l2i[L2_SIZE];
	alignas(64) float l2[L2_SIZE];
	alignas(64) float l3[L3_SIZE];

	// Pairwise mul
	for (int i = 0; i < L1_SIZE / 2; i += SHORTS_PER_VEC) {
		ivec stm_val1 = simd::load_ivec((ivec *)&stm.val[i]);
		ivec stm_val2 = simd::load_ivec((ivec *)&stm.val[i + L1_SIZE / 2]);
		ivec ntm_val1 = simd::load_ivec((ivec *)&ntm.val[i]);
		ivec ntm_val2 = simd::load_ivec((ivec *)&ntm.val[i + L1_SIZE / 2]);

		stm_val1 = simd::clamp_i16(stm_val1, zero, clip);
		stm_val2 = simd::clamp_i16(stm_val2, zero, clip);

		ntm_val1 = simd::clamp_i16(ntm_val1, zero, clip);
		ntm_val2 = simd::clamp_i16(ntm_val2, zero, clip);

		ivec stm_pair = simd::shift_mulhi(stm_val1, stm_val2);
		ivec ntm_pair = simd::shift_mulhi(ntm_val1, ntm_val2);

		simd::store_u16_u8(&l1[i], stm_pair);
		simd::store_u16_u8(&l1[i + L1_SIZE / 2], ntm_pair);
	}

	constexpr int NUM_OUT_VECS = L2_SIZE / FLOATS_PER_VEC;
	const int8_t *l1_weights = net.l1_weights[nbucket];
	const int32_t *chunks = (const int32_t *)l1;

	uint16_t indices[L1_SIZE / 4];
	int count = findNonZeroIndices(l1, indices);

	ivec l1_sums[NUM_OUT_VECS][L1_UNROLL] = {};

	int i = 0;
	for (; i + 2 * L1_UNROLL <= count; i += 2 * L1_UNROLL) {
		for (int p = 0; p < L1_UNROLL; p++) {
			int c0 = indices[i + 2 * p];
			int c1 = indices[i + 2 * p + 1];
			ivec in0 = simd::broadcast_i32(chunks[c0]);
			ivec in1 = simd::broadcast_i32(chunks[c1]);
			for (int o = 0; o < NUM_OUT_VECS; o++) {
				ivec w0 = simd::load_ivec((const ivec *)&l1_weights[c0 * L2_SIZE * 4 + o * BYTES_PER_VEC]);
				ivec w1 = simd::load_ivec((const ivec *)&l1_weights[c1 * L2_SIZE * 4 + o * BYTES_PER_VEC]);
				l1_sums[o][p] = simd::dpbusdx2(l1_sums[o][p], in0, w0, in1, w1);
			}
		}
	}
	for (; i < count; i++) {
		int c = indices[i];
		ivec in = simd::broadcast_i32(chunks[c]);
		for (int o = 0; o < NUM_OUT_VECS; o++) {
			ivec weight = simd::load_ivec((const ivec *)&l1_weights[c * L2_SIZE * 4 + o * BYTES_PER_VEC]);
			l1_sums[o][0] = simd::dpbusd(l1_sums[o][0], in, weight);
		}
	}

	for (int o = 0; o < NUM_OUT_VECS; o++) {
		ivec acc = l1_sums[o][0];
		for (int p = 1; p < L1_UNROLL; p++)
			acc = simd::add_i32(acc, l1_sums[o][p]);
		simd::store_i32(&l2i[o * FLOATS_PER_VEC], acc);
	}

	// Convert l2 into a proper float array
	for (int i = 0; i < L2_SIZE; i += FLOATS_PER_VEC) {
		ivec i_val = simd::load_ivec((ivec *)&l2i[i]);
		fvec val = simd::cvt_i32_f32(i_val);

		fvec bias = simd::load_fvec(&net.l1_biases[nbucket][i]);

		val = simd::fma_f32(val, div, bias);

		val = simd::clamp_f32(val, f_zero, f_clip);

		val = simd::mul_f32(val, val);

		simd::store_f32(&l2[i], val);
	}

	for (int i = 0; i < L3_SIZE; i += FLOATS_PER_VEC * L2_UNROLL) {
		fvec sums[L2_UNROLL];
		for (int j = 0; j < L2_UNROLL; j++)
			sums[j] = simd::load_fvec(&net.l2_biases[nbucket][i + j * FLOATS_PER_VEC]);

		for (int j = 0; j < L2_SIZE; j++) {
			fvec val = simd::broadcast_f32(l2[j]);

			for (int k = 0; k < L2_UNROLL; k++) {
				fvec weight = simd::load_fvec(&net.l2_weights[nbucket][j][i + k * FLOATS_PER_VEC]);
				sums[k] = simd::fma_f32(val, weight, sums[k]);
			}
		}

		for (int j = 0; j < L2_UNROLL; j++)
			simd::store_f32(&l3[i + j * FLOATS_PER_VEC], sums[j]);
	}

	fvec sums[L3_UNROLL];
	for (int i = 0; i < L3_UNROLL; i++)
		sums[i] = f_zero;

	for (int i = 0; i < L3_SIZE; i += FLOATS_PER_VEC) {
		fvec val = simd::load_fvec(&l3[i]);

		val = simd::clamp_f32(val, f_zero, f_clip);

		fvec weight = simd::load_fvec(&net.output_weights[nbucket][i]);

		int idx = i / FLOATS_PER_VEC % L3_UNROLL;
		sums[idx] = simd::fma_f32(simd::mul_f32(val, val), weight, sums[idx]);
	}

	int num = L3_UNROLL;
	while (num > 1) {
		num /= 2;
		for (int i = 0; i < num; i++)
			sums[i] = simd::add_f32(sums[i], sums[i + num]);
	}

	float score = simd::reduce_add_ps(sums[0]) + net.output_biases[nbucket];

	return score * SCALE;
}
