/*
 * oracle_core.c -- C implementation of DouEnMH unified oracle DP
 *
 * Implements dp_unified (N_active + N_preserve in a single pass) as a
 * compact C shared library callable from Python via ctypes.
 *
 * Compile (Windows/MSVC):
 *   cl /O2 /LD /std:c11 oracle_core.c /Fe:oracle_core.dll
 * Compile (Linux/GCC):
 *   gcc -O3 -shared -fPIC -std=c11 oracle_core.c -o oracle_core.so
 *
 * Algorithm: PerfectDou-style count-vector DP with optimizations:
 *   1. 15-dim int count vectors -- O(1) hashing, no sort / no Counter
 *   2. Bitmask pre-filter: (cmask & hmask) == cmask -- O(1)
 *   3. Combos sorted by size descending -- greedy near-optimal first
 *   4. Lower-bound pruning: ceil(remaining / max_combo_size) >= best -> skip
 *   5. Single recursion returning both N_active & N_preserve
 *   6. Global hash-table memo -- cross-call cache within one episode
 *   7. Greedy initial upper bound -- pruning active from first combo
 *   8. bomb_mask (uint16) + ctz -- no sentinel scan
 *   9. first_ci_for_rem lookup -- skip oversized combos instantly
 *  10. Merged hash+hmask computation -- single pass over 15 ranks
 *  11. MemoSlot 16-byte packing -- key==0 as empty sentinel
 *  12. Single-combo fast path -- skip recursion when combo exhausts hand
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* -- Tunable constants --------------------------------------------------- */

#define MEMO_BITS     20
#define MEMO_SIZE     (1 << MEMO_BITS)   /* 1048576 slots (~16 MB) */
#define MEMO_MASK     (MEMO_SIZE - 1)
#define MAX_PROBES    64
#define FLOAT_INF     1e30f              /* larger than max possible N (~20) */
#define NUM_RANKS     15
#define BOMB_PENALTY  1.0f

/* -- Portable count-trailing-zeros -------------------------------------- */

#ifdef _MSC_VER
  #include <intrin.h>
  static inline int ctz32(unsigned int x) {
      unsigned long idx;
      _BitScanForward(&idx, x);
      return (int)idx;
  }
#else
  static inline int ctz32(unsigned int x) {
      return __builtin_ctz(x);
  }
#endif

/* -- Hash-table for DP memoization -------------------------------------- */

typedef struct {
    uint64_t key;      /* 45-bit perfect hash; key==0 means empty slot   */
    float    val_a;    /* N_active                                       */
    float    val_p;    /* N_preserve                                     */
} MemoSlot;            /* 16 bytes -- 4 slots per 64B cache line         */

static MemoSlot *g_memo = NULL;

/* Forward declarations */
static inline uint64_t hash_and_mask(const int *hv, int *hmask_out);
static inline uint32_t key_to_slot(uint64_t k);

static void memo_init(void) {
    if (!g_memo) {
        g_memo = (MemoSlot *)calloc(MEMO_SIZE, sizeof(MemoSlot));
        /* calloc zeroes everything -> key==0 for all slots -> empty */
    }
}

static void memo_clear(void) {
    if (g_memo) {
        memset(g_memo, 0, MEMO_SIZE * sizeof(MemoSlot));
    }
}

static int memo_lookup(uint64_t key, float *a, float *p) {
    if (!g_memo) return 0;
    uint32_t idx = key_to_slot(key);
    for (int i = 0; i < MAX_PROBES; i++) {
        MemoSlot *slot = &g_memo[(idx + i) & MEMO_MASK];
        if (slot->key == 0) return 0;     /* empty slot -> miss */
        if (slot->key == key) {
            *a = slot->val_a;
            *p = slot->val_p;
            return 1;
        }
    }
    return 0;  /* probe limit exceeded -> treat as miss (non-fatal) */
}

static void memo_insert(uint64_t key, float a, float p) {
    if (!g_memo) memo_init();
    uint32_t idx = key_to_slot(key);
    for (int i = 0; i < MAX_PROBES; i++) {
        MemoSlot *slot = &g_memo[(idx + i) & MEMO_MASK];
        if (slot->key == 0 || slot->key == key) {
            slot->key   = key;
            slot->val_a = a;
            slot->val_p = p;
            return;
        }
    }
    /* table too full -- drop entry; correctness unaffected, only perf */
}

/* -- Perfect hash for 15-dim hand count vector -------------------------- */

static inline uint64_t hash_hand(const int *hv) {
    uint64_t h = 0;
    for (int i = 0; i < NUM_RANKS; i++) {
        h |= (uint64_t)(hv[i] & 0x7) << (i * 3);
    }
    return h;
}

/* -- Merged hash + hmask: single pass over 15 ranks -------------------- */

static inline uint64_t hash_and_mask(const int *hv, int *hmask_out) {
    uint64_t h = 0;
    int m = 0;
    for (int i = 0; i < NUM_RANKS; i++) {
        int c = hv[i];
        h |= (uint64_t)(c & 0x7) << (i * 3);
        if (c) m |= (1 << i);
    }
    *hmask_out = m;
    return h;
}

/* -- Mix 64-bit key into table index ----------------------------------- */

static inline uint32_t key_to_slot(uint64_t k) {
    /* SplitMix64-style finalizer: spread all 45 key bits across
       the low 17 bits so every rank influences the table slot. */
    k = (k ^ (k >> 30)) * 0xbf58476d1ce4e5b9ULL;
    k = (k ^ (k >> 27)) * 0x94d049bb133111ebULL;
    k = k ^ (k >> 31);
    return (uint32_t)(k & MEMO_MASK);
}

/* -- Greedy upper bound ------------------------------------------------- */

static void greedy_bound(
    const int *hv,
    int         remaining,
    const int  *combo_counts,
    const int  *combo_masks,
    const int  *combo_sizes,
    const uint16_t *combo_bomb_masks,
    int         n_combos,
    const int  *first_ci_for_rem,
    float      *bound_a,
    float      *bound_p)
{
    int hv_copy[NUM_RANKS];
    memcpy(hv_copy, hv, NUM_RANKS * sizeof(int));
    int rem = remaining;

    float ga = 0.0f;
    float gp = 0.0f;

    while (rem > 0) {
        int hmask;
        uint64_t _unused = hash_and_mask(hv_copy, &hmask);
        (void)_unused;

        int start_ci = first_ci_for_rem[rem];
        int found = 0;

        for (int ci = start_ci; ci < n_combos; ci++) {
            if ((combo_masks[ci] & hmask) != combo_masks[ci]) continue;

            const int *cv = combo_counts + ci * NUM_RANKS;
            int ok = 1;
            for (int i = 0; i < NUM_RANKS; i++) {
                if (cv[i] > hv_copy[i]) { ok = 0; break; }
            }
            if (!ok) continue;

            /* bomb penalty check before modifying hv_copy */
            float penalty = 0.0f;
            uint16_t bm = combo_bomb_masks[ci];
            while (bm) {
                int r = ctz32((unsigned int)bm);
                bm &= bm - 1;
                if (hv_copy[r] == 4) penalty += BOMB_PENALTY;
            }

            /* apply combo */
            for (int i = 0; i < NUM_RANKS; i++) hv_copy[i] -= cv[i];
            rem -= combo_sizes[ci];
            ga += 1.0f;
            gp += 1.0f + penalty;
            found = 1;
            break;
        }
        if (!found) break;  /* should never happen for valid hands */
    }

    *bound_a = ga;
    *bound_p = gp;
}

/* -- DP recursion ------------------------------------------------------- */

static void dp_recursive(
    const int  *hv,                  /* current hand count vector [15]      */
    int         remaining,           /* sum(hv) -- cards left               */
    const int  *combo_counts,        /* flattened combo cv: [n_combos*15]   */
    const int  *combo_masks,         /* combo bitmasks:       [n_combos]    */
    const int  *combo_sizes,         /* combo total cards:    [n_combos]    */
    const uint16_t *combo_bomb_masks,/* bomb rank bitmasks:   [n_combos]    */
    int         n_combos,            /* number of combos                    */
    int         max_combo_size,      /* combo_sizes[0] (largest combo)      */
    const int  *first_ci_for_rem,    /* first ci with size <= r, by r [21]  */
    float       init_a,              /* greedy upper bound for N_active     */
    float       init_p,              /* greedy upper bound for N_preserve   */
    float      *out_a,               /* [out] N_active                      */
    float      *out_p)               /* [out] N_preserve                    */
{
    int hmask;
    uint64_t key = hash_and_mask(hv, &hmask);

    if (memo_lookup(key, out_a, out_p)) return;

    /* Base case: empty hand */
    if (remaining == 0) {
        *out_a = 0.0f;
        *out_p = 0.0f;
        memo_insert(key, 0.0f, 0.0f);
        return;
    }

    /* Lower bound: ceil(remaining / max_combo_cards) */
    int lb = (remaining + max_combo_size - 1) / max_combo_size;

    float best_a = init_a;
    float best_p = init_p;

    int start_ci = first_ci_for_rem[remaining];

    for (int ci = start_ci; ci < n_combos; ci++) {
        /* ---- lower-bound pruning ---- */
        if (lb >= best_a && lb >= best_p) break;

        /* ---- bitmask pre-filter ---- */
        int cmask = combo_masks[ci];
        if ((cmask & hmask) != cmask) continue;

        /* ---- element-wise applicability check + remainder ---- */
        const int *cv = combo_counts + ci * NUM_RANKS;
        int ok = 1;
        int rem_arr[NUM_RANKS];
        for (int i = 0; i < NUM_RANKS; i++) {
            if (cv[i] > hv[i]) { ok = 0; break; }
            rem_arr[i] = hv[i] - cv[i];
        }
        if (!ok) continue;

        int combo_sz = combo_sizes[ci];

        /* ---- single-combo fast path: combo exhausts hand ---- */
        if (combo_sz == remaining) {
            float val_a = 1.0f;
            if (val_a < best_a) best_a = val_a;

            float penalty = 0.0f;
            uint16_t bm = combo_bomb_masks[ci];
            while (bm) {
                int r = ctz32((unsigned int)bm);
                bm &= bm - 1;
                if (hv[r] == 4) penalty += BOMB_PENALTY;
            }
            float val_p = 1.0f + penalty;
            if (val_p < best_p) best_p = val_p;
            continue;
        }

        /* ---- recurse ---- */
        int rem_cards = remaining - combo_sz;
        float sub_a, sub_p;
        dp_recursive(rem_arr, rem_cards,
                     combo_counts, combo_masks, combo_sizes,
                     combo_bomb_masks, n_combos, max_combo_size,
                     first_ci_for_rem,
                     FLOAT_INF,  /* no bound propagation — unsafe */
                     FLOAT_INF,
                     &sub_a, &sub_p);

        /* ---- N_active (no penalty) ---- */
        float val_a = 1.0f + sub_a;
        if (val_a < best_a) best_a = val_a;

        /* ---- N_preserve (bomb-break penalty) ---- */
        float penalty = 0.0f;
        uint16_t bm = combo_bomb_masks[ci];
        while (bm) {
            int r = ctz32((unsigned int)bm);
            bm &= bm - 1;
            if (hv[r] == 4) penalty += BOMB_PENALTY;
        }
        float val_p = 1.0f + sub_p + penalty;
        if (val_p < best_p) best_p = val_p;
    }

    *out_a = best_a;
    *out_p = best_p;
    memo_insert(key, best_a, best_p);
}

/* -- Public API --------------------------------------------------------- */

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT __attribute__((visibility("default")))
#endif

/*
 * dp_unified_c -- single-call DP returning (N_active, N_preserve).
 *
 * All combo_* arrays must be sorted by size descending (same as
 * _precompute_combo_data in Python).
 *
 * combo_bomb_masks: uint16 per combo, bit i=1 if combo uses 1-3 cards
 *   of rank i (could break a bomb).  Zero if no bomb ranks.
 *
 * first_ci_for_rem[21]: lookup table; first_ci_for_rem[r] = index of
 *   first combo with size <= r.  Computed once per hand, reused across
 *   all recursive calls.
 *
 * Returns 0 on success.  result[0]=n_active, result[1]=n_preserve.
 * The internal hash table is checked before computing and updated
 * after, so repeated calls with sub-hands hit in O(1).
 */
EXPORT int dp_unified_c(
    const int     *hand_vec,            /* [15] count vector              */
    const int     *combo_counts,        /* [n_combos * 15] flattened      */
    const int     *combo_masks,         /* [n_combos] bitmasks            */
    const int     *combo_sizes,         /* [n_combos] total cards         */
    const uint16_t *combo_bomb_masks,   /* [n_combos] bomb rank bitmasks  */
    int            n_combos,
    const int     *first_ci_for_rem,    /* [21] lookup: remaining -> ci   */
    float         *result)              /* [2] output: n_active, n_preserve */
{
    memo_init();

    /* Global-cache hit -> instant return */
    int hmask;
    uint64_t key = hash_and_mask(hand_vec, &hmask);
    (void)hmask;
    if (memo_lookup(key, &result[0], &result[1])) {
        return 0;
    }

    /* Sum hand vector */
    int remaining = 0;
    for (int i = 0; i < NUM_RANKS; i++) {
        remaining += hand_vec[i];
    }

    if (remaining == 0) {
        result[0] = 0.0f;
        result[1] = 0.0f;
        memo_insert(key, 0.0f, 0.0f);
        return 0;
    }

    int max_size = (n_combos > 0) ? combo_sizes[0] : 1;

    /* ---- greedy upper bound: pruning active from the first combo ---- */
    float greedy_a = FLOAT_INF;
    float greedy_p = FLOAT_INF;
    if (n_combos > 0) {
        greedy_bound(hand_vec, remaining,
                     combo_counts, combo_masks, combo_sizes,
                     combo_bomb_masks, n_combos, first_ci_for_rem,
                     &greedy_a, &greedy_p);
    }

    dp_recursive(hand_vec, remaining,
                 combo_counts, combo_masks, combo_sizes,
                 combo_bomb_masks, n_combos, max_size,
                 first_ci_for_rem,
                 greedy_a, greedy_p,
                 &result[0], &result[1]);

    return 0;
}

/* dp_clear_cache_c -- clear the internal memo table.
 * Call at the start of each new game episode. */
EXPORT void dp_clear_cache_c(void) {
    memo_clear();
}

/* dp_cache_size_c -- return number of cached entries (for debugging). */
EXPORT int dp_cache_size_c(void) {
    if (!g_memo) return 0;
    int n = 0;
    for (int i = 0; i < MEMO_SIZE; i++) {
        if (g_memo[i].key != 0) n++;
    }
    return n;
}
