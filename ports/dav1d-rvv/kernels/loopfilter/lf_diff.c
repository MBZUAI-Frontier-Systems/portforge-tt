/* exact single-edge differential for the lpf RVV port: builds random edges
 * (all wd values, random E/I/H from a checkasm-style LUT), runs the C
 * loop_filter (verbatim 8bpc from loopfilter_tmpl.c) on one copy and the RVV
 * sb wrappers on another, and prints full inputs for the first mismatch. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <signal.h>
#include <initializer_list>
#include <unistd.h>
#include <sys/wait.h>
#include <ucontext.h>

typedef void (*lpf_fn)(uint8_t *dst, ptrdiff_t stride, const uint32_t *mask,
                       const uint8_t (*l)[4], ptrdiff_t b4_stride,
                       const void *lut, int w);

extern "C" void dav1d_lpf_h_sb_y_8bpc_rvv(uint8_t *, ptrdiff_t, const uint32_t *,
                                      const uint8_t (*)[4], ptrdiff_t, const void *, int);
extern "C" void dav1d_lpf_v_sb_y_8bpc_rvv(uint8_t *, ptrdiff_t, const uint32_t *,
                                      const uint8_t (*)[4], ptrdiff_t, const void *, int);
extern "C" void dav1d_lpf_h_sb_uv_8bpc_rvv(uint8_t *, ptrdiff_t, const uint32_t *,
                                       const uint8_t (*)[4], ptrdiff_t, const void *, int);
extern "C" void dav1d_lpf_v_sb_uv_8bpc_rvv(uint8_t *, ptrdiff_t, const uint32_t *,
                                       const uint8_t (*)[4], ptrdiff_t, const void *, int);

static uint64_t rs;
static uint32_t rnd(void)
{
    rs ^= rs >> 12; rs ^= rs << 25; rs ^= rs >> 27;
    return (uint32_t)(rs * 0x2545F4914F6CDD1DULL >> 32);
}

static inline int iclip(int v, int min, int max)
{
    return v < min ? min : v > max ? max : v;
}
static inline int iclip_diff_(int v) { return iclip(v, -128, 127); }
static inline int imin_(int a, int b) { return a < b ? a : b; }

static void loop_filter_c(uint8_t *dst, int E, int I, int H,
                          const ptrdiff_t stridea, const ptrdiff_t strideb,
                          const int wd)
{
    const int F = 1;
    for (int i = 0; i < 4; i++, dst += stridea) {
        int p6, p5, p4, p3, p2;
        int p1 = dst[strideb * -2], p0 = dst[strideb * -1];
        int q0 = dst[strideb * +0], q1 = dst[strideb * +1];
        int q2, q3, q4, q5, q6;
        int fm, flat8out = 0, flat8in = 0;
        fm = abs(p1 - p0) <= I && abs(q1 - q0) <= I &&
             abs(p0 - q0) * 2 + (abs(p1 - q1) >> 1) <= E;
        if (wd > 4) {
            p2 = dst[strideb * -3];
            q2 = dst[strideb * +2];
            fm &= abs(p2 - p1) <= I && abs(q2 - q1) <= I;
            if (wd > 6) {
                p3 = dst[strideb * -4];
                q3 = dst[strideb * +3];
                fm &= abs(p3 - p2) <= I && abs(q3 - q2) <= I;
            }
        }
        if (!fm) continue;
        if (wd >= 16) {
            p6 = dst[strideb * -7]; p5 = dst[strideb * -6];
            p4 = dst[strideb * -5]; q4 = dst[strideb * +4];
            q5 = dst[strideb * +5]; q6 = dst[strideb * +6];
            flat8out = abs(p6 - p0) <= F && abs(p5 - p0) <= F &&
                       abs(p4 - p0) <= F && abs(q4 - q0) <= F &&
                       abs(q5 - q0) <= F && abs(q6 - q0) <= F;
        }
        if (wd >= 6)
            flat8in = abs(p2 - p0) <= F && abs(p1 - p0) <= F &&
                      abs(q1 - q0) <= F && abs(q2 - q0) <= F;
        if (wd >= 8)
            flat8in &= abs(p3 - p0) <= F && abs(q3 - q0) <= F;
        if (wd >= 16 && (flat8out & flat8in)) {
            dst[strideb * -6] = (6 * p6 + 7 * p6 + 2 * p5 + 2 * p4 + p3 + p2 + p1 + p0 + q0 + 8) >> 4;
            dst[strideb * -5] = (5 * p6 + 2 * p5 + 2 * p4 + 2 * p3 + p2 + p1 + p0 + q0 + q1 + 8) >> 4;
            dst[strideb * -4] = (4 * p6 + p5 + 2 * p4 + 2 * p3 + 2 * p2 + p1 + p0 + q0 + q1 + q2 + 8) >> 4;
            dst[strideb * -3] = (3 * p6 + p5 + p4 + 2 * p3 + 2 * p2 + 2 * p1 + p0 + q0 + q1 + q2 + q3 + 8) >> 4;
            dst[strideb * -2] = (2 * p6 + p5 + p4 + p3 + 2 * p2 + 2 * p1 + 2 * p0 + q0 + q1 + q2 + q3 + q4 + 8) >> 4;
            dst[strideb * -1] = (p6 + p5 + p4 + p3 + p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + q2 + q3 + q4 + q5 + 8) >> 4;
            dst[strideb * +0] = (p5 + p4 + p3 + p2 + p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + q3 + q4 + q5 + q6 + 8) >> 4;
            dst[strideb * +1] = (p4 + p3 + p2 + p1 + p0 + 2 * q0 + 2 * q1 + 2 * q2 + q3 + q4 + q5 + 2 * q6 + 8) >> 4;
            dst[strideb * +2] = (p3 + p2 + p1 + p0 + q0 + 2 * q1 + 2 * q2 + 2 * q3 + q4 + q5 + 3 * q6 + 8) >> 4;
            dst[strideb * +3] = (p2 + p1 + p0 + q0 + q1 + 2 * q2 + 2 * q3 + 2 * q4 + q5 + 4 * q6 + 8) >> 4;
            dst[strideb * +4] = (p1 + p0 + q0 + q1 + q2 + 2 * q3 + 2 * q4 + 2 * q5 + 5 * q6 + 8) >> 4;
            dst[strideb * +5] = (p0 + q0 + q1 + q2 + q3 + 2 * q4 + 2 * q5 + 7 * q6 + 8) >> 4;
        } else if (wd >= 8 && flat8in) {
            dst[strideb * -3] = (3 * p3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3;
            dst[strideb * -2] = (2 * p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3;
            dst[strideb * -1] = (p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3;
            dst[strideb * +0] = (p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3;
            dst[strideb * +1] = (p1 + p0 + q0 + 2 * q1 + q2 + 2 * q3 + 4) >> 3;
            dst[strideb * +2] = (p0 + q0 + q1 + 2 * q2 + 3 * q3 + 4) >> 3;
        } else if (wd == 6 && flat8in) {
            dst[strideb * -2] = (3 * p2 + 2 * p1 + 2 * p0 + q0 + 4) >> 3;
            dst[strideb * -1] = (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3;
            dst[strideb * +0] = (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3;
            dst[strideb * +1] = (p0 + 2 * q0 + 2 * q1 + 3 * q2 + 4) >> 3;
        } else {
            const int hev = abs(p1 - p0) > H || abs(q1 - q0) > H;
            if (hev) {
                int f = iclip_diff_(p1 - q1);
                f = iclip_diff_(3 * (q0 - p0) + f);
                int f1 = imin_(f + 4, 127) >> 3;
                int f2 = imin_(f + 3, 127) >> 3;
                dst[strideb * -1] = iclip(p0 + f2, 0, 255);
                dst[strideb * +0] = iclip(q0 - f1, 0, 255);
            } else {
                int f = iclip_diff_(3 * (q0 - p0));
                int f1 = imin_(f + 4, 127) >> 3;
                int f2 = imin_(f + 3, 127) >> 3;
                dst[strideb * -1] = iclip(p0 + f2, 0, 255);
                dst[strideb * +0] = iclip(q0 - f1, 0, 255);
                f = (f1 + 1) >> 1;
                dst[strideb * -2] = iclip(p1 + f, 0, 255);
                dst[strideb * +1] = iclip(q1 - f, 0, 255);
            }
        }
    }
}

static void handler(int sig, siginfo_t *si, void *uc)
{
    ucontext_t *ctx = (ucontext_t *)uc;
    fprintf(stderr, "TRAP sig=%d addr=%p pc=%p\n", sig, si->si_addr,
            (void *)ctx->uc_mcontext.__gregs[0]);
    _exit(100);
}

#define NL 160
#define PITCH 32
static uint8_t cmem[NL * PITCH], amem[NL * PITCH], pmem[NL * PITCH];

static int try_edge(lpf_fn fn, int dir, int wd, int E, int I, int H, int nlines)
{
    memset(cmem, 0x80, sizeof(cmem));
    memset(amem, 0x80, sizeof(amem));
    for (int i = 0; i < NL * PITCH; i++) {
        cmem[i] = amem[i] = rnd() & 255;
    }
    /* dst: h -> rows, edge at byte 8 of each 32-byte row; v -> columns */
    uint8_t *cd = dir ? cmem + 8 * PITCH + 8 : cmem + 8 * PITCH + 8;
    uint8_t *ad = dir ? amem + 8 * PITCH + 8 : amem + 8 * PITCH + 8;
    ptrdiff_t stride = PITCH;
    uint32_t vmask[4] = { 1, 0, 0, 0 };
    uint8_t l[8][4] = { { (uint8_t)((H << 4) | 32) } }; /* L=32: E/I from LUT */
    struct { uint8_t e[64]; uint8_t i[64]; uint64_t sharp[2]; } lut;
    for (int level = 0; level < 64; level++) {
        int limit = level;
        const int sharp = 0;
        if (sharp > 0) { limit >>= (sharp + 3) >> 2; }
        limit = limit < 1 ? 1 : limit;
        lut.i[level] = limit;
        lut.e[level] = 2 * (level + 2) + limit;
    }
    (void)E; (void)I;
    pid_t pid = fork();
    if (pid == 0) {
        if (dir == 0 && wd >= 8)
            dav1d_lpf_h_sb_y_8bpc_rvv(ad, stride, vmask,
                                      (const uint8_t (*)[4])l + 1, 4, &lut, 1);
        else if (dir == 0)
            dav1d_lpf_h_sb_uv_8bpc_rvv(ad, stride, vmask,
                                       (const uint8_t (*)[4])l + 1, 4, &lut, 1);
        else if (wd >= 8)
            dav1d_lpf_v_sb_y_8bpc_rvv(ad, stride, vmask,
                                      (const uint8_t (*)[4])l + 32, 32, &lut, 1);
        else
            dav1d_lpf_v_sb_uv_8bpc_rvv(ad, stride, vmask,
                                       (const uint8_t (*)[4])l + 32, 32, &lut, 1);
        _exit(0);
    }
    int st;
    waitpid(pid, &st, 0);
    if (WIFSIGNALED(st) || WEXITSTATUS(st)) return -1;

    /* C reference on cmem (wd<=6 chroma variant unused for y) */
    for (int i = 0; i < 4; i++) {
        /* each of the 4 lines gets the same filter; process all at once */
    }
    if (dir == 0)
        loop_filter_c(cd, lut.e[32], lut.i[32], H, stride, 1, wd);
    else
        loop_filter_c(cd, lut.e[32], lut.i[32], H, 1, stride, wd);
    (void)nlines;

    return memcmp(cmem, amem, sizeof(cmem)) != 0;
}

static int fails, total;

static void run_sb(lpf_fn fn, const char *name, int dir, int wdclass, uint64_t seed)
{
    /* wdclass: 0 = wd4, 1 = wd8, 2 = wd16, 3 = wd6(chroma) */
    int wd = wdclass == 0 ? 4 : wdclass == 1 ? 8 : wdclass == 2 ? 16 : 6;
    int n_blks = wdclass == 3 ? 16 : 32;
    memset(cmem, 0x80, sizeof(cmem));
    memset(amem, 0x80, sizeof(amem));
    for (int i = 0; i < NL * PITCH; i++) pmem[i] = 0x80;
    uint8_t l[64][4];
    memset(l, 0, sizeof(l));
    struct { uint8_t e[64]; uint8_t i[64]; uint64_t sharp[2]; } lut;
    for (int level = 0; level < 64; level++) {
        int limit = level;
        limit = limit < 1 ? 1 : limit;
        lut.i[level] = limit;
        lut.e[level] = 2 * (level + 2) + limit;
    }
    for (int j = 0; j < n_blks; j++) {
        uint8_t Lv = getenv("ALLL") ? 32 : (rnd() & 63);
        l[dir ? j + 32 : 2 * j + 1][wdclass == 3 ? 2 : 0] = Lv;
    }
    uint32_t vmask[4] = { 0, 0, 0, 0 };
    /* every block gets an edge; wd via vmask word for y, [1] for uv */
    if (wdclass == 3) vmask[1] = (n_blks == 16) ? 0xffffu : 0xffffffffu;
    else vmask[0] = (n_blks == 16) ? 0xffffu : 0xffffffffu;
    if (wdclass == 2) vmask[1] = vmask[0];
    if (wdclass == 1) vmask[1] = vmask[0];
    if (getenv("ONEBIT")) { vmask[0] = 1; vmask[1] = 0; vmask[2] = 0; }

    /* checkasm-style border init: mostly filterable lines */
    for (int bit = 0; bit < n_blks; bit++) {
        const int Lrow = l[dir ? 32 + bit : 1 + 2 * bit][wdclass == 3 ? 2 : 0];
        const int L = Lrow ? Lrow : l[dir ? bit : 2 * bit][wdclass == 3 ? 2 : 0];
        const int E0 = lut.e[L], I0 = lut.i[L];
        for (int i = 0; i < 4; i++) {
            const int kind = rnd() % 4;
            const int base = rnd() & 255;
            const int edge_diff = (int)(rnd() % ((E0 + 2) * 4)) - 2 * (E0 + 2);
            uint8_t *d = dir ? cmem + 8 * PITCH + 8 + bit * 4 + i
                             : cmem + 8 * PITCH + 8 + (bit * 4 + i) * PITCH;
            const int stp = dir ? PITCH : 1;
            const int F = 1;
            if (kind == 0) {
                for (int k = -8; k < 8; k++) d[k * stp] = rnd() & 255;
            } else {
                d[0] = base;
                d[-1 * stp] = iclip(base + edge_diff, 0, 255);
                int lo = 1, hi = 7;
                if (kind == 2) lo = 4;
                for (int k = lo; k <= hi; k++) {
                    const int flat = (kind == 1) || (kind == 2 && k >= 4);
                    const int range = flat ? (2 * (F + 1)) : (2 * (I0 + 1));
                    const int off = flat ? (F + 1) : (I0 + 1);
                    d[-(1 + k) * stp] = iclip(d[-1 * stp] + (int)(rnd() % range) - off, 0, 255);
                    d[+k * stp] = iclip(d[0] + (int)(rnd() % range) - off, 0, 255);
                }
                if (kind == 3) {
                    for (int k = 4; k < 8; k++) {
                        d[-(1 + k) * stp] = rnd() & 255;
                        d[+k * stp] = rnd() & 255;
                    }
                }
            }
        }
    }
    memcpy(pmem, cmem, sizeof(cmem));
    memcpy(amem, cmem, sizeof(cmem));
    uint8_t *cd = dir ? cmem + 8 * PITCH + 8 : cmem + 8 * PITCH + 8;
    uint8_t *ad = dir ? amem + 8 * PITCH + 8 : amem + 8 * PITCH + 8;

    if (dir == 0 && wdclass != 3)
        dav1d_lpf_h_sb_y_8bpc_rvv(ad, PITCH, vmask,
                                  (const uint8_t (*)[4])((uint8_t *)l + 4), 2, &lut, n_blks);
    else if (dir == 1 && wdclass != 3)
        dav1d_lpf_v_sb_y_8bpc_rvv(ad, PITCH, vmask,
                                  (const uint8_t (*)[4])((uint8_t *)l + 128), 32, &lut, n_blks);
    else if (dir == 0)
        dav1d_lpf_h_sb_uv_8bpc_rvv(ad, PITCH, vmask,
                                   (const uint8_t (*)[4])((uint8_t *)l + 6), 2, &lut, n_blks);
    else
        dav1d_lpf_v_sb_uv_8bpc_rvv(ad, PITCH, vmask,
                                   (const uint8_t (*)[4])((uint8_t *)l + 130), 32, &lut, n_blks);
    /* C reference */
    for (uint32_t y = 1; ; y <<= 1) {
        uint32_t vm = vmask[0] | vmask[1] | (wdclass == 3 ? 0 : vmask[2]);
        if (!(vm & ~(y - 1))) break;
        if (vm & y) {
            int bit = __builtin_ctz(y);
            const uint8_t *row = l[(dir ? 32 + bit : 1 + 2 * bit)];
            int L = row[wdclass == 3 ? 2 : dir];
            if (!L) L = l[dir ? bit : 2 * bit][wdclass == 3 ? 2 : 0];
            if (L) {
                int E = lut.e[L], I = lut.i[L], H = L >> 4;
                if (dir) loop_filter_c(cd + bit * 4, E, I, H, 1, PITCH, wd);
                else loop_filter_c(cd + bit * 4 * PITCH, E, I, H, PITCH, 1, wd);
            }
        }
    }
    total++;
    if (memcmp(cmem, amem, sizeof(cmem))) {
        fails++;
        if (fails <= 2) {
            printf("asm wrote:");
            for (int r = 0; r < NL; r++)
                if (memcmp(amem + r * PITCH, pmem + r * PITCH, PITCH)) printf(" %d", r);
            printf("\nC wrote:  ");
            for (int r = 0; r < NL; r++)
                if (memcmp(cmem + r * PITCH, pmem + r * PITCH, PITCH)) printf(" %d", r);
            printf("\n");
            printf("%s wd=%d seed=%lu: MISMATCH; first differing rows:\n", name, wd, (unsigned long)seed);
            int shown = 0;
            for (int r = 0; r < NL && shown < 4; r++) {
                if (memcmp(cmem + r * PITCH, amem + r * PITCH, PITCH)) {
                    printf(" row %d (bit %d line %d, L=%d E=%d I=%d H=%d)\n  pre:", r,
                       r / 4, r % 4,
                       l[dir ? 32 + r / 4 : 1 + 2 * (r / 4)][0],
                       lut.e[l[dir ? 32 + r / 4 : 1 + 2 * (r / 4)][0]],
                       lut.i[l[dir ? 32 + r / 4 : 1 + 2 * (r / 4)][0]],
                       l[dir ? 32 + r / 4 : 1 + 2 * (r / 4)][0] >> 4);
                for (int cc = 0; cc < PITCH; cc++) printf(" %02x", pmem[r * PITCH + cc]);
                printf("\n  C  :", r);
                    for (int cc = 0; cc < PITCH; cc++) printf(" %02x", cmem[r * PITCH + cc]);
                    printf("\n  RVV:");
                    for (int cc = 0; cc < PITCH; cc++) printf(" %02x", amem[r * PITCH + cc]);
                    printf("\n");
                    shown++;
                }
            }
        }
    }
}

static int known_test(void)
{
    /* One edge, all 4 lines identical, taps set explicitly:
     * p6..p4=1, p3=2, p2=3, p1=10, p0=20 | q0=30, q1=40, q2=3, q3=2, q4..q6=1
     * L = 63 -> I=63, E=193, H=3. wd4: fm passes (|10-20|=10<=63 etc),
     * hev (10>3) -> expect dst[-1]=20, dst[0]=30, no dst[-2]/[+1] writes. */
    memset(cmem, 0x80, sizeof(cmem));
    memset(amem, 0x80, sizeof(amem));
    memset(pmem, 0x80, sizeof(pmem));
    struct { uint8_t e[64]; uint8_t i[64]; uint64_t sharp[2]; } lut;
    for (int level = 0; level < 64; level++) {
        int limit = level; limit = limit < 1 ? 1 : limit;
        lut.i[level] = limit;
        lut.e[level] = 2 * (level + 2) + limit;
    }
    for (int line = 0; line < 4; line++) {
        uint8_t *d = cmem + 8 * PITCH + 8 + line * PITCH;
        d[-2] = 10 + line; d[-1] = 20 + line;
        d[0] = 30 + line; d[1] = 40 + line;
        for (int k = -8; k < -2; k++) d[k] = 1;
        for (int k = 2; k < 8; k++) d[k] = 2;
        memcpy(amem + (d - cmem) - 8, d - 8, 24); /* d[-8..+7] */
    }
    uint32_t vmask[4] = { 1, 0, 0, 0 };
    uint8_t l[8][4] = { { 63 } };
    pid_t pid = fork();
    if (pid == 0) {
        dav1d_lpf_h_sb_y_8bpc_rvv(amem + 8 * PITCH + 8, PITCH, vmask,
                                  (const uint8_t (*)[4])((uint8_t *)l + 4), 2, &lut, 1);
        _exit(0);
    }
    int st; waitpid(pid, &st, 0);
    loop_filter_c(cmem + 8 * PITCH + 8, lut.e[63], lut.i[63], 3, PITCH, 1, 4);
    printf("known test: C  line0: dst[-2..1] = %02x %02x %02x %02x\n",
           cmem[8 * PITCH + 6], cmem[8 * PITCH + 7], cmem[8 * PITCH + 8], cmem[8 * PITCH + 9]);
    printf("known test: RVV line0: dst[-2..1] = %02x %02x %02x %02x\n",
           amem[8 * PITCH + 6], amem[8 * PITCH + 7], amem[8 * PITCH + 8], amem[8 * PITCH + 9]);
    for (int line = 1; line < 4; line++) {
        int o = 8 * PITCH + 8 + line * PITCH;
        printf("known test: RVV line%d: dst[-2..1] = %02x %02x %02x %02x\n", line,
               amem[o - 2], amem[o - 1], amem[o], amem[o + 1]);
    }
    return memcmp(cmem, amem, sizeof(cmem)) != 0;
}

int main(void)
{
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = handler;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGILL, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGSEGV, &sa, NULL);

    if (known_test()) printf("KNOWN TEST FAILED\n");
    else printf("known test passed\n");
    /* H=0 sub-cases */
    {
        struct { int p1, p0, q0, q1, L; const char *what; int exp[4]; } cases[] = {
            { 10, 20, 35, 40, 15, "hev H=0", { 11, 22, 33, 39 } },
            { 20, 20, 30, 30, 15, "!hev H=0", { 22, 24, 26, 28 } },
        };
        for (unsigned ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
            memset(cmem, 0x80, sizeof(cmem));
            memset(amem, 0x80, sizeof(amem));
            struct { uint8_t e[64]; uint8_t i[64]; uint64_t sharp[2]; } lut;
            for (int level = 0; level < 64; level++) {
                int limit = level; limit = limit < 1 ? 1 : limit;
                lut.i[level] = limit;
                lut.e[level] = 2 * (level + 2) + limit;
            }
            const int L = cases[ci].L;
            const int fI = getenv("BIGI") ? 255 : lut.i[L];
            const int fE = getenv("BIGE") ? 255 : lut.e[L];
            lut.i[L] = fI; lut.e[L] = fE;
            for (int line = 0; line < 4; line++) {
                uint8_t *d = cmem + 8 * PITCH + 8 + line * PITCH;
                for (int k = -8; k < 8; k++) d[k] = 0x40;
                d[-2] = cases[ci].p1; d[-1] = cases[ci].p0;
                d[0] = cases[ci].q0; d[1] = cases[ci].q1;
                memcpy(amem + (d - cmem) - 8, d - 8, 24);
            }
            uint32_t vmask[4] = { 1, 0, 0, 0 };
            uint8_t l[8][4]; memset(l, 0, sizeof(l)); l[1][0] = (uint8_t)L;
            dav1d_lpf_h_sb_y_8bpc_rvv(amem + 8 * PITCH + 8, PITCH, vmask,
                                      (const uint8_t (*)[4])((uint8_t *)l + 4), 2, &lut, 1);
            loop_filter_c(cmem + 8 * PITCH + 8, fE, fI, L >> 4, PITCH, 1, 4);
            int o = 8 * PITCH + 8;
            printf("  sub-markers: entry=%02x lut=%02x core=%02x\n",
                   amem[o + 1200], amem[o + 1204], amem[o + 1208]);
            printf("%s: C  = %02x %02x %02x %02x | RVV = %02x %02x %02x %02x | exp %02x %02x %02x %02x\n",
                   cases[ci].what,
                   cmem[o - 2], cmem[o - 1], cmem[o], cmem[o + 1],
                   amem[o - 2], amem[o - 1], amem[o], amem[o + 1],
                   cases[ci].exp[0], cases[ci].exp[1], cases[ci].exp[2], cases[ci].exp[3]);
            {   /* debug: wrapper/core markers at dst+1200.. */
                const uint8_t *db = amem + o + 1200;
                printf("  markers: entry=%02x lut=%02x core=%02x fmdbg=%02x\n",
                       db[0], db[4], db[8], db[400 - 1200 + 800]);
            }
        }
    }

    const char *names[4] = { "wd4", "wd8", "wd16", "wd6uv" };
    for (int wc = 0; wc < 4; wc++) {
        for (uint64_t seed = 1; seed <= 4; seed++) {
            rs = seed * 0x9E3779B97F4A7C15ULL;
            run_sb(wc == 3 ? dav1d_lpf_h_sb_uv_8bpc_rvv : dav1d_lpf_h_sb_y_8bpc_rvv,
                   names[wc], 0, wc, seed);
            run_sb(wc == 3 ? dav1d_lpf_v_sb_uv_8bpc_rvv : dav1d_lpf_v_sb_y_8bpc_rvv,
                   names[wc], 1, wc, seed);
        }
    }
    printf("%d failures\n", fails);
    return fails != 0;
}
