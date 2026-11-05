// Prints key=value facts about the RISC-V vector unit and counters of the
// machine it runs on: VLEN, whether perf_event cycle counting works, and the
// clock estimated from cycles over rdtime ticks.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <riscv_vector.h>

static int open_counter(uint64_t config) {
  struct perf_event_attr attr;
  memset(&attr, 0, sizeof attr);
  attr.type = PERF_TYPE_HARDWARE;
  attr.size = sizeof attr;
  attr.config = config;
  attr.exclude_kernel = 1;
  attr.exclude_hv = 1;
  return (int) syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0);
}

int main(void) {
  printf("vlen_bits=%zu\n", __riscv_vsetvlmax_e8m1() * 8);
  int fd = open_counter(PERF_COUNT_HW_CPU_CYCLES);
  printf("perf_cycles=%s\n", fd >= 0 ? "yes" : "no");
  uint64_t t0, t1, c = 0;
  __asm__ volatile("rdtime %0" : "=r"(t0));
  volatile double x = 0;
  for (int i = 0; i < 2000000; i++) x += i;
  __asm__ volatile("rdtime %0" : "=r"(t1));
  if (fd >= 0 && read(fd, &c, sizeof c) != sizeof c) c = 0;
  // tt-bh-linux's device tree sets timebase-frequency to 50 MHz.
  if (c) printf("clock_ghz=%.3f\n", c / ((t1 - t0) / 50e6) / 1e9);
  return 0;
}
