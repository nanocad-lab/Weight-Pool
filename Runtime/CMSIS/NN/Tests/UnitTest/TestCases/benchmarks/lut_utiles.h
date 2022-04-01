#include <stdint.h>

void mem_write_range(int start_addr, int size);
void kernel_idx_gen(uint32_t* kernel_idx, const uint32_t start_addr, const int size);
