#include "lut_utiles.h"

void mem_write_range(int start_addr, int size){
    //Used to create/initialize the kernel pool into specific address of the memory, size is in byte
    for (int i = 0; i < size; i++){
        volatile unsigned char *ptr;
        int addr = start_addr + i;
        ptr = (char*)addr;
        *ptr = i;
    }
}

void kernel_idx_gen(uint32_t* kernel_idx, const uint32_t start_addr, const int size){
   for(int i = 0; i < size; i++){
      kernel_idx[i] = start_addr + 9*i;
   }
}