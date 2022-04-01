/*
The filter pooling impelemntation that clusters weights in z-dimension (channels) instead of 2d filters
For this impelmentation, the input and weights need to be assumed to be packed in z-dimension in memory as well
Thus, the output generated by the covnolution should also be packed in z-dimension. 
CMSIS already pack the acitvation and weight in cahnnel first order
Need also to find a way to check whether the indexing logic is correctly implemented or not. 

In this implementation, the normalization factor is no longer for each channel, now it's for each 8-wide weight 
cluster along the z dimension. 
*/
#include "..\..\Include\arm_nnfunctions.h"
#include "..\..\Include\arm_nnsupportfunctions.h"

#define LUT_PREC 5  
#define LUT_SIZE 32
#define LUT_EFFECTIVE_SIZE LUT_SIZE //just for faster debugging and testing, should not be used in actual runtime benchmark
#define FW_GRAN 8 //granularity of fixed weight, should be power of 2 for better efficiency
//Define some macros to get and set bit for index generation
#define GETBIT(var, bit)	(((var) >> (bit)) & 1)
#define SETBIT(var, bit)	var |= (1 << (bit))

arm_status lut_conv_zdim_v1(const cmsis_nn_context *ctx,
                           const cmsis_nn_conv_params *conv_params,
                           const cmsis_nn_per_channel_quant_params *quant_params,
                           const cmsis_nn_dims *input_dims,
                           const q7_t *input_data,
                           const cmsis_nn_dims *filter_dims,
                           const uint8_t* kernel_idx,
                           const cmsis_nn_dims *bias_dims,
                           const int32_t *bias_data,
                           const cmsis_nn_dims *output_dims,
                           const uint8_t* filter_pool_data,
                           q7_t *output_data)
{
  (void)bias_dims;
  q15_t *buffer_a = (q15_t *)ctx->buf;

  const uint16_t input_batches = input_dims->n;
  const uint16_t input_x = input_dims->w;
  const uint16_t input_y = input_dims->h;
  const uint16_t input_ch = input_dims->c;
  const uint16_t kernel_x = filter_dims->w;
  const uint16_t kernel_y = filter_dims->h;
  const uint16_t output_x = output_dims->w;
  const uint16_t output_y = output_dims->h;
  const uint16_t output_ch = output_dims->c;

  const uint16_t pad_x = conv_params->padding.w;
  const uint16_t pad_y = conv_params->padding.h;
  const uint16_t stride_x = conv_params->stride.w;
  const uint16_t stride_y = conv_params->stride.h;

  const int32_t input_offset = conv_params->input_offset;
  const int32_t out_offset = conv_params->output_offset;
  const int32_t out_activation_min = conv_params->activation.min;
  const int32_t out_activation_max = conv_params->activation.max;
  int32_t *output_mult = quant_params->multiplier;
  int32_t *output_shift = quant_params->shift;
  char* result_ptr;
  char input_tmp, extracted_bit;

  uint16_t input_index[LUT_PREC] = {0};
  int16_t* conv_out_holder = malloc(output_ch*sizeof(int16_t));
  //static int16_t conv_out_holder[128];

  int i_batch;
  for (i_batch = 0; i_batch < input_batches; i_batch++)
  {
      /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
      (void)buffer_a;
      int32_t i_out_ch, i_out_y, i_out_x, i_input_ch, i_ker_y, i_ker_x, j_input_ch;
      int32_t conv_out, logical_kernel_idx, result_idx, block_cnt;//block_cnt is a counter to count the number of 8-wide weight blocks processed
      int16_t partial_sum;
      uint8_t physical_kernel_idx;

      for (i_out_y = 0; i_out_y < output_y; i_out_y++)
      {
          for (i_out_x = 0; i_out_x < output_x; i_out_x++)
          {   
              block_cnt = 0;//initialize and reset the block counter
              //conv_out = 0;
              //int16_t conv_out_holder[output_ch]; //initialize the conv result holder, one for each filter
              memset(conv_out_holder, 0, output_ch*sizeof(int16_t));//set the conv out holder to zero for accurate accumulation

              const int32_t base_idx_y = stride_y * i_out_y - pad_y;
              const int32_t base_idx_x = stride_x * i_out_x - pad_x;

              const int32_t ker_y_start = MAX(0, -base_idx_y);
              const int32_t ker_x_start = MAX(0, -base_idx_x); 

              const int32_t ker_y_end = MIN(kernel_y, input_y - base_idx_y);
              const int32_t ker_x_end = MIN(kernel_x, input_x - base_idx_x);

              uint8_t lut_buffer[LUT_PREC*LUT_SIZE];

              //for (i_input_ch = 0; i_input_ch < input_ch; i_input_ch++)
              for (i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
              {
                for (i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                {
                  const int32_t in_row = base_idx_y + i_ker_y; 
                  const int32_t in_col = base_idx_x + i_ker_x;

                  for (i_input_ch = 0; i_input_ch < input_ch; i_input_ch = i_input_ch + FW_GRAN)
                  {
                    int kernel_pixel_cnt = 0;
                    //index generation, outer loop is for input pixels, and inner loop is for 8 index
                    
                    for (j_input_ch = i_input_ch; j_input_ch < i_input_ch + FW_GRAN; j_input_ch++)
                    {
                      //loop through the channels of a single pixel to generate the index to the kernel pool   

                      input_tmp = input_data[(in_row * input_x + in_col) * input_ch + j_input_ch] + input_offset;
                      for (int i = 0; i < LUT_PREC; i++){
                        extracted_bit = GETBIT(input_tmp, i);
                        if(extracted_bit){
                          SETBIT(input_index[i], kernel_pixel_cnt);//set the bit if extracted is 1
                        }
                      }  
                      kernel_pixel_cnt++;
                    }
                    //copy the corresponding lut block of each bit from flash to ram
                    //The overhead of this memcopy will be compenstated by sharing it across all filters
                    for(int bit = 0; bit < LUT_PREC; bit++){
                      memcpy(lut_buffer + bit*LUT_SIZE, filter_pool_data + input_index[bit]*LUT_SIZE, LUT_SIZE);
                    }
                    //loop through different filters of the same channel to extract the results, so that the index can be shared
                    for (i_out_ch = 0; i_out_ch < output_ch; i_out_ch++)
                    {
                      partial_sum = 0; //reset the conv_out for each kernel
                      logical_kernel_idx = output_ch * block_cnt + i_out_ch;//get the current logical kernel index to find the corresponding physical kenrel in kernel pool
                      physical_kernel_idx = kernel_idx[logical_kernel_idx];
                      //Loop for bit-serial processing, total iteration is the precision used
                      //need to shift according to the bit precision as well
                      for(int bit = 0; bit < LUT_PREC; bit++)
                      {
                        result_idx = bit*LUT_SIZE + physical_kernel_idx;               
                        partial_sum += ((int16_t)(lut_buffer[result_idx])<<bit);
                      }
                      conv_out_holder[i_out_ch] += partial_sum; //accumulate the partial sums to the output result holder                  
                    }
                    block_cnt++;
                  }
                }
              }
              //write the results into output array, make it outside the channel loop so that it won't be repeated
              //or can writ e a condition inside the previous loop and only process the following code if its the last channel, but need to repeat the comparison every time.
              for (i_out_ch = 0; i_out_ch < output_ch; i_out_ch++) 
              {
                if (bias_data)
                {
                    conv_out_holder[i_out_ch] += bias_data[i_out_ch];
                }
                conv_out_holder[i_out_ch] = arm_nn_requantize(conv_out_holder[i_out_ch], output_mult[i_out_ch], output_shift[i_out_ch]);
                conv_out_holder[i_out_ch] += out_offset;
                conv_out_holder[i_out_ch] = MAX(conv_out_holder[i_out_ch], out_activation_min);
                conv_out_holder[i_out_ch] = MIN(conv_out_holder[i_out_ch], out_activation_max);
                output_data[i_out_ch + (i_out_y * output_x + i_out_x) * output_ch] = (int8_t)conv_out_holder[i_out_ch];
              }
          }
      }
      /* Advance to the next batch */
      input_data += (input_x * input_y * input_ch);
      output_data += (output_x * output_y * output_ch);
    }
    //free(conv_out_holder);

    /* Return to application */
    
    return ARM_MATH_SUCCESS;
}

arm_status lut_conv_zdim_v2_double_lookup(const cmsis_nn_context *ctx,
                           const cmsis_nn_conv_params *conv_params,
                           const cmsis_nn_per_channel_quant_params *quant_params,
                           const cmsis_nn_dims *input_dims,
                           const q7_t *input_data,
                           const cmsis_nn_dims *filter_dims,
                           const uint8_t* kernel_idx,
                           const cmsis_nn_dims *bias_dims,
                           const int32_t *bias_data,
                           const cmsis_nn_dims *output_dims,
                           const uint8_t* filter_pool_data,
                           q7_t *output_data)
{
  (void)bias_dims;
  q15_t *buffer_a = (q15_t *)ctx->buf;

  const uint16_t input_batches = input_dims->n;
  const uint16_t input_x = input_dims->w;
  const uint16_t input_y = input_dims->h;
  const uint16_t input_ch = input_dims->c;
  const uint16_t kernel_x = filter_dims->w;
  const uint16_t kernel_y = filter_dims->h;
  const uint16_t output_x = output_dims->w;
  const uint16_t output_y = output_dims->h;
  const uint16_t output_ch = output_dims->c;

  const uint16_t pad_x = conv_params->padding.w;
  const uint16_t pad_y = conv_params->padding.h;
  const uint16_t stride_x = conv_params->stride.w;
  const uint16_t stride_y = conv_params->stride.h;

  const int32_t input_offset = conv_params->input_offset;
  const int32_t out_offset = conv_params->output_offset;
  const int32_t out_activation_min = conv_params->activation.min;
  const int32_t out_activation_max = conv_params->activation.max;
  int32_t *output_mult = quant_params->multiplier;
  int32_t *output_shift = quant_params->shift;
  char* result_ptr;
  char input_tmp, extracted_bit;

  uint16_t input_index[LUT_PREC] = {0};
  int16_t* conv_out_holder = malloc(output_ch*sizeof(int16_t));
  //static int16_t conv_out_holder[128];

  int i_batch;
  for (i_batch = 0; i_batch < input_batches; i_batch++)
  {
      /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
      (void)buffer_a;
      int32_t i_out_ch, i_out_y, i_out_x, i_input_ch, i_ker_y, i_ker_x, j_input_ch, i_phy_ft;
      int32_t conv_out, logical_kernel_idx, result_idx, block_cnt;//block_cnt is a counter to count the number of 8-wide weight blocks processed
      int16_t partial_sum;
      uint8_t physical_kernel_idx;

      for (i_out_y = 0; i_out_y < output_y; i_out_y++)
      {
          for (i_out_x = 0; i_out_x < output_x; i_out_x++)
          {   
              block_cnt = 0;//initialize and reset the block counter
              //conv_out = 0;
              //int16_t conv_out_holder[output_ch]; //initialize the conv result holder, one for each filter
              memset(conv_out_holder, 0, output_ch*sizeof(int16_t) );//set the conv out holder to zero for accurate accumulation

              const int32_t base_idx_y = stride_y * i_out_y - pad_y;
              const int32_t base_idx_x = stride_x * i_out_x - pad_x;

              const int32_t ker_y_start = MAX(0, -base_idx_y);
              const int32_t ker_x_start = MAX(0, -base_idx_x); 

              const int32_t ker_y_end = MIN(kernel_y, input_y - base_idx_y);
              const int32_t ker_x_end = MIN(kernel_x, input_x - base_idx_x);

              uint8_t lut_buffer[LUT_PREC*LUT_SIZE];

              //loop through channels first, so that the input index can be reused among different filters to hide the index generation overhead
              //for (i_input_ch = 0; i_input_ch < input_ch; i_input_ch++)
              for (i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
              {
                for (i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                {
                  const int32_t in_row = base_idx_y + i_ker_y; 
                  const int32_t in_col = base_idx_x + i_ker_x;

                  for (i_input_ch = 0; i_input_ch < input_ch; i_input_ch = i_input_ch + FW_GRAN)
                  {
                    int kernel_pixel_cnt = 0;
                    //index generation, outer loop is for input pixels, and inner loop is for 8 index
                    uint16_t result_mem[LUT_SIZE] = {0};//array to hold temporary filter results
                    for (j_input_ch = i_input_ch; j_input_ch < i_input_ch + FW_GRAN; j_input_ch++)
                    {
                      //loop through the channels of a single pixel to generate the index to the kernel pool   

                      input_tmp = input_data[(in_row * input_x + in_col) * input_ch + j_input_ch] + input_offset;
                      for (int i = 0; i < LUT_PREC; i++){
                        extracted_bit = GETBIT(input_tmp, i);
                        if(extracted_bit){
                          SETBIT(input_index[i], kernel_pixel_cnt);//set the bit if extracted is 1
                        }
                      }  
                      kernel_pixel_cnt++;
                    }
                    //copy the corresponding lut block of each bit from flash to ram
                    //For the memorization version, the overhead of this memcpy cannot be shared across all filters, but is shared across all physical filters (LUT size)
                    for(int bit = 0; bit < LUT_PREC; bit++){ 

                      memcpy(lut_buffer + bit*LUT_SIZE, filter_pool_data + input_index[bit]*LUT_SIZE, LUT_SIZE);
                    }

                    for (i_phy_ft = 0; i_phy_ft < LUT_EFFECTIVE_SIZE; i_phy_ft++){
                      partial_sum = 0; //reset the conv_out for each physical kernel
                      for(int bit = 0; bit < LUT_PREC; bit++)
                      {
                        result_idx = bit*LUT_EFFECTIVE_SIZE + i_phy_ft;               
                        partial_sum += ((int16_t)(lut_buffer[result_idx])<<bit);
                      }
                      result_mem[i_phy_ft] = partial_sum;
                    }

                    //loop through different filters of the same channel to extract the results, so that the index can be shared
                    //for large layers where n_channel >> LUT_size, this loop may start to domionate runtime
                    //kernel_idx are still directly read from FLASH, which can be further optimized by pre-fectching 
                    for (i_out_ch = 0; i_out_ch < output_ch; i_out_ch++)
                    {
                      partial_sum = 0; //reset the conv_out for each kernel
                      logical_kernel_idx = output_ch * block_cnt + i_out_ch;//get the current logical kernel index to find the corresponding physical kenrel in kenrel pool
                      physical_kernel_idx = kernel_idx[logical_kernel_idx];
                      //This filter has already been computed, read the result from buffer, multiplied by coefficients
                      conv_out_holder[i_out_ch] += result_mem[physical_kernel_idx];                
                    }
                    block_cnt++;
                  }
                }
              }
              //write the results into output array, make it outside the channel loop so that it won't be repeated
              //or can writ e a condition inside the previous loop and only process the following code if its the last channel, but need to repeat the comparison every time.
              for (i_out_ch = 0; i_out_ch < output_ch; i_out_ch++) 
              {
                if (bias_data)
                {
                  conv_out_holder[i_out_ch] += bias_data[i_out_ch];
                }
                conv_out_holder[i_out_ch] = arm_nn_requantize(conv_out_holder[i_out_ch], output_mult[i_out_ch], output_shift[i_out_ch]);
                conv_out_holder[i_out_ch] += out_offset;
                conv_out_holder[i_out_ch] = MAX(conv_out_holder[i_out_ch], out_activation_min);
                conv_out_holder[i_out_ch] = MIN(conv_out_holder[i_out_ch], out_activation_max);
                output_data[i_out_ch + (i_out_y * output_x + i_out_x) * output_ch] = (int8_t)conv_out_holder[i_out_ch];
              }
          }
      }
      /* Advance to the next batch */
      input_data += (input_x * input_y * input_ch);
      output_data += (output_x * output_y * output_ch);
    }
    //free(conv_out_holder);

    /* Return to application */
    return ARM_MATH_SUCCESS;
}



arm_status lut_conv_zdim_v5_fusedpooling(const cmsis_nn_context *ctx,
                           const cmsis_nn_conv_params *conv_params,
                           const cmsis_nn_per_channel_quant_params *quant_params,
                           const cmsis_nn_dims *input_dims,
                           const q7_t *input_data,
                           const cmsis_nn_dims *filter_dims,
                           const uint8_t* kernel_idx,
                           const cmsis_nn_dims *bias_dims,
                           const int32_t *bias_data,
                           const cmsis_nn_dims *output_dims,
                           const uint8_t* filter_pool_data,
                           q7_t *output_data)
{
  //this version implements fused batchnorm, relu and pooling, so that the activation memory can be reduced. (especially for first layer)
  (void)bias_dims;
  q15_t *buffer_a = (q15_t *)ctx->buf;

  const uint16_t input_batches = input_dims->n;
  const uint16_t input_x = input_dims->w;
  const uint16_t input_y = input_dims->h;
  const uint16_t input_ch = input_dims->c;
  const uint16_t kernel_x = filter_dims->w;
  const uint16_t kernel_y = filter_dims->h;
  const uint16_t output_x = output_dims->w;
  const uint16_t output_y = output_dims->h;
  const uint16_t output_ch = output_dims->c;

  const uint16_t pad_x = conv_params->padding.w;
  const uint16_t pad_y = conv_params->padding.h;
  const uint16_t stride_x = conv_params->stride.w;
  const uint16_t stride_y = conv_params->stride.h;

  const int32_t input_offset = conv_params->input_offset;
  const int32_t out_offset = conv_params->output_offset;
  const int32_t out_activation_min = conv_params->activation.min;
  const int32_t out_activation_max = conv_params->activation.max;
  int32_t *output_mult = quant_params->multiplier;
  int32_t *output_shift = quant_params->shift;
  char* result_ptr;
  char input_tmp, extracted_bit;

  uint16_t input_index[LUT_PREC] = {0};
  int16_t* conv_out_holder = malloc(output_ch*sizeof(int16_t));
  //static int16_t conv_out_holder[128];

  int i_batch;
  for (i_batch = 0; i_batch < input_batches; i_batch++)
  {
      /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
      (void)buffer_a;
      int32_t i_out_ch, i_out_y, i_out_x, i_input_ch, i_ker_y, i_ker_x, j_input_ch;
      int32_t conv_out, logical_kernel_idx, result_idx, block_cnt;//block_cnt is a counter to count the number of 8-wide weight blocks processed
      int16_t partial_sum;
      uint8_t physical_kernel_idx;

      for (i_out_y = 0; i_out_y < output_y; i_out_y++)
      {
          for (i_out_x = 0; i_out_x < output_x; i_out_x++)
          {   
              block_cnt = 0;//initialize and reset the block counter
              //conv_out = 0;
              //int16_t conv_out_holder[output_ch]; //initialize the conv result holder, one for each filter
              memset(conv_out_holder, 0, output_ch*sizeof(int16_t));//set the conv out holder to zero for accurate accumulation

              const int32_t base_idx_y = stride_y * i_out_y - pad_y;
              const int32_t base_idx_x = stride_x * i_out_x - pad_x;

              const int32_t ker_y_start = MAX(0, -base_idx_y);
              const int32_t ker_x_start = MAX(0, -base_idx_x); 

              const int32_t ker_y_end = MIN(kernel_y, input_y - base_idx_y);
              const int32_t ker_x_end = MIN(kernel_x, input_x - base_idx_x);

              uint8_t lut_buffer[LUT_PREC*LUT_SIZE];

              //for (i_input_ch = 0; i_input_ch < input_ch; i_input_ch++)
              for (i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
              {
                for (i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                {
                  const int32_t in_row = base_idx_y + i_ker_y; 
                  const int32_t in_col = base_idx_x + i_ker_x;

                  for (i_input_ch = 0; i_input_ch < input_ch; i_input_ch = i_input_ch + FW_GRAN)
                  {
                    int kernel_pixel_cnt = 0;
                    //index generation, outer loop is for input pixels, and inner loop is for 8 index
                    
                    for (j_input_ch = i_input_ch; j_input_ch < i_input_ch + FW_GRAN; j_input_ch++)
                    {
                      //loop through the channels of a single pixel to generate the index to the kernel pool   

                      input_tmp = input_data[(in_row * input_x + in_col) * input_ch + j_input_ch] + input_offset;
                      for (int i = 0; i < LUT_PREC; i++){
                        extracted_bit = GETBIT(input_tmp, i);
                        if(extracted_bit){
                          SETBIT(input_index[i], kernel_pixel_cnt);//set the bit if extracted is 1
                        }
                      }  
                      kernel_pixel_cnt++;
                    }
                    //copy the corresponding lut block of each bit from flash to ram
                    //The overhead of this memcopy will be compenstated by sharing it across all filters
                    for(int bit = 0; bit < LUT_PREC; bit++){
                      memcpy(lut_buffer + bit*LUT_SIZE, filter_pool_data + input_index[bit]*LUT_SIZE, LUT_SIZE);
                    }
                    //loop through different filters of the same channel to extract the results, so that the index can be shared
                    for (i_out_ch = 0; i_out_ch < output_ch; i_out_ch++)
                    {
                      partial_sum = 0; //reset the conv_out for each kernel
                      logical_kernel_idx = output_ch * block_cnt + i_out_ch;//get the current logical kernel index to find the corresponding physical kenrel in kernel pool
                      physical_kernel_idx = kernel_idx[logical_kernel_idx];
                      //Loop for bit-serial processing, total iteration is the precision used
                      //need to shift according to the bit precision as well
                      for(int bit = 0; bit < LUT_PREC; bit++)
                      {
                        result_idx = bit*LUT_SIZE + physical_kernel_idx;               
                        partial_sum += ((int16_t)(lut_buffer[result_idx])<<bit);
                      }
                      conv_out_holder[i_out_ch] += partial_sum; //accumulate the partial sums to the output result holder                  
                    }
                    block_cnt++;
                  }
                }
              }
              //write the results into output array, make it outside the channel loop so that it won't be repeated
              //or can writ e a condition inside the previous loop and only process the following code if its the last channel, but need to repeat the comparison every time.
              for (i_out_ch = 0; i_out_ch < output_ch; i_out_ch++) 
              {
                if (bias_data)
                {
                    conv_out_holder[i_out_ch] += bias_data[i_out_ch];
                }
                conv_out_holder[i_out_ch] = arm_nn_requantize(conv_out_holder[i_out_ch], output_mult[i_out_ch], output_shift[i_out_ch]);
                conv_out_holder[i_out_ch] += out_offset;
                conv_out_holder[i_out_ch] = MAX(conv_out_holder[i_out_ch], out_activation_min);
                conv_out_holder[i_out_ch] = MIN(conv_out_holder[i_out_ch], out_activation_max);
                output_data[i_out_ch + (i_out_y * output_x + i_out_x) * output_ch] = (int8_t)conv_out_holder[i_out_ch];
              }
          }
      }
      /* Advance to the next batch */
      input_data += (input_x * input_y * input_ch);
      output_data += (output_x * output_y * output_ch);
    }
    //free(conv_out_holder);


    /* Return to application */
    
    return ARM_MATH_SUCCESS;
}