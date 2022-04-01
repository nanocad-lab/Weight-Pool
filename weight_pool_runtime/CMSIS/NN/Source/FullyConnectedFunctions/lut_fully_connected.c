#define LUT_PREC 8

#define LUT_SIZE 32
#define BLOCK_SIZE 8 //how many weights are fixed

#define GETBIT(var, bit)	(((var) >> (bit)) & 1)
#define SETBIT(var, bit)	var |= (1 << (bit))

//#include "arm_nnfunctions.h"
//#include "arm_nnsupportfunctions.h"
#include "D:\research\fixedweightNN\CMSIS_5-develop\CMSIS_5-develop\CMSIS\NN\Include\arm_nnfunctions.h"
#include "D:\research\fixedweightNN\CMSIS_5-develop\CMSIS_5-develop\CMSIS\NN\Include\arm_nnsupportfunctions.h"


/*
Baseline version of lookup table and fixed weight version of fully connected layer
An issue is since baseline CMSIS doesn't using the regular loop-style implementation,
integrate requantize and bias/offset function might be an issue
 */

arm_status lut_fully_connected_v1_withcoeff(const cmsis_nn_context *ctx,
                                  const cmsis_nn_fc_params *fc_params,
                                  const cmsis_nn_per_tensor_quant_params *quant_params,
                                  const cmsis_nn_dims *input_dims,
                                  const q7_t *input,
                                  const cmsis_nn_dims *filter_dims,
                                  const q7_t *weight_pool_idx,
                                  const cmsis_nn_dims *bias_dims,
                                  const int32_t *bias,
                                  const cmsis_nn_dims *output_dims,
                                  const uint8_t *weight_pool_data,
                                  const uint8_t *coeffs,
                                  q7_t *output)
{
    (void)bias_dims;
    (void)ctx;
    (void)fc_params->filter_offset;

    int32_t batch_cnt = input_dims->n;
    uint8_t input_tmp, extracted_bit,physical_kernel_idx;
    uint16_t input_index[LUT_PREC] = {0};
    uint16_t partial_sum, blk_idx, result_idx;
    int logical_kernel_idx;

    while (batch_cnt)
    {
        int i_neurons,i_weights,i_inputs,i_blkelements;
        int16_t partial_sum_accumulator[output_dims->c]; //initialize the conv result holder, one for each filter
        memset( partial_sum_accumulator, 0, (output_dims->c)*sizeof(int16_t) );//set the conv out holder to zero for accurate accumulation
        //first iterate through input blocks
        //check whether filter_dims-> is the input size using simulator
        blk_idx = 0;
        for(i_inputs = 0; i_inputs < filter_dims->n; i_inputs+=BLOCK_SIZE)
        {
            for(i_blkelements = 0; i_blkelements<BLOCK_SIZE; i_blkelements++)
            {
                //generate the input index
                input_tmp = input[i_inputs+i_blkelements] + fc_params->input_offset;
                for (int i = 0; i < LUT_PREC; i++)
                {
                  extracted_bit = GETBIT(input_tmp, i);
                  if(extracted_bit){
                    SETBIT(input_index[i], i_blkelements);//set the bit if extracted is 1
                  }
                } 
            }
            //Then iterate over neurons to do the result lookup
            for (i_neurons = 0; i_neurons < output_dims->c; i_neurons++){
                partial_sum = 0;
                logical_kernel_idx = i_neurons*filter_dims->n + blk_idx;
                physical_kernel_idx = weight_pool_idx[logical_kernel_idx];
                //THen iterate over bits for bit-setial processing
                for(int bit = 0; bit < LUT_PREC; bit++)
                {
                    result_idx = physical_kernel_idx + input_index[bit];
                    partial_sum += ((int16_t)(weight_pool_data[result_idx])<<bit);
                }
                partial_sum = partial_sum * coeffs[logical_kernel_idx];               
                partial_sum_accumulator[i_neurons] += partial_sum;
            }
            blk_idx++;//increse the block index by 1
        }
        //Then iterate over the neurons to process the final results and write back to output array
        for (i_neurons = 0; i_neurons < output_dims->c; i_neurons++)
        {   
            int16_t tmp_output = partial_sum_accumulator[i_neurons];
            if (bias)
            {
                tmp_output += bias[i_neurons];
            }
            tmp_output = arm_nn_requantize(tmp_output, quant_params->multiplier, quant_params->shift);
            tmp_output += fc_params->output_offset;
            tmp_output = MAX(tmp_output, fc_params->activation.min);
            tmp_output = MIN(tmp_output, fc_params->activation.max);
            output[i_neurons] = tmp_output;
        }
        //Move to the next batch
        //be careful here about proper array indexing
        input += filter_dims->n;
        output += output_dims->c;
        batch_cnt--;      
    }
    return (ARM_MATH_SUCCESS);
}

/**
 * @} end of FC group
 */
