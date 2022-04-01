#include <stdlib.h>
#include "lut_utiles.h"
#include <math.h>
#include "D:\research\fixedweightNN\CMSIS_5-develop\CMSIS_5-develop\CMSIS\NN\Tests\UnitTest\TestCases\test_arm_convolve_s8\lut_data.h"

#include "D:\research\fixedweightNN\CMSIS_NN\Include\arm_nnfunctions.h"
//#include <unity.h>

#include "../Utils/validate.h"
#include "D:\research\fixedweightNN\CMSIS_5-develop\CMSIS_5-develop\CMSIS\NN\Tests\full_network\conv_full_layer1\test_data.h"
#include "D:\research\fixedweightNN\CMSIS_5-develop\CMSIS_5-develop\CMSIS\NN\Tests\full_network\conv_full_layer2\test_data.h"
#include "D:\research\fixedweightNN\CMSIS_5-develop\CMSIS_5-develop\CMSIS\NN\Tests\full_network\conv_full_layer3\test_data.h"
#include "D:\research\fixedweightNN\CMSIS_5-develop\CMSIS_5-develop\CMSIS\NN\Tests\full_network\conv_full_layer4\test_data.h"
#include "D:\research\fixedweightNN\CMSIS_5-develop\CMSIS_5-develop\CMSIS\NN\Tests\full_network\fc_full_layer1\test_data.h"

#include "D:\research\fixedweightNN\CMSIS_5-develop\CMSIS_5-develop\index_data\full_network_coeff_data.h"
#include "D:\research\fixedweightNN\CMSIS_5-develop\CMSIS_5-develop\index_data\full_network_index_data.h"



void conv_s8_full_network(void)
{

    const arm_status expected = ARM_MATH_SUCCESS;
    //declare the input and output buffer in RAM, both should be the largest amoung all layers
    static q7_t buffer_1[CONV_FULL_LAYER1_DST_SIZE] __attribute__((section("flash"))); 
    static q7_t buffer_2[CONV_FULL_LAYER1_DST_SIZE] __attribute__((section("flash")));
    //buffer_1 = {0};
    //buffer_2 = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims_conv1, input_dims_conv2, input_dims_conv3, input_dims_conv4, input_dims_fc1;
    cmsis_nn_dims filter_dims_conv1, filter_dims_conv2, filter_dims_conv3, filter_dims_conv4, filter_dims_fc1;
    cmsis_nn_dims bias_dims_conv1, bias_dims_conv2, bias_dims_conv3, bias_dims_conv4, bias_dims_fc1;
    cmsis_nn_dims output_dims_conv1, output_dims_conv2, output_dims_conv3, output_dims_conv4, output_dims_fc1;

    const q31_t *bias_data_conv1 = conv_full_layer1_biases;
    const q7_t *kernel_data_conv1 = conv_full_layer1_weights;
    const q7_t *input_data_conv1 = conv_full_layer1_input;

    input_dims_conv1.n = CONV_FULL_LAYER1_INPUT_BATCHES;
    input_dims_conv1.w = CONV_FULL_LAYER1_INPUT_W;
    input_dims_conv1.h = CONV_FULL_LAYER1_INPUT_H;
    input_dims_conv1.c = CONV_FULL_LAYER1_IN_CH;
    filter_dims_conv1.w = CONV_FULL_LAYER1_FILTER_X;
    filter_dims_conv1.h = CONV_FULL_LAYER1_FILTER_Y;
    output_dims_conv1.w = CONV_FULL_LAYER1_OUTPUT_W;
    output_dims_conv1.h = CONV_FULL_LAYER1_OUTPUT_H;
    output_dims_conv1.c = CONV_FULL_LAYER1_OUT_CH;

    const q31_t *bias_data_conv2 = conv_full_layer2_biases;
    const q7_t *kernel_data_conv2 = conv_full_layer2_weights;

    input_dims_conv2.n = CONV_FULL_LAYER2_INPUT_BATCHES;
    input_dims_conv2.w = CONV_FULL_LAYER2_INPUT_W;
    input_dims_conv2.h = CONV_FULL_LAYER2_INPUT_H;
    input_dims_conv2.c = CONV_FULL_LAYER2_IN_CH;
    filter_dims_conv2.w = CONV_FULL_LAYER2_FILTER_X;
    filter_dims_conv2.h = CONV_FULL_LAYER2_FILTER_Y;
    output_dims_conv2.w = CONV_FULL_LAYER2_OUTPUT_W;
    output_dims_conv2.h = CONV_FULL_LAYER2_OUTPUT_H;
    output_dims_conv2.c = CONV_FULL_LAYER2_OUT_CH;    

    const q31_t *bias_data_conv3 = conv_full_layer3_biases;
    const q7_t *kernel_data_conv3 = conv_full_layer3_weights;

    input_dims_conv3.n = CONV_FULL_LAYER3_INPUT_BATCHES;
    input_dims_conv3.w = CONV_FULL_LAYER3_INPUT_W;
    input_dims_conv3.h = CONV_FULL_LAYER3_INPUT_H;
    input_dims_conv3.c = CONV_FULL_LAYER3_IN_CH;
    filter_dims_conv3.w = CONV_FULL_LAYER3_FILTER_X;
    filter_dims_conv3.h = CONV_FULL_LAYER3_FILTER_Y;
    output_dims_conv3.w = CONV_FULL_LAYER3_OUTPUT_W;
    output_dims_conv3.h = CONV_FULL_LAYER3_OUTPUT_H;
    output_dims_conv3.c = CONV_FULL_LAYER3_OUT_CH;  

    const q31_t *bias_data_conv4 = conv_full_layer4_biases;
    const q7_t *kernel_data_conv4 = conv_full_layer4_weights;

    input_dims_conv4.n = CONV_FULL_LAYER4_INPUT_BATCHES;
    input_dims_conv4.w = CONV_FULL_LAYER4_INPUT_W;
    input_dims_conv4.h = CONV_FULL_LAYER4_INPUT_H;
    input_dims_conv4.c = CONV_FULL_LAYER4_IN_CH;
    filter_dims_conv4.w = CONV_FULL_LAYER4_FILTER_X;
    filter_dims_conv4.h = CONV_FULL_LAYER4_FILTER_Y;
    output_dims_conv4.w = CONV_FULL_LAYER4_OUTPUT_W;
    output_dims_conv4.h = CONV_FULL_LAYER4_OUTPUT_H;
    output_dims_conv4.c = CONV_FULL_LAYER4_OUT_CH;  


    //Folowing parameters are shared across the layers, they are same
    conv_params.padding.w = CONV_FULL_LAYER1_PAD_X;
    conv_params.padding.h = CONV_FULL_LAYER1_PAD_Y;
    conv_params.stride.w = CONV_FULL_LAYER1_STRIDE_X;
    conv_params.stride.h = CONV_FULL_LAYER1_STRIDE_Y;

    conv_params.input_offset = CONV_FULL_LAYER1_INPUT_OFFSET;
    conv_params.output_offset = CONV_FULL_LAYER1_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_FULL_LAYER1_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_FULL_LAYER1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_full_layer1_output_mult;
    quant_params.shift = (int32_t *)conv_full_layer1_output_shift;

    arm_status result_arm;

    //start the first layer
    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv1, &filter_dims_conv1);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = lut_conv_v3_coeff_lutreorder(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv1,
                                        input_data_conv1,
                                        &filter_dims_conv1,
                                        full_network_index_layer_1,
                                        &bias_dims_conv1,
                                        bias_data_conv1,
                                        &output_dims_conv1,
                                        lut_data,
                                        full_network_coeff_layer_1,
                                        buffer_1);

    free(ctx.buf);

    //do pooling and move to next layer
    //copy the output buffer to the input buffer, memcpy(source, destination)
    //memcpy(buff, output_buffer, sizeof(output_buffer));

    arm_max_pool_22(&ctx, &output_dims_conv1, buffer_1, &input_dims_conv2, buffer_2);
    //results now stored in buffer 2

    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv2, &filter_dims_conv2);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = lut_conv_v3_coeff_lutreorder(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv2,
                                        buffer_2,
                                        &filter_dims_conv2,
                                        full_network_index_layer_2,
                                        &bias_dims_conv2,
                                        bias_data_conv2,
                                        &output_dims_conv2,
                                        lut_data,
                                        full_network_coeff_layer_2,
                                        buffer_1);

    free(ctx.buf);
    //move to next layer
    arm_max_pool_22(&ctx, &output_dims_conv2, buffer_1, &input_dims_conv3, buffer_2);

    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv3, &filter_dims_conv3);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = lut_conv_v3_coeff_lutreorder(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv3,
                                        buffer_2,
                                        &filter_dims_conv3,
                                        full_network_index_layer_3,
                                        &bias_dims_conv3,
                                        bias_data_conv3,
                                        &output_dims_conv3,
                                        lut_data,
                                        full_network_coeff_layer_3,
                                        buffer_1);

    free(ctx.buf);

    //move to next layer
    arm_max_pool_22(&ctx, &output_dims_conv3, buffer_1, &input_dims_conv4, buffer_2);

    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv4, &filter_dims_conv4);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = lut_conv_v3_coeff_lutreorder(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv4, 
                                        buffer_2,
                                        &filter_dims_conv4,
                                        full_network_index_layer_4,
                                        &bias_dims_conv4,
                                        bias_data_conv4,
                                        &output_dims_conv4,
                                        lut_data,
                                        full_network_coeff_layer_4,
                                        buffer_1);

    free(ctx.buf);

    //All convolution layers finished, the results are stored in buffer_1 (in theory)
}


int main(){
  while(1){
    
    conv_s8_full_network();
    
  }
return 0;
}