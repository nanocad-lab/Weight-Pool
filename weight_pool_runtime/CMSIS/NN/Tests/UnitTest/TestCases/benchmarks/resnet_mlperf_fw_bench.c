#include "../Utils/validate.h"
#include "..\..\..\..\Include\arm_nnfunctions.h"
#include "..\..\..\..\..\..\lut_zdim64_data.h"
#include "..\..\..\..\..\..\TestData_fullnetwork\resnet_mlperf\test_data.h"
#include "..\..\..\..\..\..\index_data\resnet_mlperf_data\resnet_mlperf_index_data.h"
#include "..\..\..\..\..\..\STM32F2xx_HAL_Driver\Inc\stm32f2xx_hal.h" //include file for clock generation code
#include "lut_utiles.h"
#include <math.h>
#include <stdlib.h>


void SysTick_Handler(void)
{
  /* USER CODE BEGIN SysTick_IRQn 0 */

  /* USER CODE END SysTick_IRQn 0 */
  HAL_IncTick();
  /* USER CODE BEGIN SysTick_IRQn 1 */

  /* USER CODE END SysTick_IRQn 1 */
}

void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 13;
  RCC_OscInitStruct.PLL.PLLN = 195;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 5;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    while (1){}
  }
  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK)
  {
    while (1){}
  }
}

void conv_fw_resnet_mlperf(void)
{
    //Todo: Implement first layer using CMSIS (for filter pool benchmark), and load input from flash to ram first before start of first layer so input is always loaded from flash
    const arm_status expected = ARM_MATH_SUCCESS;
    /*for large networks like resnet, first layer's ouptut size will be 64KB, and is a lot larger than all subsequent layers, which is not necessery,
    and will cause ram issues if using double buffering for this size, so create a dymamic array to hold the output of first layer, and destroy it after second layer
    First layer's input can still be put into RAM since first layer only contains 3 channels, not 64, and requires way less memory than the output. 
    */

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims_conv1, input_dims_conv2, input_dims_conv3, input_dims_conv4, input_dims_conv5,
        input_dims_conv6, input_dims_conv7, input_dims_conv8, input_dims_conv9, input_dims_conv10, input_dims_conv11, input_dims_conv12, input_dims_conv13, input_dims_fc1;
    cmsis_nn_dims filter_dims_conv1, filter_dims_conv2, filter_dims_conv3, filter_dims_conv4, filter_dims_conv5,
        filter_dims_conv6, filter_dims_conv7, filter_dims_conv8, filter_dims_conv9, filter_dims_conv10, filter_dims_conv11, filter_dims_conv12, filter_dims_conv13, filter_dims_fc1;
    cmsis_nn_dims bias_dims_conv1, bias_dims_conv2, bias_dims_conv3, bias_dims_conv4, bias_dims_conv5, bias_dims_conv6,
        bias_dims_conv7, bias_dims_conv8, bias_dims_conv9, bias_dims_conv10, bias_dims_conv11, bias_dims_conv12, bias_dims_conv13, bias_dims_fc1;
    cmsis_nn_dims output_dims_conv1, output_dims_conv2, output_dims_conv3, output_dims_conv4, output_dims_conv5,
        output_dims_conv6, output_dims_conv7, output_dims_conv8, output_dims_conv9, output_dims_conv10, output_dims_conv11, output_dims_conv12, output_dims_conv13, output_dims_fc1;

    const q7_t *input_data_conv1 = convlayer1_input;
    const q31_t *bias_data_conv1 = convlayer1_biases;
    const q7_t *kernel_data_conv1 = convlayer1_weights;

    input_dims_conv1.n = CONVLAYER1_INPUT_BATCHES;
    input_dims_conv1.w = CONVLAYER1_INPUT_W;
    input_dims_conv1.h = CONVLAYER1_INPUT_H;
    input_dims_conv1.c = CONVLAYER1_IN_CH;
    filter_dims_conv1.w = CONVLAYER1_FILTER_X;
    filter_dims_conv1.h = CONVLAYER1_FILTER_Y;
    output_dims_conv1.w = CONVLAYER1_OUTPUT_W;
    output_dims_conv1.h = CONVLAYER1_OUTPUT_H;
    output_dims_conv1.c = CONVLAYER1_OUT_CH;

    const q31_t *bias_data_conv2 = convlayer2_biases;
    const q7_t *kernel_data_conv2 = convlayer2_weights;

    input_dims_conv2.n = CONVLAYER2_INPUT_BATCHES;
    input_dims_conv2.w = CONVLAYER2_INPUT_W;
    input_dims_conv2.h = CONVLAYER2_INPUT_H;
    input_dims_conv2.c = CONVLAYER2_IN_CH;
    filter_dims_conv2.w = CONVLAYER2_FILTER_X;
    filter_dims_conv2.h = CONVLAYER2_FILTER_Y;
    output_dims_conv2.w = CONVLAYER2_OUTPUT_W;
    output_dims_conv2.h = CONVLAYER2_OUTPUT_H;
    output_dims_conv2.c = CONVLAYER2_OUT_CH;
    
    const q31_t *bias_data_conv3 = convlayer3_biases;
    const q7_t *kernel_data_conv3 = convlayer3_weights;

    input_dims_conv3.n = CONVLAYER3_INPUT_BATCHES;
    input_dims_conv3.w = CONVLAYER3_INPUT_W;
    input_dims_conv3.h = CONVLAYER3_INPUT_H;
    input_dims_conv3.c = CONVLAYER3_IN_CH;
    filter_dims_conv3.w = CONVLAYER3_FILTER_X;
    filter_dims_conv3.h = CONVLAYER3_FILTER_Y;
    output_dims_conv3.w = CONVLAYER3_OUTPUT_W;
    output_dims_conv3.h = CONVLAYER3_OUTPUT_H;
    output_dims_conv3.c = CONVLAYER3_OUT_CH;
    
    const q31_t *bias_data_conv4 = convlayer4_biases;
    const q7_t *kernel_data_conv4 = convlayer4_weights;

    input_dims_conv4.n = CONVLAYER4_INPUT_BATCHES;
    input_dims_conv4.w = CONVLAYER4_INPUT_W;
    input_dims_conv4.h = CONVLAYER4_INPUT_H;
    input_dims_conv4.c = CONVLAYER4_IN_CH;
    filter_dims_conv4.w = CONVLAYER4_FILTER_X;
    filter_dims_conv4.h = CONVLAYER4_FILTER_Y;
    output_dims_conv4.w = CONVLAYER4_OUTPUT_W;
    output_dims_conv4.h = CONVLAYER4_OUTPUT_H;
    output_dims_conv4.c = CONVLAYER4_OUT_CH;
    
    const q31_t *bias_data_conv5 = convlayer5_biases;
    const q7_t *kernel_data_conv5 = convlayer5_weights;

    input_dims_conv5.n = CONVLAYER5_INPUT_BATCHES;
    input_dims_conv5.w = CONVLAYER5_INPUT_W;
    input_dims_conv5.h = CONVLAYER5_INPUT_H;
    input_dims_conv5.c = CONVLAYER5_IN_CH;
    filter_dims_conv5.w = CONVLAYER5_FILTER_X;
    filter_dims_conv5.h = CONVLAYER5_FILTER_Y;
    output_dims_conv5.w = CONVLAYER5_OUTPUT_W;
    output_dims_conv5.h = CONVLAYER5_OUTPUT_H;
    output_dims_conv5.c = CONVLAYER5_OUT_CH;
    
    const q31_t *bias_data_conv6 = convlayer6_biases;
    const q7_t *kernel_data_conv6 = convlayer6_weights;

    input_dims_conv6.n = CONVLAYER6_INPUT_BATCHES;
    input_dims_conv6.w = CONVLAYER6_INPUT_W;
    input_dims_conv6.h = CONVLAYER6_INPUT_H;
    input_dims_conv6.c = CONVLAYER6_IN_CH;
    filter_dims_conv6.w = CONVLAYER6_FILTER_X;
    filter_dims_conv6.h = CONVLAYER6_FILTER_Y;
    output_dims_conv6.w = CONVLAYER6_OUTPUT_W;
    output_dims_conv6.h = CONVLAYER6_OUTPUT_H;
    output_dims_conv6.c = CONVLAYER6_OUT_CH;
    
    const q31_t *bias_data_conv7 = convlayer7_biases;
    const q7_t *kernel_data_conv7 = convlayer7_weights;

    input_dims_conv7.n = CONVLAYER7_INPUT_BATCHES;
    input_dims_conv7.w = CONVLAYER7_INPUT_W;
    input_dims_conv7.h = CONVLAYER7_INPUT_H;
    input_dims_conv7.c = CONVLAYER7_IN_CH;
    filter_dims_conv7.w = CONVLAYER7_FILTER_X;
    filter_dims_conv7.h = CONVLAYER7_FILTER_Y;
    output_dims_conv7.w = CONVLAYER7_OUTPUT_W;
    output_dims_conv7.h = CONVLAYER7_OUTPUT_H;
    output_dims_conv7.c = CONVLAYER7_OUT_CH;
    
    const q31_t *bias_data_conv8 = convlayer8_biases;
    const q7_t *kernel_data_conv8 = convlayer8_weights;

    input_dims_conv8.n = CONVLAYER8_INPUT_BATCHES;
    input_dims_conv8.w = CONVLAYER8_INPUT_W;
    input_dims_conv8.h = CONVLAYER8_INPUT_H;
    input_dims_conv8.c = CONVLAYER8_IN_CH;
    filter_dims_conv8.w = CONVLAYER8_FILTER_X;
    filter_dims_conv8.h = CONVLAYER8_FILTER_Y;
    output_dims_conv8.w = CONVLAYER8_OUTPUT_W;
    output_dims_conv8.h = CONVLAYER8_OUTPUT_H;
    output_dims_conv8.c = CONVLAYER8_OUT_CH;
    
    const q31_t *bias_data_conv9 = convlayer9_biases;
    const q7_t *kernel_data_conv9 = convlayer9_weights;

    input_dims_conv9.n = CONVLAYER9_INPUT_BATCHES;
    input_dims_conv9.w = CONVLAYER9_INPUT_W;
    input_dims_conv9.h = CONVLAYER9_INPUT_H;
    input_dims_conv9.c = CONVLAYER9_IN_CH;
    filter_dims_conv9.w = CONVLAYER9_FILTER_X;
    filter_dims_conv9.h = CONVLAYER9_FILTER_Y;
    output_dims_conv9.w = CONVLAYER9_OUTPUT_W;
    output_dims_conv9.h = CONVLAYER9_OUTPUT_H;
    output_dims_conv9.c = CONVLAYER9_OUT_CH;
    
    const q31_t *bias_data_conv10 = convlayer2_biases;
    const q7_t *kernel_data_conv10 = convlayer2_weights;

    input_dims_conv10.n = CONVLAYER10_INPUT_BATCHES;
    input_dims_conv10.w = CONVLAYER10_INPUT_W;
    input_dims_conv10.h = CONVLAYER10_INPUT_H;
    input_dims_conv10.c = CONVLAYER10_IN_CH;
    filter_dims_conv10.w = CONVLAYER10_FILTER_X;
    filter_dims_conv10.h = CONVLAYER10_FILTER_Y;
    output_dims_conv10.w = CONVLAYER10_OUTPUT_W;
    output_dims_conv10.h = CONVLAYER10_OUTPUT_H;
    output_dims_conv10.c = CONVLAYER10_OUT_CH;
    
    const q31_t *bias_data_conv11 = convlayer2_biases;
    const q7_t *kernel_data_conv11 = convlayer2_weights;

    input_dims_conv11.n = CONVLAYER11_INPUT_BATCHES;
    input_dims_conv11.w = CONVLAYER11_INPUT_W;
    input_dims_conv11.h = CONVLAYER11_INPUT_H;
    input_dims_conv11.c = CONVLAYER11_IN_CH;
    filter_dims_conv11.w = CONVLAYER11_FILTER_X;
    filter_dims_conv11.h = CONVLAYER11_FILTER_Y;
    output_dims_conv11.w = CONVLAYER11_OUTPUT_W;
    output_dims_conv11.h = CONVLAYER11_OUTPUT_H;
    output_dims_conv11.c = CONVLAYER11_OUT_CH;
    
    const q31_t *bias_data_conv12 = convlayer2_biases;
    const q7_t *kernel_data_conv12 = convlayer2_weights;

    input_dims_conv12.n = CONVLAYER12_INPUT_BATCHES;
    input_dims_conv12.w = CONVLAYER12_INPUT_W;
    input_dims_conv12.h = CONVLAYER12_INPUT_H;
    input_dims_conv12.c = CONVLAYER12_IN_CH;
    filter_dims_conv12.w = CONVLAYER12_FILTER_X;
    filter_dims_conv12.h = CONVLAYER12_FILTER_Y;
    output_dims_conv12.w = CONVLAYER12_OUTPUT_W;
    output_dims_conv12.h = CONVLAYER12_OUTPUT_H;
    output_dims_conv12.c = CONVLAYER12_OUT_CH;
    
    const q31_t *bias_data_conv13 = convlayer2_biases;
    const q7_t *kernel_data_conv13 = convlayer2_weights;

    input_dims_conv13.n = CONVLAYER13_INPUT_BATCHES;
    input_dims_conv13.w = CONVLAYER13_INPUT_W;
    input_dims_conv13.h = CONVLAYER13_INPUT_H;
    input_dims_conv13.c = CONVLAYER13_IN_CH;
    filter_dims_conv13.w = CONVLAYER13_FILTER_X;
    filter_dims_conv13.h = CONVLAYER13_FILTER_Y;
    output_dims_conv13.w = CONVLAYER13_OUTPUT_W;
    output_dims_conv13.h = CONVLAYER13_OUTPUT_H;
    output_dims_conv13.c = CONVLAYER13_OUT_CH;

    //Folowing parameters are shared across the layers, they are same
    conv_params.padding.w = CONVLAYER1_PAD_X;
    conv_params.padding.h = CONVLAYER1_PAD_Y;
    conv_params.stride.w = CONVLAYER1_STRIDE_X;
    conv_params.stride.h = CONVLAYER1_STRIDE_Y;

    conv_params.input_offset = CONVLAYER1_INPUT_OFFSET;
    conv_params.output_offset = CONVLAYER1_OUTPUT_OFFSET;
    conv_params.activation.min = CONVLAYER1_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONVLAYER1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)convlayer1_output_mult;
    quant_params.shift = (int32_t *)convlayer1_output_shift;

    q7_t* first_layer_outbuf = malloc(CONVLAYER1_OUT_CH*CONVLAYER1_OUTPUT_W*CONVLAYER1_OUTPUT_H*sizeof(q7_t));
    q7_t* first_layer_inbuf = malloc(CONVLAYER1_IN_CH*CONVLAYER1_INPUT_W*CONVLAYER1_INPUT_H*sizeof(q7_t));

    arm_status result_arm;

    //code for layers
    //First copy the inputs from flash to ram
    memcpy(first_layer_inbuf, input_data_conv1, CONVLAYER1_IN_CH*CONVLAYER1_INPUT_W*CONVLAYER1_INPUT_H);

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv1, &filter_dims_conv1);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = arm_convolve_s8(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv1,
                                        first_layer_inbuf,
                                        &filter_dims_conv1,
                                        kernel_data_conv1,
                                        &bias_dims_conv1,
                                        bias_data_conv1,
                                        &output_dims_conv1,
                                        first_layer_outbuf);

    free(ctx.buf);
    free(first_layer_inbuf); //destory the input data buffer for first layer and release the memory
    q7_t* actbuf1 = malloc(16*16*64); //create the first general activation buffer, 16*16*64 should be enough for rest layers

    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv2, &filter_dims_conv2);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v1(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv2,
                                        first_layer_outbuf,
                                        &filter_dims_conv2,
                                        resnet_mlperf_index_layer_2,
                                        &bias_dims_conv2,
                                        bias_data_conv2,
                                        &output_dims_conv2,
                                        lut_data,
                                        actbuf1);

    free(ctx.buf);
    free(first_layer_outbuf);//destroy the first layer output buffer and realse the memory
    q7_t* actbuf2 = malloc(16*16*64); //create the second general activation buffer, 16*16*64 should be enough for rest layers


    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv3, &filter_dims_conv3);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v1(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv3,
                                        actbuf1,
                                        &filter_dims_conv3,
                                        resnet_mlperf_index_layer_3,
                                        &bias_dims_conv3,
                                        bias_data_conv3,
                                        &output_dims_conv3,
                                        lut_data,
                                        actbuf2);

    free(ctx.buf);    

    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv4, &filter_dims_conv4);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v1(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv4,
                                        actbuf2,
                                        &filter_dims_conv4,
                                        resnet_mlperf_index_layer_4,
                                        &bias_dims_conv4,
                                        bias_data_conv4,
                                        &output_dims_conv4,
                                        lut_data,
                                        actbuf1);

    free(ctx.buf); 
    
    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv5, &filter_dims_conv5);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v1(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv5,
                                        actbuf1,
                                        &filter_dims_conv5,
                                        resnet_mlperf_index_layer_5,
                                        &bias_dims_conv5,
                                        bias_data_conv5,
                                        &output_dims_conv5,
                                        lut_data,
                                        actbuf2);

    free(ctx.buf); 
    
    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv6, &filter_dims_conv6);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v1(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv6,
                                        actbuf2,
                                        &filter_dims_conv6,
                                        resnet_mlperf_index_layer_6,
                                        &bias_dims_conv6,
                                        bias_data_conv6,
                                        &output_dims_conv6,
                                        lut_data,
                                        actbuf1);

    free(ctx.buf); 
    
    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv7, &filter_dims_conv7);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v1(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv7,
                                        actbuf1,
                                        &filter_dims_conv7,
                                        resnet_mlperf_index_layer_7,
                                        &bias_dims_conv7,
                                        bias_data_conv7,
                                        &output_dims_conv7,
                                        lut_data,
                                        actbuf2);

    free(ctx.buf); 
    
    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv8, &filter_dims_conv8);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v1(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv8,
                                        actbuf2,
                                        &filter_dims_conv8,
                                        resnet_mlperf_index_layer_8,
                                        &bias_dims_conv8,
                                        bias_data_conv8,
                                        &output_dims_conv8,
                                        lut_data,
                                        actbuf1);

    free(ctx.buf); 
    
    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv9, &filter_dims_conv9);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v1(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv9,
                                        actbuf1,
                                        &filter_dims_conv9,
                                        resnet_mlperf_index_layer_9,
                                        &bias_dims_conv9,
                                        bias_data_conv9,
                                        &output_dims_conv9,
                                        lut_data,
                                        actbuf2);

    free(ctx.buf); 

    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv10, &filter_dims_conv10);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    //result_arm = lut_conv_zdim_v1(&ctx,
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv10,
                                        actbuf2,
                                        &filter_dims_conv10,
                                        resnet_mlperf_index_layer_10,
                                        &bias_dims_conv10,
                                        bias_data_conv10,
                                        &output_dims_conv10,
                                        lut_data,
                                        actbuf1);

    free(ctx.buf); 
    
    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv11, &filter_dims_conv11);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    //result_arm = lut_conv_zdim_v1(&ctx,
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv11,
                                        actbuf1,
                                        &filter_dims_conv11,
                                        resnet_mlperf_index_layer_11,
                                        &bias_dims_conv11,
                                        bias_data_conv11,
                                        &output_dims_conv11,
                                        lut_data,
                                        actbuf2);

    free(ctx.buf); 
    
    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv12, &filter_dims_conv12);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    //result_arm = lut_conv_zdim_v1(&ctx,
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv12,
                                        actbuf2,
                                        &filter_dims_conv12,
                                        resnet_mlperf_index_layer_12,
                                        &bias_dims_conv12,
                                        bias_data_conv12,
                                        &output_dims_conv12,
                                        lut_data,
                                        actbuf1);

    free(ctx.buf); 
    
    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv13, &filter_dims_conv13);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,
    //result_arm = lut_conv_zdim_v1(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv13,
                                        actbuf1,
                                        &filter_dims_conv13,
                                        resnet_mlperf_index_layer_13,
                                        &bias_dims_conv13,
                                        bias_data_conv13,
                                        &output_dims_conv13,
                                        lut_data,
                                        actbuf2);

    free(ctx.buf); 

}

int main(){
	  HAL_Init();
    SystemClock_Config();
    while(1){
        conv_fw_resnet_mlperf();
    }
return 0;
}