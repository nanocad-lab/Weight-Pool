#include "../Utils/validate.h"
#include "D:\research\fixedweightNN\CMSIS_NN\Include\arm_nnfunctions.h"
#include "D:\research\fixedweightNN\CMSIS_5-develop\CMSIS_5-develop\TestData_fullnetwork\benchmarklayers\test_data.h"
#include "C:\Users\shuru\AppData\Local\Arm\Packs\Keil\STM32F2xx_DFP\2.9.0\Drivers\STM32F2xx_HAL_Driver\Inc\stm32f2xx_hal.h" //include file for clock generation code
#include "D:\research\fixedweightNN\CMSIS_5-develop\CMSIS_5-develop\lut_zdim64_data.h"
#include "D:\research\fixedweightNN\CMSIS_5-develop\CMSIS_5-develop\index_data\benchmarklayers_data\benchmarklayers_index_data.h"
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

void conv_zdim_benchmark(void)
{
    //Todo: Implement first layer using CMSIS (for filter pool benchmark), and load input from flash to ram first before start of first layer so input is always loaded from flash
    const arm_status expected = ARM_MATH_SUCCESS;
    /*for large networks like resnet, first layer's ouptut size will be 64KB, and is a lot larger than all subsequent layers, which is not necessery,
    and will cause ram issues if using double buffering for this size, so create a dymamic array to hold the output of first layer, and destroy it after second layer
    First layer's input can still be put into RAM since first layer only contains 3 channels, not 64, and requires way less memory than the output. 
    */
    q7_t* first_layer_outbuf = malloc(CONVLAYER1_OUT_CH*CONVLAYER1_OUTPUT_W*CONVLAYER1_OUTPUT_H*sizeof(q7_t));
    q7_t* first_layer_inbuf = malloc(CONVLAYER1_IN_CH*CONVLAYER1_INPUT_W*CONVLAYER1_INPUT_H*sizeof(q7_t));

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims_conv1, input_dims_conv2, input_dims_conv3, input_dims_conv4;
    cmsis_nn_dims filter_dims_conv1, filter_dims_conv2, filter_dims_conv3, filter_dims_conv4;
    cmsis_nn_dims bias_dims_conv1, bias_dims_conv2, bias_dims_conv3, bias_dims_conv4;
    cmsis_nn_dims output_dims_conv1, output_dims_conv2, output_dims_conv3, output_dims_conv4;

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

    arm_status result_arm;

    //code for layers
    //First copy the inputs from flash to ram
    memcpy(first_layer_inbuf, input_data_conv1, CONVLAYER1_IN_CH*CONVLAYER1_INPUT_W*CONVLAYER1_INPUT_H);

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv1, &filter_dims_conv1);
    //ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_nocaching(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv1,
                                        first_layer_inbuf,
                                        &filter_dims_conv1,
                                        benchmarklayers_index_layer_1,
                                        &bias_dims_conv1,
                                        bias_data_conv1,
                                        &output_dims_conv1,
                                        lut_data,
                                        first_layer_outbuf);

    //free(ctx.buf);
    free(first_layer_inbuf); //destory the input data buffer for first layer and release the memory
    q7_t* actbuf1 = malloc(16*16*64); //create the first general activation buffer, 16*16*64 should be enough for rest layers

    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv2, &filter_dims_conv2);
    //ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_nocaching(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv2,
                                        first_layer_outbuf,
                                        &filter_dims_conv2,
                                        benchmarklayers_index_layer_2,
                                        &bias_dims_conv2,
                                        bias_data_conv2,
                                        &output_dims_conv2,
                                        lut_data,
                                        actbuf1);

    //free(ctx.buf);
    free(first_layer_outbuf);//destroy the first layer output buffer and realse the memory
    q7_t* actbuf2 = malloc(16*16*64); //create the second general activation buffer, 16*16*64 should be enough for rest layers


    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv3, &filter_dims_conv3);
    //ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_nocaching(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv3,
                                        actbuf1,
                                        &filter_dims_conv3,
                                        benchmarklayers_index_layer_3,
                                        &bias_dims_conv3,
                                        bias_data_conv3,
                                        &output_dims_conv3,
                                        lut_data,
                                        actbuf2);

    //free(ctx.buf);    
    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv4, &filter_dims_conv4);
    //ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_nocaching(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv4,
                                        actbuf1,
                                        &filter_dims_conv4,
                                        benchmarklayers_index_layer_4,
                                        &bias_dims_conv4,
                                        bias_data_conv4,
                                        &output_dims_conv4,
                                        lut_data,
                                        actbuf2);

}

int main(){
		HAL_Init();
    SystemClock_Config();
    while(1){
        conv_zdim_benchmark();
    }
return 0;
}