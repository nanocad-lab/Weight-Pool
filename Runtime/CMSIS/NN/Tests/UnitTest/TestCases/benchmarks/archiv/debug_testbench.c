#include "C:\Users\shuru\AppData\Local\Arm\Packs\Keil\STM32F2xx_DFP\2.9.0\Drivers\STM32F2xx_HAL_Driver\Inc\stm32f2xx_hal.h" //include file for clock generation code
#include "D:\research\fixedweightNN\CMSIS_5-develop\CMSIS_5-develop\CMSIS\NN\Tests\UnitTest\TestCases\test_arm_convolve_s8\lut_data.h"
#include "lut_utiles.h"
#include <math.h>
#include <stdlib.h>

//#include <arm_nnfunctions.h>
#include "D:\research\fixedweightNN\CMSIS_NN\Include\arm_nnfunctions.h"
//#include <unity.h>

#include "../Utils/validate.h"
//#include "../TestData/conv_large/test_data.h"
//#include "../TestData/conv_small/test_data.h"
//#include "../TestData/conv_1616_3232/test_data.h"
#include "../TestData/conv_benchmark/test_data.h"
//#include "../TestData/fully_connected_debug/test_data.h"

//#include "D:\research\fixedweightNN\CMSIS_5-develop\CMSIS_5-develop\index_data\full_network_coeff_data.h"
//#include "D:\research\fixedweightNN\CMSIS_5-develop\CMSIS_5-develop\index_data\full_network_index_data.h"
#include "D:\research\fixedweightNN\CMSIS_5-develop\CMSIS_5-develop\index_data\benchmark_idx_data.h"

#define BLOCK_SIZE 8
//#define FW_GRAN 9 //granularity of fixed weight, for fixing 3*3 kernel it should be 9
//#define LUT_ENTRY 512 //should be equal to 2^(FW_GRAN)

#define INDEX_LEN_CONV (CONV_BENCHMARK_OUT_CH * CONV_BENCHMARK_IN_CH)

//#define INDEX_LEN_FC (FULLY_CONNECTED_DEBUG_IN_CH * FULLY_CONNECTED_DEBUG_OUT_CH / BLOCK_SIZE)

uint8_t *lut_index_gen_conv(uint8_t *index_arr)
{
    // number of datas need to be same as number of filters
    for (int i = 0; i < CONV_BENCHMARK_INPUT_W * CONV_BENCHMARK_INPUT_H * CONV_BENCHMARK_IN_CH; i++)
    {
        index_arr[i] = (uint8_t)rand();
    }
}

/*
uint8_t * lut_index_gen_fc(uint8_t *index_arr){
  //number of datas need to same as number of filters
  for(int i=0; i<INDEX_LEN_FC; i++){
    index_arr[i] = (uint8_t)rand();
  }
  //return index_arr;
}
*/

/**
 * @brief System Clock Configuration
 * @retval None
 */
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
        while (1)
        {
        }
    }
    /** Initializes the CPU, AHB and APB buses clocks
     */
    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK)
    {
        while (1)
        {
        }
    }
}

void conv_fw_convolve_debug(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t* output = malloc(CONV_BENCHMARK_DST_SIZE*sizeof(q7_t));

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const q31_t *bias_data = conv_benchmark_biases;
    const q7_t *kernel_data = conv_benchmark_weights;
    const q7_t *input_data = conv_benchmark_input;
    const q7_t *output_ref = conv_benchmark_output_ref;
    const int32_t output_ref_size = CONV_BENCHMARK_DST_SIZE;

    input_dims.n = CONV_BENCHMARK_INPUT_BATCHES;
    input_dims.w = CONV_BENCHMARK_INPUT_W;
    input_dims.h = CONV_BENCHMARK_INPUT_H;
    input_dims.c = CONV_BENCHMARK_IN_CH;
    filter_dims.w = CONV_BENCHMARK_FILTER_X;
    filter_dims.h = CONV_BENCHMARK_FILTER_Y;
    output_dims.w = CONV_BENCHMARK_OUTPUT_W;
    output_dims.h = CONV_BENCHMARK_OUTPUT_H;
    output_dims.c = CONV_BENCHMARK_OUT_CH;

    conv_params.padding.w = CONV_BENCHMARK_PAD_X;
    conv_params.padding.h = CONV_BENCHMARK_PAD_Y;
    conv_params.stride.w = CONV_BENCHMARK_STRIDE_X;
    conv_params.stride.h = CONV_BENCHMARK_STRIDE_Y;

    conv_params.input_offset = CONV_BENCHMARK_INPUT_OFFSET;
    conv_params.output_offset = CONV_BENCHMARK_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_BENCHMARK_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_BENCHMARK_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_benchmark_output_mult;
    quant_params.shift = (int32_t *)conv_benchmark_output_shift;

    static uint8_t lut_index[INDEX_LEN_CONV];
    static uint8_t coeff_data[INDEX_LEN_CONV];

    // lut_index_gen_conv(lut_index);
    // lut_index_gen_conv(coeff_data);

    //static q7_t ram_input[CONV_BENCHMARK_INPUT_W * CONV_BENCHMARK_INPUT_H * CONV_BENCHMARK_IN_CH];
    q7_t* ram_input = malloc(CONV_BENCHMARK_INPUT_W * CONV_BENCHMARK_INPUT_H * CONV_BENCHMARK_IN_CH*sizeof(q7_t));
    lut_index_gen_conv(ram_input);

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);

    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    
    arm_status result_arm = arm_convolve_s8(&ctx,
                                            &conv_params,
                                            &quant_params,
                                            &input_dims,
                                            // input_data,
                                            ram_input,
                                            &filter_dims,
                                            kernel_data,
                                            &bias_dims,
                                            bias_data,
                                            &output_dims,
                                            output);

    free(ctx.buf);

    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    arm_status result_lut_1 = lut_conv_zdim_v1(&ctx,
                                               &conv_params,
                                               &quant_params,
                                               &input_dims,
                                               // input_data,
                                               ram_input,
                                               &filter_dims,
                                               // lut_index,
                                               benchmart_idx_layer_1,
                                               &bias_dims,
                                               bias_data,
                                               &output_dims,
                                               lut_data,
                                               output);

    free(ctx.buf);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_status result_lut_2 = lut_conv_zdim_v2_double_lookup(&ctx,
                                                             &conv_params,
                                                             &quant_params,
                                                             &input_dims,
                                                             // input_data,
                                                             ram_input,
                                                             &filter_dims,
                                                             // lut_index,
                                                             benchmart_idx_layer_1,
                                                             &bias_dims,
                                                             bias_data,
                                                             &output_dims,
                                                             lut_data,
                                                             output);

    free(ctx.buf);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

}

/*
void fc_fw_lut_debug(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    static q7_t output[FULLY_CONNECTED_DEBUG_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const q31_t *bias_data = fully_connected_debug_biases;
    const q7_t *kernel_data = fully_connected_debug_weights;
    const q7_t *input_data = fully_connected_debug_input;
    const q7_t *output_ref = fully_connected_debug_output_ref;
    const int32_t output_ref_size = FULLY_CONNECTED_DEBUG_DST_SIZE;

    input_dims.n = FULLY_CONNECTED_DEBUG_INPUT_BATCHES;
    input_dims.w = FULLY_CONNECTED_DEBUG_INPUT_W;
    input_dims.h = FULLY_CONNECTED_DEBUG_INPUT_H;
    input_dims.c = FULLY_CONNECTED_DEBUG_IN_CH;
    filter_dims.n = FULLY_CONNECTED_DEBUG_ACCUMULATION_DEPTH;
    filter_dims.c = FULLY_CONNECTED_DEBUG_OUT_CH;
    output_dims.n = FULLY_CONNECTED_DEBUG_INPUT_BATCHES;
    output_dims.c = FULLY_CONNECTED_DEBUG_OUT_CH;

    fc_params.input_offset = FULLY_CONNECTED_DEBUG_INPUT_OFFSET;
    fc_params.filter_offset = 0;
    fc_params.output_offset = FULLY_CONNECTED_DEBUG_OUTPUT_OFFSET;
    fc_params.activation.min = FULLY_CONNECTED_DEBUG_OUT_ACTIVATION_MIN;
    fc_params.activation.max = FULLY_CONNECTED_DEBUG_OUT_ACTIVATION_MAX;

    quant_params.multiplier = FULLY_CONNECTED_DEBUG_OUTPUT_MULTIPLIER;
    quant_params.shift = FULLY_CONNECTED_DEBUG_OUTPUT_SHIFT;

    static uint8_t lut_index[INDEX_LEN_FC];
    static uint8_t coeff_data[INDEX_LEN_FC];

    int index_len = output_dims.c * input_dims.c / BLOCK_SIZE;
    lut_index_gen_fc(lut_index);
    lut_index_gen_fc(coeff_data);

    int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    arm_status result_s8 = arm_fully_connected_s8(&ctx,
                                               &fc_params,
                                               &quant_params,
                                               &input_dims,
                                               input_data,
                                               &filter_dims,
                                               kernel_data,
                                               &bias_dims,
                                               bias_data,
                                               &output_dims,
                                               output);

    free(ctx.buf);

    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    arm_status result_lutv1 = lut_fully_connected_v1_withcoeff(&ctx,
                                            &fc_params,
                                            &quant_params,
                                            &input_dims,
                                            input_data,
                                            &filter_dims,
                                            lut_index,
                                            &bias_dims,
                                            bias_data,
                                            &output_dims,
                                            lut_data,
                                            coeff_data,
                                            output);
}
*/
int main()
{
    HAL_Init();
    SystemClock_Config();
    while (1)
    {

        // conv_2_arm_convolve_s8();
        // static q7_t output[6272] = {0};
        conv_fw_convolve_debug();
        // fc_fw_lut_debug();
    }
    return 0;
}