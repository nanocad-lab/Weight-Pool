#include "../Utils/validate.h"
#include "..\..\..\..\Include\arm_nnfunctions.h"
#include "..\..\..\..\..\..\lut_zdim64_data.h"
#include "..\..\..\..\..\..\TestData_fullnetwork\mobilenet_v2\test_data.h"
#include "..\..\..\..\..\..\index_data\mobilenet_v2_data\mobilenet_v2_index_data.h"
#include "..\..\..\..\..\..\STM32F2xx_HAL_Driver\Inc\stm32f2xx_hal.h" //include file for clock generation code
#include "lut_utiles.h"
#include <math.h>
#include <stdlib.h>


void SysTick_Handler(void)
{
  /* USER CODE BEGIN SysTick_IRQn 0 */

  /* USER CODE END SysTick_IRQn 0 */
  HAL_IncTick();
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

void conv_fw_mobilenet_v2(void)
{
    //Todo: Implement first layer using CMSIS (for filter pool benchmark), and load input from flash to ram first before start of first layer so input is always loaded from flash
    const arm_status expected = ARM_MATH_SUCCESS;
    /*for large networks like resnet, first layer's ouptut size will be 64KB, and is a lot larger than all subsequent layers, which is not necessery,
    and will cause ram issues if using double buffering for this size, so create a dymamic array to hold the output of first layer, and destroy it after second layer
    First layer's input can still be put into RAM since first layer only contains 3 channels, not 64, and requires way less memory than the output. 
    */

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params, stride2_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims_conv1,input_dims_conv2,input_dims_conv3,input_dims_conv4,input_dims_conv5,input_dims_conv6,input_dims_conv7,input_dims_conv8,input_dims_conv9,input_dims_conv10,input_dims_conv11,input_dims_conv12,input_dims_conv13,input_dims_conv14,input_dims_conv15,input_dims_conv16,input_dims_conv17,input_dims_conv18,input_dims_conv19,input_dims_conv20,input_dims_conv21,input_dims_conv22,input_dims_conv23,input_dims_conv24,input_dims_conv25,input_dims_conv26,input_dims_conv27,input_dims_conv28,input_dims_conv29,input_dims_conv30,input_dims_conv31,input_dims_conv32,input_dims_conv33,input_dims_conv34,input_dims_conv35,input_dims_conv36,input_dims_conv37,input_dims_conv38,input_dims_conv39,input_dims_conv40,input_dims_conv41,input_dims_conv42,input_dims_conv43,input_dims_conv44,input_dims_conv45,input_dims_conv46,input_dims_conv47,input_dims_conv48,input_dims_conv49,input_dims_conv50,input_dims_conv51,input_dims_conv52,input_dims_conv53,input_dims_conv54,input_dims_conv55,input_dims_conv56,input_dims_conv57,input_dims_conv58;
    cmsis_nn_dims filter_dims_conv1,filter_dims_conv2,filter_dims_conv3,filter_dims_conv4,filter_dims_conv5,filter_dims_conv6,filter_dims_conv7,filter_dims_conv8,filter_dims_conv9,filter_dims_conv10,filter_dims_conv11,filter_dims_conv12,filter_dims_conv13,filter_dims_conv14,filter_dims_conv15,filter_dims_conv16,filter_dims_conv17,filter_dims_conv18,filter_dims_conv19,filter_dims_conv20,filter_dims_conv21,filter_dims_conv22,filter_dims_conv23,filter_dims_conv24,filter_dims_conv25,filter_dims_conv26,filter_dims_conv27,filter_dims_conv28,filter_dims_conv29,filter_dims_conv30,filter_dims_conv31,filter_dims_conv32,filter_dims_conv33,filter_dims_conv34,filter_dims_conv35,filter_dims_conv36,filter_dims_conv37,filter_dims_conv38,filter_dims_conv39,filter_dims_conv40,filter_dims_conv41,filter_dims_conv42,filter_dims_conv43,filter_dims_conv44,filter_dims_conv45,filter_dims_conv46,filter_dims_conv47,filter_dims_conv48,filter_dims_conv49,filter_dims_conv50,filter_dims_conv51,filter_dims_conv52,filter_dims_conv53,filter_dims_conv54,filter_dims_conv55,filter_dims_conv56,filter_dims_conv57,filter_dims_conv58;
    cmsis_nn_dims bias_dims_conv1,bias_dims_conv2,bias_dims_conv3,bias_dims_conv4,bias_dims_conv5,bias_dims_conv6,bias_dims_conv7,bias_dims_conv8,bias_dims_conv9,bias_dims_conv10,bias_dims_conv11,bias_dims_conv12,bias_dims_conv13,bias_dims_conv14,bias_dims_conv15,bias_dims_conv16,bias_dims_conv17,bias_dims_conv18,bias_dims_conv19,bias_dims_conv20,bias_dims_conv21,bias_dims_conv22,bias_dims_conv23,bias_dims_conv24,bias_dims_conv25,bias_dims_conv26,bias_dims_conv27,bias_dims_conv28,bias_dims_conv29,bias_dims_conv30,bias_dims_conv31,bias_dims_conv32,bias_dims_conv33,bias_dims_conv34,bias_dims_conv35,bias_dims_conv36,bias_dims_conv37,bias_dims_conv38,bias_dims_conv39,bias_dims_conv40,bias_dims_conv41,bias_dims_conv42,bias_dims_conv43,bias_dims_conv44,bias_dims_conv45,bias_dims_conv46,bias_dims_conv47,bias_dims_conv48,bias_dims_conv49,bias_dims_conv50,bias_dims_conv51,bias_dims_conv52,bias_dims_conv53,bias_dims_conv54,bias_dims_conv55,bias_dims_conv56,bias_dims_conv57,bias_dims_conv58;
    cmsis_nn_dims output_dims_conv1,output_dims_conv2,output_dims_conv3,output_dims_conv4,output_dims_conv5,output_dims_conv6,output_dims_conv7,output_dims_conv8,output_dims_conv9,output_dims_conv10,output_dims_conv11,output_dims_conv12,output_dims_conv13,output_dims_conv14,output_dims_conv15,output_dims_conv16,output_dims_conv17,output_dims_conv18,output_dims_conv19,output_dims_conv20,output_dims_conv21,output_dims_conv22,output_dims_conv23,output_dims_conv24,output_dims_conv25,output_dims_conv26,output_dims_conv27,output_dims_conv28,output_dims_conv29,output_dims_conv30,output_dims_conv31,output_dims_conv32,output_dims_conv33,output_dims_conv34,output_dims_conv35,output_dims_conv36,output_dims_conv37,output_dims_conv38,output_dims_conv39,output_dims_conv40,output_dims_conv41,output_dims_conv42,output_dims_conv43,output_dims_conv44,output_dims_conv45,output_dims_conv46,output_dims_conv47,output_dims_conv48,output_dims_conv49,output_dims_conv50,output_dims_conv51,output_dims_conv52,output_dims_conv53,output_dims_conv54,output_dims_conv55,output_dims_conv56,output_dims_conv57,output_dims_conv58;

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

    const q31_t *bias_data_conv14 = convlayer14_biases;
    const q7_t *kernel_data_conv14 = convlayer14_weights;

    input_dims_conv14.n = CONVLAYER14_INPUT_BATCHES;
    input_dims_conv14.w = CONVLAYER14_INPUT_W;
    input_dims_conv14.h = CONVLAYER14_INPUT_H;
    input_dims_conv14.c = CONVLAYER14_IN_CH;
    filter_dims_conv14.w = CONVLAYER14_FILTER_X;
    filter_dims_conv14.h = CONVLAYER14_FILTER_Y;
    output_dims_conv14.w = CONVLAYER14_OUTPUT_W;
    output_dims_conv14.h = CONVLAYER14_OUTPUT_H;
    output_dims_conv14.c = CONVLAYER14_OUT_CH;
    
    const q31_t *bias_data_conv15 = convlayer1_biases;
    const q7_t *kernel_data_conv15 = convlayer1_weights;

    input_dims_conv15.n = CONVLAYER15_INPUT_BATCHES;
    input_dims_conv15.w = CONVLAYER15_INPUT_W;
    input_dims_conv15.h = CONVLAYER15_INPUT_H;
    input_dims_conv15.c = CONVLAYER15_IN_CH;
    filter_dims_conv15.w = CONVLAYER15_FILTER_X;
    filter_dims_conv15.h = CONVLAYER15_FILTER_Y;
    output_dims_conv15.w = CONVLAYER15_OUTPUT_W;
    output_dims_conv15.h = CONVLAYER15_OUTPUT_H;
    output_dims_conv15.c = CONVLAYER15_OUT_CH;


    const q31_t *bias_data_conv16 = convlayer1_biases;
    const q7_t *kernel_data_conv16 = convlayer1_weights;

    input_dims_conv16.n = CONVLAYER16_INPUT_BATCHES;
    input_dims_conv16.w = CONVLAYER16_INPUT_W;
    input_dims_conv16.h = CONVLAYER16_INPUT_H;
    input_dims_conv16.c = CONVLAYER16_IN_CH;
    filter_dims_conv16.w = CONVLAYER16_FILTER_X;
    filter_dims_conv16.h = CONVLAYER16_FILTER_Y;
    output_dims_conv16.w = CONVLAYER16_OUTPUT_W;
    output_dims_conv16.h = CONVLAYER16_OUTPUT_H;
    output_dims_conv16.c = CONVLAYER16_OUT_CH;


    const q31_t *bias_data_conv17 = convlayer1_biases;
    const q7_t *kernel_data_conv17 = convlayer1_weights;

    input_dims_conv17.n = CONVLAYER17_INPUT_BATCHES;
    input_dims_conv17.w = CONVLAYER17_INPUT_W;
    input_dims_conv17.h = CONVLAYER17_INPUT_H;
    input_dims_conv17.c = CONVLAYER17_IN_CH;
    filter_dims_conv17.w = CONVLAYER17_FILTER_X;
    filter_dims_conv17.h = CONVLAYER17_FILTER_Y;
    output_dims_conv17.w = CONVLAYER17_OUTPUT_W;
    output_dims_conv17.h = CONVLAYER17_OUTPUT_H;
    output_dims_conv17.c = CONVLAYER17_OUT_CH;


    const q31_t *bias_data_conv18 = convlayer18_biases;
    const q7_t *kernel_data_conv18 = convlayer18_weights;

    input_dims_conv18.n = CONVLAYER18_INPUT_BATCHES;
    input_dims_conv18.w = CONVLAYER18_INPUT_W;
    input_dims_conv18.h = CONVLAYER18_INPUT_H;
    input_dims_conv18.c = CONVLAYER18_IN_CH;
    filter_dims_conv18.w = CONVLAYER18_FILTER_X;
    filter_dims_conv18.h = CONVLAYER18_FILTER_Y;
    output_dims_conv18.w = CONVLAYER18_OUTPUT_W;
    output_dims_conv18.h = CONVLAYER18_OUTPUT_H;
    output_dims_conv18.c = CONVLAYER18_OUT_CH;


    const q31_t *bias_data_conv19 = convlayer1_biases;
    const q7_t *kernel_data_conv19 = convlayer1_weights;

    input_dims_conv19.n = CONVLAYER19_INPUT_BATCHES;
    input_dims_conv19.w = CONVLAYER19_INPUT_W;
    input_dims_conv19.h = CONVLAYER19_INPUT_H;
    input_dims_conv19.c = CONVLAYER19_IN_CH;
    filter_dims_conv19.w = CONVLAYER19_FILTER_X;
    filter_dims_conv19.h = CONVLAYER19_FILTER_Y;
    output_dims_conv19.w = CONVLAYER19_OUTPUT_W;
    output_dims_conv19.h = CONVLAYER19_OUTPUT_H;
    output_dims_conv19.c = CONVLAYER19_OUT_CH;


    const q31_t *bias_data_conv20 = convlayer1_biases;
    const q7_t *kernel_data_conv20 = convlayer1_weights;

    input_dims_conv20.n = CONVLAYER20_INPUT_BATCHES;
    input_dims_conv20.w = CONVLAYER20_INPUT_W;
    input_dims_conv20.h = CONVLAYER20_INPUT_H;
    input_dims_conv20.c = CONVLAYER20_IN_CH;
    filter_dims_conv20.w = CONVLAYER20_FILTER_X;
    filter_dims_conv20.h = CONVLAYER20_FILTER_Y;
    output_dims_conv20.w = CONVLAYER20_OUTPUT_W;
    output_dims_conv20.h = CONVLAYER20_OUTPUT_H;
    output_dims_conv20.c = CONVLAYER20_OUT_CH;


    const q31_t *bias_data_conv21 = convlayer21_biases;
    const q7_t *kernel_data_conv21 = convlayer21_weights;

    input_dims_conv21.n = CONVLAYER21_INPUT_BATCHES;
    input_dims_conv21.w = CONVLAYER21_INPUT_W;
    input_dims_conv21.h = CONVLAYER21_INPUT_H;
    input_dims_conv21.c = CONVLAYER21_IN_CH;
    filter_dims_conv21.w = CONVLAYER21_FILTER_X;
    filter_dims_conv21.h = CONVLAYER21_FILTER_Y;
    output_dims_conv21.w = CONVLAYER21_OUTPUT_W;
    output_dims_conv21.h = CONVLAYER21_OUTPUT_H;
    output_dims_conv21.c = CONVLAYER21_OUT_CH;


    const q31_t *bias_data_conv22 = convlayer1_biases;
    const q7_t *kernel_data_conv22 = convlayer1_weights;

    input_dims_conv22.n = CONVLAYER22_INPUT_BATCHES;
    input_dims_conv22.w = CONVLAYER22_INPUT_W;
    input_dims_conv22.h = CONVLAYER22_INPUT_H;
    input_dims_conv22.c = CONVLAYER22_IN_CH;
    filter_dims_conv22.w = CONVLAYER22_FILTER_X;
    filter_dims_conv22.h = CONVLAYER22_FILTER_Y;
    output_dims_conv22.w = CONVLAYER22_OUTPUT_W;
    output_dims_conv22.h = CONVLAYER22_OUTPUT_H;
    output_dims_conv22.c = CONVLAYER22_OUT_CH;


    const q31_t *bias_data_conv23 = convlayer1_biases;
    const q7_t *kernel_data_conv23 = convlayer1_weights;

    input_dims_conv23.n = CONVLAYER23_INPUT_BATCHES;
    input_dims_conv23.w = CONVLAYER23_INPUT_W;
    input_dims_conv23.h = CONVLAYER23_INPUT_H;
    input_dims_conv23.c = CONVLAYER23_IN_CH;
    filter_dims_conv23.w = CONVLAYER23_FILTER_X;
    filter_dims_conv23.h = CONVLAYER23_FILTER_Y;
    output_dims_conv23.w = CONVLAYER23_OUTPUT_W;
    output_dims_conv23.h = CONVLAYER23_OUTPUT_H;
    output_dims_conv23.c = CONVLAYER23_OUT_CH;


    const q31_t *bias_data_conv24 = convlayer24_biases;
    const q7_t *kernel_data_conv24 = convlayer24_weights;

    input_dims_conv24.n = CONVLAYER24_INPUT_BATCHES;
    input_dims_conv24.w = CONVLAYER24_INPUT_W;
    input_dims_conv24.h = CONVLAYER24_INPUT_H;
    input_dims_conv24.c = CONVLAYER24_IN_CH;
    filter_dims_conv24.w = CONVLAYER24_FILTER_X;
    filter_dims_conv24.h = CONVLAYER24_FILTER_Y;
    output_dims_conv24.w = CONVLAYER24_OUTPUT_W;
    output_dims_conv24.h = CONVLAYER24_OUTPUT_H;
    output_dims_conv24.c = CONVLAYER24_OUT_CH;


    const q31_t *bias_data_conv25 = convlayer1_biases;
    const q7_t *kernel_data_conv25 = convlayer1_weights;

    input_dims_conv25.n = CONVLAYER25_INPUT_BATCHES;
    input_dims_conv25.w = CONVLAYER25_INPUT_W;
    input_dims_conv25.h = CONVLAYER25_INPUT_H;
    input_dims_conv25.c = CONVLAYER25_IN_CH;
    filter_dims_conv25.w = CONVLAYER25_FILTER_X;
    filter_dims_conv25.h = CONVLAYER25_FILTER_Y;
    output_dims_conv25.w = CONVLAYER25_OUTPUT_W;
    output_dims_conv25.h = CONVLAYER25_OUTPUT_H;
    output_dims_conv25.c = CONVLAYER25_OUT_CH;


    const q31_t *bias_data_conv26 = convlayer1_biases;
    const q7_t *kernel_data_conv26 = convlayer1_weights;

    input_dims_conv26.n = CONVLAYER26_INPUT_BATCHES;
    input_dims_conv26.w = CONVLAYER26_INPUT_W;
    input_dims_conv26.h = CONVLAYER26_INPUT_H;
    input_dims_conv26.c = CONVLAYER26_IN_CH;
    filter_dims_conv26.w = CONVLAYER26_FILTER_X;
    filter_dims_conv26.h = CONVLAYER26_FILTER_Y;
    output_dims_conv26.w = CONVLAYER26_OUTPUT_W;
    output_dims_conv26.h = CONVLAYER26_OUTPUT_H;
    output_dims_conv26.c = CONVLAYER26_OUT_CH;


    const q31_t *bias_data_conv27 = convlayer27_biases;
    const q7_t *kernel_data_conv27 = convlayer27_weights;

    input_dims_conv27.n = CONVLAYER27_INPUT_BATCHES;
    input_dims_conv27.w = CONVLAYER27_INPUT_W;
    input_dims_conv27.h = CONVLAYER27_INPUT_H;
    input_dims_conv27.c = CONVLAYER27_IN_CH;
    filter_dims_conv27.w = CONVLAYER27_FILTER_X;
    filter_dims_conv27.h = CONVLAYER27_FILTER_Y;
    output_dims_conv27.w = CONVLAYER27_OUTPUT_W;
    output_dims_conv27.h = CONVLAYER27_OUTPUT_H;
    output_dims_conv27.c = CONVLAYER27_OUT_CH;


    const q31_t *bias_data_conv28 = convlayer1_biases;
    const q7_t *kernel_data_conv28 = convlayer1_weights;

    input_dims_conv28.n = CONVLAYER28_INPUT_BATCHES;
    input_dims_conv28.w = CONVLAYER28_INPUT_W;
    input_dims_conv28.h = CONVLAYER28_INPUT_H;
    input_dims_conv28.c = CONVLAYER28_IN_CH;
    filter_dims_conv28.w = CONVLAYER28_FILTER_X;
    filter_dims_conv28.h = CONVLAYER28_FILTER_Y;
    output_dims_conv28.w = CONVLAYER28_OUTPUT_W;
    output_dims_conv28.h = CONVLAYER28_OUTPUT_H;
    output_dims_conv28.c = CONVLAYER28_OUT_CH;


    const q31_t *bias_data_conv29 = convlayer1_biases;
    const q7_t *kernel_data_conv29 = convlayer1_weights;

    input_dims_conv29.n = CONVLAYER29_INPUT_BATCHES;
    input_dims_conv29.w = CONVLAYER29_INPUT_W;
    input_dims_conv29.h = CONVLAYER29_INPUT_H;
    input_dims_conv29.c = CONVLAYER29_IN_CH;
    filter_dims_conv29.w = CONVLAYER29_FILTER_X;
    filter_dims_conv29.h = CONVLAYER29_FILTER_Y;
    output_dims_conv29.w = CONVLAYER29_OUTPUT_W;
    output_dims_conv29.h = CONVLAYER29_OUTPUT_H;
    output_dims_conv29.c = CONVLAYER29_OUT_CH;


    const q31_t *bias_data_conv30 = convlayer30_biases;
    const q7_t *kernel_data_conv30 = convlayer30_weights;

    input_dims_conv30.n = CONVLAYER30_INPUT_BATCHES;
    input_dims_conv30.w = CONVLAYER30_INPUT_W;
    input_dims_conv30.h = CONVLAYER30_INPUT_H;
    input_dims_conv30.c = CONVLAYER30_IN_CH;
    filter_dims_conv30.w = CONVLAYER30_FILTER_X;
    filter_dims_conv30.h = CONVLAYER30_FILTER_Y;
    output_dims_conv30.w = CONVLAYER30_OUTPUT_W;
    output_dims_conv30.h = CONVLAYER30_OUTPUT_H;
    output_dims_conv30.c = CONVLAYER30_OUT_CH;


    const q31_t *bias_data_conv31 = convlayer1_biases;
    const q7_t *kernel_data_conv31 = convlayer1_weights;

    input_dims_conv31.n = CONVLAYER31_INPUT_BATCHES;
    input_dims_conv31.w = CONVLAYER31_INPUT_W;
    input_dims_conv31.h = CONVLAYER31_INPUT_H;
    input_dims_conv31.c = CONVLAYER31_IN_CH;
    filter_dims_conv31.w = CONVLAYER31_FILTER_X;
    filter_dims_conv31.h = CONVLAYER31_FILTER_Y;
    output_dims_conv31.w = CONVLAYER31_OUTPUT_W;
    output_dims_conv31.h = CONVLAYER31_OUTPUT_H;
    output_dims_conv31.c = CONVLAYER31_OUT_CH;


    const q31_t *bias_data_conv32 = convlayer1_biases;
    const q7_t *kernel_data_conv32 = convlayer1_weights;

    input_dims_conv32.n = CONVLAYER32_INPUT_BATCHES;
    input_dims_conv32.w = CONVLAYER32_INPUT_W;
    input_dims_conv32.h = CONVLAYER32_INPUT_H;
    input_dims_conv32.c = CONVLAYER32_IN_CH;
    filter_dims_conv32.w = CONVLAYER32_FILTER_X;
    filter_dims_conv32.h = CONVLAYER32_FILTER_Y;
    output_dims_conv32.w = CONVLAYER32_OUTPUT_W;
    output_dims_conv32.h = CONVLAYER32_OUTPUT_H;
    output_dims_conv32.c = CONVLAYER32_OUT_CH;


    const q31_t *bias_data_conv33 = convlayer33_biases;
    const q7_t *kernel_data_conv33 = convlayer33_weights;

    input_dims_conv33.n = CONVLAYER33_INPUT_BATCHES;
    input_dims_conv33.w = CONVLAYER33_INPUT_W;
    input_dims_conv33.h = CONVLAYER33_INPUT_H;
    input_dims_conv33.c = CONVLAYER33_IN_CH;
    filter_dims_conv33.w = CONVLAYER33_FILTER_X;
    filter_dims_conv33.h = CONVLAYER33_FILTER_Y;
    output_dims_conv33.w = CONVLAYER33_OUTPUT_W;
    output_dims_conv33.h = CONVLAYER33_OUTPUT_H;
    output_dims_conv33.c = CONVLAYER33_OUT_CH;


    const q31_t *bias_data_conv34 = convlayer1_biases;
    const q7_t *kernel_data_conv34 = convlayer1_weights;

    input_dims_conv34.n = CONVLAYER34_INPUT_BATCHES;
    input_dims_conv34.w = CONVLAYER34_INPUT_W;
    input_dims_conv34.h = CONVLAYER34_INPUT_H;
    input_dims_conv34.c = CONVLAYER34_IN_CH;
    filter_dims_conv34.w = CONVLAYER34_FILTER_X;
    filter_dims_conv34.h = CONVLAYER34_FILTER_Y;
    output_dims_conv34.w = CONVLAYER34_OUTPUT_W;
    output_dims_conv34.h = CONVLAYER34_OUTPUT_H;
    output_dims_conv34.c = CONVLAYER34_OUT_CH;


    const q31_t *bias_data_conv35 = convlayer1_biases;
    const q7_t *kernel_data_conv35 = convlayer1_weights;

    input_dims_conv35.n = CONVLAYER35_INPUT_BATCHES;
    input_dims_conv35.w = CONVLAYER35_INPUT_W;
    input_dims_conv35.h = CONVLAYER35_INPUT_H;
    input_dims_conv35.c = CONVLAYER35_IN_CH;
    filter_dims_conv35.w = CONVLAYER35_FILTER_X;
    filter_dims_conv35.h = CONVLAYER35_FILTER_Y;
    output_dims_conv35.w = CONVLAYER35_OUTPUT_W;
    output_dims_conv35.h = CONVLAYER35_OUTPUT_H;
    output_dims_conv35.c = CONVLAYER35_OUT_CH;


    const q31_t *bias_data_conv36 = convlayer36_biases;
    const q7_t *kernel_data_conv36 = convlayer36_weights;

    input_dims_conv36.n = CONVLAYER36_INPUT_BATCHES;
    input_dims_conv36.w = CONVLAYER36_INPUT_W;
    input_dims_conv36.h = CONVLAYER36_INPUT_H;
    input_dims_conv36.c = CONVLAYER36_IN_CH;
    filter_dims_conv36.w = CONVLAYER36_FILTER_X;
    filter_dims_conv36.h = CONVLAYER36_FILTER_Y;
    output_dims_conv36.w = CONVLAYER36_OUTPUT_W;
    output_dims_conv36.h = CONVLAYER36_OUTPUT_H;
    output_dims_conv36.c = CONVLAYER36_OUT_CH;

    const q31_t *bias_data_conv37 = convlayer1_biases;
    const q7_t *kernel_data_conv37 = convlayer1_weights;

    input_dims_conv37.n = CONVLAYER37_INPUT_BATCHES;
    input_dims_conv37.w = CONVLAYER37_INPUT_W;
    input_dims_conv37.h = CONVLAYER37_INPUT_H;
    input_dims_conv37.c = CONVLAYER37_IN_CH;
    filter_dims_conv37.w = CONVLAYER37_FILTER_X;
    filter_dims_conv37.h = CONVLAYER37_FILTER_Y;
    output_dims_conv37.w = CONVLAYER37_OUTPUT_W;
    output_dims_conv37.h = CONVLAYER37_OUTPUT_H;
    output_dims_conv37.c = CONVLAYER37_OUT_CH;

    const q31_t *bias_data_conv38 = convlayer1_biases;
    const q7_t *kernel_data_conv38 = convlayer1_weights;

    input_dims_conv38.n = CONVLAYER38_INPUT_BATCHES;
    input_dims_conv38.w = CONVLAYER38_INPUT_W;
    input_dims_conv38.h = CONVLAYER38_INPUT_H;
    input_dims_conv38.c = CONVLAYER38_IN_CH;
    filter_dims_conv38.w = CONVLAYER38_FILTER_X;
    filter_dims_conv38.h = CONVLAYER38_FILTER_Y;
    output_dims_conv38.w = CONVLAYER38_OUTPUT_W;
    output_dims_conv38.h = CONVLAYER38_OUTPUT_H;
    output_dims_conv38.c = CONVLAYER38_OUT_CH;

    const q31_t *bias_data_conv39 = convlayer1_biases;
    const q7_t *kernel_data_conv39 = convlayer1_weights;

    input_dims_conv39.n = CONVLAYER39_INPUT_BATCHES;
    input_dims_conv39.w = CONVLAYER39_INPUT_W;
    input_dims_conv39.h = CONVLAYER39_INPUT_H;
    input_dims_conv39.c = CONVLAYER39_IN_CH;
    filter_dims_conv39.w = CONVLAYER39_FILTER_X;
    filter_dims_conv39.h = CONVLAYER39_FILTER_Y;
    output_dims_conv39.w = CONVLAYER39_OUTPUT_W;
    output_dims_conv39.h = CONVLAYER39_OUTPUT_H;
    output_dims_conv39.c = CONVLAYER39_OUT_CH;

    const q31_t *bias_data_conv40 = convlayer40_biases;
    const q7_t *kernel_data_conv40 = convlayer40_weights;

    input_dims_conv40.n = CONVLAYER40_INPUT_BATCHES;
    input_dims_conv40.w = CONVLAYER40_INPUT_W;
    input_dims_conv40.h = CONVLAYER40_INPUT_H;
    input_dims_conv40.c = CONVLAYER40_IN_CH;
    filter_dims_conv40.w = CONVLAYER40_FILTER_X;
    filter_dims_conv40.h = CONVLAYER40_FILTER_Y;
    output_dims_conv40.w = CONVLAYER40_OUTPUT_W;
    output_dims_conv40.h = CONVLAYER40_OUTPUT_H;
    output_dims_conv40.c = CONVLAYER40_OUT_CH;

    const q31_t *bias_data_conv41 = convlayer1_biases;
    const q7_t *kernel_data_conv41 = convlayer1_weights;

    input_dims_conv41.n = CONVLAYER41_INPUT_BATCHES;
    input_dims_conv41.w = CONVLAYER41_INPUT_W;
    input_dims_conv41.h = CONVLAYER41_INPUT_H;
    input_dims_conv41.c = CONVLAYER41_IN_CH;
    filter_dims_conv41.w = CONVLAYER41_FILTER_X;
    filter_dims_conv41.h = CONVLAYER41_FILTER_Y;
    output_dims_conv41.w = CONVLAYER41_OUTPUT_W;
    output_dims_conv41.h = CONVLAYER41_OUTPUT_H;
    output_dims_conv41.c = CONVLAYER41_OUT_CH;

    const q31_t *bias_data_conv42 = convlayer1_biases;
    const q7_t *kernel_data_conv42 = convlayer1_weights;

    input_dims_conv42.n = CONVLAYER42_INPUT_BATCHES;
    input_dims_conv42.w = CONVLAYER42_INPUT_W;
    input_dims_conv42.h = CONVLAYER42_INPUT_H;
    input_dims_conv42.c = CONVLAYER42_IN_CH;
    filter_dims_conv42.w = CONVLAYER42_FILTER_X;
    filter_dims_conv42.h = CONVLAYER42_FILTER_Y;
    output_dims_conv42.w = CONVLAYER42_OUTPUT_W;
    output_dims_conv42.h = CONVLAYER42_OUTPUT_H;
    output_dims_conv42.c = CONVLAYER42_OUT_CH;

    const q31_t *bias_data_conv43 = convlayer43_biases;
    const q7_t *kernel_data_conv43 = convlayer43_weights;

    input_dims_conv43.n = CONVLAYER43_INPUT_BATCHES;
    input_dims_conv43.w = CONVLAYER43_INPUT_W;
    input_dims_conv43.h = CONVLAYER43_INPUT_H;
    input_dims_conv43.c = CONVLAYER43_IN_CH;
    filter_dims_conv43.w = CONVLAYER43_FILTER_X;
    filter_dims_conv43.h = CONVLAYER43_FILTER_Y;
    output_dims_conv43.w = CONVLAYER43_OUTPUT_W;
    output_dims_conv43.h = CONVLAYER43_OUTPUT_H;
    output_dims_conv43.c = CONVLAYER43_OUT_CH;

    const q31_t *bias_data_conv44 = convlayer1_biases;
    const q7_t *kernel_data_conv44 = convlayer1_weights;

    input_dims_conv44.n = CONVLAYER44_INPUT_BATCHES;
    input_dims_conv44.w = CONVLAYER44_INPUT_W;
    input_dims_conv44.h = CONVLAYER44_INPUT_H;
    input_dims_conv44.c = CONVLAYER44_IN_CH;
    filter_dims_conv44.w = CONVLAYER44_FILTER_X;
    filter_dims_conv44.h = CONVLAYER44_FILTER_Y;
    output_dims_conv44.w = CONVLAYER44_OUTPUT_W;
    output_dims_conv44.h = CONVLAYER44_OUTPUT_H;
    output_dims_conv44.c = CONVLAYER44_OUT_CH;

    const q31_t *bias_data_conv45 = convlayer1_biases;
    const q7_t *kernel_data_conv45 = convlayer1_weights;

    input_dims_conv45.n = CONVLAYER45_INPUT_BATCHES;
    input_dims_conv45.w = CONVLAYER45_INPUT_W;
    input_dims_conv45.h = CONVLAYER45_INPUT_H;
    input_dims_conv45.c = CONVLAYER45_IN_CH;
    filter_dims_conv45.w = CONVLAYER45_FILTER_X;
    filter_dims_conv45.h = CONVLAYER45_FILTER_Y;
    output_dims_conv45.w = CONVLAYER45_OUTPUT_W;
    output_dims_conv45.h = CONVLAYER45_OUTPUT_H;
    output_dims_conv45.c = CONVLAYER45_OUT_CH;

    const q31_t *bias_data_conv46 = convlayer46_biases;
    const q7_t *kernel_data_conv46 = convlayer46_weights;

    input_dims_conv46.n = CONVLAYER46_INPUT_BATCHES;
    input_dims_conv46.w = CONVLAYER46_INPUT_W;
    input_dims_conv46.h = CONVLAYER46_INPUT_H;
    input_dims_conv46.c = CONVLAYER46_IN_CH;
    filter_dims_conv46.w = CONVLAYER46_FILTER_X;
    filter_dims_conv46.h = CONVLAYER46_FILTER_Y;
    output_dims_conv46.w = CONVLAYER46_OUTPUT_W;
    output_dims_conv46.h = CONVLAYER46_OUTPUT_H;
    output_dims_conv46.c = CONVLAYER46_OUT_CH;

    const q31_t *bias_data_conv47 = convlayer1_biases;
    const q7_t *kernel_data_conv47 = convlayer1_weights;

    input_dims_conv47.n = CONVLAYER47_INPUT_BATCHES;
    input_dims_conv47.w = CONVLAYER47_INPUT_W;
    input_dims_conv47.h = CONVLAYER47_INPUT_H;
    input_dims_conv47.c = CONVLAYER47_IN_CH;
    filter_dims_conv47.w = CONVLAYER47_FILTER_X;
    filter_dims_conv47.h = CONVLAYER47_FILTER_Y;
    output_dims_conv47.w = CONVLAYER47_OUTPUT_W;
    output_dims_conv47.h = CONVLAYER47_OUTPUT_H;
    output_dims_conv47.c = CONVLAYER47_OUT_CH;

    const q31_t *bias_data_conv48 = convlayer1_biases;
    const q7_t *kernel_data_conv48 = convlayer1_weights;

    input_dims_conv48.n = CONVLAYER48_INPUT_BATCHES;
    input_dims_conv48.w = CONVLAYER48_INPUT_W;
    input_dims_conv48.h = CONVLAYER48_INPUT_H;
    input_dims_conv48.c = CONVLAYER48_IN_CH;
    filter_dims_conv48.w = CONVLAYER48_FILTER_X;
    filter_dims_conv48.h = CONVLAYER48_FILTER_Y;
    output_dims_conv48.w = CONVLAYER48_OUTPUT_W;
    output_dims_conv48.h = CONVLAYER48_OUTPUT_H;
    output_dims_conv48.c = CONVLAYER48_OUT_CH;

    const q31_t *bias_data_conv49 = convlayer49_biases;
    const q7_t *kernel_data_conv49 = convlayer49_weights;

    input_dims_conv49.n = CONVLAYER49_INPUT_BATCHES;
    input_dims_conv49.w = CONVLAYER49_INPUT_W;
    input_dims_conv49.h = CONVLAYER49_INPUT_H;
    input_dims_conv49.c = CONVLAYER49_IN_CH;
    filter_dims_conv49.w = CONVLAYER49_FILTER_X;
    filter_dims_conv49.h = CONVLAYER49_FILTER_Y;
    output_dims_conv49.w = CONVLAYER49_OUTPUT_W;
    output_dims_conv49.h = CONVLAYER49_OUTPUT_H;
    output_dims_conv49.c = CONVLAYER49_OUT_CH;

    const q31_t *bias_data_conv50 = convlayer1_biases;
    const q7_t *kernel_data_conv50 = convlayer1_weights;

    input_dims_conv50.n = CONVLAYER50_INPUT_BATCHES;
    input_dims_conv50.w = CONVLAYER50_INPUT_W;
    input_dims_conv50.h = CONVLAYER50_INPUT_H;
    input_dims_conv50.c = CONVLAYER50_IN_CH;
    filter_dims_conv50.w = CONVLAYER50_FILTER_X;
    filter_dims_conv50.h = CONVLAYER50_FILTER_Y;
    output_dims_conv50.w = CONVLAYER50_OUTPUT_W;
    output_dims_conv50.h = CONVLAYER50_OUTPUT_H;
    output_dims_conv50.c = CONVLAYER50_OUT_CH;

    const q31_t *bias_data_conv51 = convlayer1_biases;
    const q7_t *kernel_data_conv51 = convlayer1_weights;

    input_dims_conv51.n = CONVLAYER51_INPUT_BATCHES;
    input_dims_conv51.w = CONVLAYER51_INPUT_W;
    input_dims_conv51.h = CONVLAYER51_INPUT_H;
    input_dims_conv51.c = CONVLAYER51_IN_CH;
    filter_dims_conv51.w = CONVLAYER51_FILTER_X;
    filter_dims_conv51.h = CONVLAYER51_FILTER_Y;
    output_dims_conv51.w = CONVLAYER51_OUTPUT_W;
    output_dims_conv51.h = CONVLAYER51_OUTPUT_H;
    output_dims_conv51.c = CONVLAYER51_OUT_CH;

    const q31_t *bias_data_conv52 = convlayer52_biases;
    const q7_t *kernel_data_conv52 = convlayer52_weights;

    input_dims_conv52.n = CONVLAYER52_INPUT_BATCHES;
    input_dims_conv52.w = CONVLAYER52_INPUT_W;
    input_dims_conv52.h = CONVLAYER52_INPUT_H;
    input_dims_conv52.c = CONVLAYER52_IN_CH;
    filter_dims_conv52.w = CONVLAYER52_FILTER_X;
    filter_dims_conv52.h = CONVLAYER52_FILTER_Y;
    output_dims_conv52.w = CONVLAYER52_OUTPUT_W;
    output_dims_conv52.h = CONVLAYER52_OUTPUT_H;
    output_dims_conv52.c = CONVLAYER52_OUT_CH;

    const q31_t *bias_data_conv53 = convlayer1_biases;
    const q7_t *kernel_data_conv53 = convlayer1_weights;

    input_dims_conv53.n = CONVLAYER53_INPUT_BATCHES;
    input_dims_conv53.w = CONVLAYER53_INPUT_W;
    input_dims_conv53.h = CONVLAYER53_INPUT_H;
    input_dims_conv53.c = CONVLAYER53_IN_CH;
    filter_dims_conv53.w = CONVLAYER53_FILTER_X;
    filter_dims_conv53.h = CONVLAYER53_FILTER_Y;
    output_dims_conv53.w = CONVLAYER53_OUTPUT_W;
    output_dims_conv53.h = CONVLAYER53_OUTPUT_H;
    output_dims_conv53.c = CONVLAYER53_OUT_CH;

    const q31_t *bias_data_conv54 = convlayer1_biases;
    const q7_t *kernel_data_conv54 = convlayer1_weights;

    input_dims_conv54.n = CONVLAYER54_INPUT_BATCHES;
    input_dims_conv54.w = CONVLAYER54_INPUT_W;
    input_dims_conv54.h = CONVLAYER54_INPUT_H;
    input_dims_conv54.c = CONVLAYER54_IN_CH;
    filter_dims_conv54.w = CONVLAYER54_FILTER_X;
    filter_dims_conv54.h = CONVLAYER54_FILTER_Y;
    output_dims_conv54.w = CONVLAYER54_OUTPUT_W;
    output_dims_conv54.h = CONVLAYER54_OUTPUT_H;
    output_dims_conv54.c = CONVLAYER54_OUT_CH;

    const q31_t *bias_data_conv55 = convlayer55_biases;
    const q7_t *kernel_data_conv55 = convlayer55_weights;

    input_dims_conv55.n = CONVLAYER55_INPUT_BATCHES;
    input_dims_conv55.w = CONVLAYER55_INPUT_W;
    input_dims_conv55.h = CONVLAYER55_INPUT_H;
    input_dims_conv55.c = CONVLAYER55_IN_CH;
    filter_dims_conv55.w = CONVLAYER55_FILTER_X;
    filter_dims_conv55.h = CONVLAYER55_FILTER_Y;
    output_dims_conv55.w = CONVLAYER55_OUTPUT_W;
    output_dims_conv55.h = CONVLAYER55_OUTPUT_H;
    output_dims_conv55.c = CONVLAYER55_OUT_CH;

    const q31_t *bias_data_conv56 = convlayer1_biases;
    const q7_t *kernel_data_conv56 = convlayer1_weights;

    input_dims_conv56.n = CONVLAYER56_INPUT_BATCHES;
    input_dims_conv56.w = CONVLAYER56_INPUT_W;
    input_dims_conv56.h = CONVLAYER56_INPUT_H;
    input_dims_conv56.c = CONVLAYER56_IN_CH;
    filter_dims_conv56.w = CONVLAYER56_FILTER_X;
    filter_dims_conv56.h = CONVLAYER56_FILTER_Y;
    output_dims_conv56.w = CONVLAYER56_OUTPUT_W;
    output_dims_conv56.h = CONVLAYER56_OUTPUT_H;
    output_dims_conv56.c = CONVLAYER56_OUT_CH;

    const q31_t *bias_data_conv57 = convlayer1_biases;
    const q7_t *kernel_data_conv57 = convlayer1_weights;

    input_dims_conv57.n = CONVLAYER57_INPUT_BATCHES;
    input_dims_conv57.w = CONVLAYER57_INPUT_W;
    input_dims_conv57.h = CONVLAYER57_INPUT_H;
    input_dims_conv57.c = CONVLAYER57_IN_CH;
    filter_dims_conv57.w = CONVLAYER57_FILTER_X;
    filter_dims_conv57.h = CONVLAYER57_FILTER_Y;
    output_dims_conv57.w = CONVLAYER57_OUTPUT_W;
    output_dims_conv57.h = CONVLAYER57_OUTPUT_H;
    output_dims_conv57.c = CONVLAYER57_OUT_CH;

    const q31_t *bias_data_conv58 = convlayer1_biases;
    const q7_t *kernel_data_conv58 = convlayer1_weights;

    input_dims_conv58.n = CONVLAYER58_INPUT_BATCHES;
    input_dims_conv58.w = CONVLAYER58_INPUT_W;
    input_dims_conv58.h = CONVLAYER58_INPUT_H;
    input_dims_conv58.c = CONVLAYER58_IN_CH;
    filter_dims_conv58.w = CONVLAYER58_FILTER_X;
    filter_dims_conv58.h = CONVLAYER58_FILTER_Y;
    output_dims_conv58.w = CONVLAYER58_OUTPUT_W;
    output_dims_conv58.h = CONVLAYER58_OUTPUT_H;
    output_dims_conv58.c = CONVLAYER58_OUT_CH;


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

    stride2_conv_params.padding.w = CONVLAYER1_PAD_X;
    stride2_conv_params.padding.h = CONVLAYER1_PAD_Y;
    stride2_conv_params.stride.w = 2;
    stride2_conv_params.stride.h = 2;

    stride2_conv_params.input_offset = CONVLAYER1_INPUT_OFFSET;
    stride2_conv_params.output_offset = CONVLAYER1_OUTPUT_OFFSET;
    stride2_conv_params.activation.min = CONVLAYER1_OUT_ACTIVATION_MIN;
    stride2_conv_params.activation.max = CONVLAYER1_OUT_ACTIVATION_MAX;

    q7_t* first_layer_outbuf = malloc(CONVLAYER1_OUT_CH*CONVLAYER1_OUTPUT_W*CONVLAYER1_OUTPUT_H*sizeof(q7_t));
    q7_t* first_layer_inbuf = malloc(CONVLAYER1_IN_CH*CONVLAYER1_INPUT_W*CONVLAYER1_INPUT_H*sizeof(q7_t));

    arm_status result_arm;

    q7_t* actbuf1 = malloc(16*16*192); //create the first general activation buffer, 16*16*64 should be enough for rest layers
    q7_t* actbuf2 = malloc(16*16*192); //create the first general activation buffer, 16*16*64 should be enough for rest layers

    //code for layers
    //First copy the inputs from flash to ram
    memcpy(actbuf1, input_data_conv1, CONVLAYER1_IN_CH*CONVLAYER1_INPUT_W*CONVLAYER1_INPUT_H);

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv1, &filter_dims_conv1);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = arm_convolve_s8(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv1,
                                        actbuf1,
                                        &filter_dims_conv1,
                                        kernel_data_conv1,
                                        &bias_dims_conv1,
                                        bias_data_conv1,
                                        &output_dims_conv1,
                                        actbuf2);

    free(ctx.buf);

    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv2, &filter_dims_conv2);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    //result_arm = lut_conv_zdim_v1(&ctx,
    result_arm = lut_conv_zdim_v1(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv2,
                                        actbuf2,
                                        &filter_dims_conv2,
                                        mobilenet_v2_index_layer_2,
                                        &bias_dims_conv2,
                                        bias_data_conv2,
                                        &output_dims_conv2,
                                        lut_data,
                                        actbuf1);

    free(ctx.buf);


    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv3, &filter_dims_conv3);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    //result_arm = lut_conv_zdim_v1(&ctx,
    result_arm = arm_depthwise_conv_s8(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv3,
                                        actbuf1,
                                        &filter_dims_conv3,
                                        kernel_data_conv3,
                                        &bias_dims_conv3,
                                        bias_data_conv3,
                                        &output_dims_conv3,
                                        actbuf2);

    free(ctx.buf);    

    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv4, &filter_dims_conv4);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    //result_arm = lut_conv_zdim_v1(&ctx,
    result_arm = lut_conv_zdim_v1(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv4,
                                        actbuf2,
                                        &filter_dims_conv4,
                                        mobilenet_v2_index_layer_4,
                                        &bias_dims_conv4,
                                        bias_data_conv4,
                                        &output_dims_conv4,
                                        lut_data,
                                        actbuf1);

    free(ctx.buf); 
    
    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv5, &filter_dims_conv5);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    //result_arm = lut_conv_zdim_v1(&ctx,
    result_arm = lut_conv_zdim_v1(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv5,
                                        actbuf1,
                                        &filter_dims_conv5,
                                        mobilenet_v2_index_layer_5,
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
                                        mobilenet_v2_index_layer_6,
                                        &bias_dims_conv6,
                                        bias_data_conv6,
                                        &output_dims_conv6,
                                        lut_data,
                                        actbuf1);

    free(ctx.buf); 
    
    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv7, &filter_dims_conv7);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = arm_depthwise_conv_s8(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv7,
                                        actbuf1,
                                        &filter_dims_conv7,
                                        kernel_data_conv7,
                                        &bias_dims_conv7,
                                        bias_data_conv7,
                                        &output_dims_conv7,
                                        actbuf2);

    free(ctx.buf); 
    
    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv8, &filter_dims_conv8);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv8,
                                        actbuf2,
                                        &filter_dims_conv8,
                                        mobilenet_v2_index_layer_8,
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
                                        mobilenet_v2_index_layer_9,
                                        &bias_dims_conv9,
                                        bias_data_conv9,
                                        &output_dims_conv9,
                                        lut_data,
                                        actbuf2);

    free(ctx.buf); 

			buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv10, &filter_dims_conv10);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v1(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv10,
                                        actbuf2,
                                        &filter_dims_conv10,
                                        mobilenet_v2_index_layer_10,
                                        &bias_dims_conv10,
                                        bias_data_conv10,
                                        &output_dims_conv10,
                                        lut_data,
                                        actbuf1);

    free(ctx.buf); 
    
    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv11, &filter_dims_conv11);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = arm_depthwise_conv_s8(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv11,
                                        actbuf1,
                                        &filter_dims_conv11,
                                        kernel_data_conv11,
                                        &bias_dims_conv11,
                                        bias_data_conv11,
                                        &output_dims_conv11,
                                        actbuf2);

    free(ctx.buf); 
    
    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv12, &filter_dims_conv12);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv12,
                                        actbuf2,
                                        &filter_dims_conv12,
                                        mobilenet_v2_index_layer_12,
                                        &bias_dims_conv12,
                                        bias_data_conv12,
                                        &output_dims_conv12,
                                        lut_data,
                                        actbuf1);

    free(ctx.buf); 
    
    buf_size = arm_convolve_s8_get_buffer_size(&input_dims_conv13, &filter_dims_conv13);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v1(&ctx,
                                        &conv_params,
                                        &quant_params,
                                        &input_dims_conv13,
                                        actbuf1,
                                        &filter_dims_conv13,
                                        mobilenet_v2_index_layer_13,
                                        &bias_dims_conv13,
                                        bias_data_conv13,
                                        &output_dims_conv13,
                                        lut_data,
                                        actbuf2);

    free(ctx.buf); 

    ctx.size = 0;
    
    result_arm = arm_depthwise_conv_s8(&ctx,&conv_params,&quant_params,&input_dims_conv14,actbuf2,&filter_dims_conv14,kernel_data_conv14,&bias_dims_conv14,bias_data_conv14,&output_dims_conv14,actbuf1);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv15,actbuf1,&filter_dims_conv15,mobilenet_v2_index_layer_15,&bias_dims_conv15,bias_data_conv15,&output_dims_conv15,lut_data,actbuf2);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v1(&ctx,&conv_params,&quant_params,&input_dims_conv16,actbuf2,&filter_dims_conv16,mobilenet_v2_index_layer_16,&bias_dims_conv16,bias_data_conv16,&output_dims_conv16,lut_data,actbuf1);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v1(&ctx,&conv_params,&quant_params,&input_dims_conv17,actbuf1,&filter_dims_conv17,mobilenet_v2_index_layer_17,&bias_dims_conv17,bias_data_conv17,&output_dims_conv17,lut_data,actbuf2);
    
    ctx.size = 0;
    
    result_arm = arm_depthwise_conv_s8(&ctx,&conv_params,&quant_params,&input_dims_conv18,actbuf2,&filter_dims_conv18,kernel_data_conv18,&bias_dims_conv18,bias_data_conv18,&output_dims_conv18,actbuf1);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv19,actbuf1,&filter_dims_conv19,mobilenet_v2_index_layer_19,&bias_dims_conv19,bias_data_conv19,&output_dims_conv19,lut_data,actbuf2);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v1(&ctx,&conv_params,&quant_params,&input_dims_conv20,actbuf2,&filter_dims_conv20,mobilenet_v2_index_layer_20,&bias_dims_conv20,bias_data_conv20,&output_dims_conv20,lut_data,actbuf1);
    
    ctx.size = 0;
    
    result_arm = arm_depthwise_conv_s8(&ctx,&conv_params,&quant_params,&input_dims_conv21,actbuf1,&filter_dims_conv21,kernel_data_conv21,&bias_dims_conv21,bias_data_conv21,&output_dims_conv21,actbuf2);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv22,actbuf2,&filter_dims_conv22,mobilenet_v2_index_layer_22,&bias_dims_conv22,bias_data_conv22,&output_dims_conv22,lut_data,actbuf1);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v1(&ctx,&conv_params,&quant_params,&input_dims_conv23,actbuf1,&filter_dims_conv23,mobilenet_v2_index_layer_23,&bias_dims_conv23,bias_data_conv23,&output_dims_conv23,lut_data,actbuf2);
    
    ctx.size = 0;
    
    result_arm = arm_depthwise_conv_s8(&ctx,&stride2_conv_params,&quant_params,&input_dims_conv24,actbuf2,&filter_dims_conv24,kernel_data_conv24,&bias_dims_conv24,bias_data_conv24,&output_dims_conv24,actbuf1);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv25,actbuf1,&filter_dims_conv25,mobilenet_v2_index_layer_25,&bias_dims_conv25,bias_data_conv25,&output_dims_conv25,lut_data,actbuf2);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv26,actbuf2,&filter_dims_conv26,mobilenet_v2_index_layer_26,&bias_dims_conv26,bias_data_conv26,&output_dims_conv26,lut_data,actbuf1);
    
    ctx.size = 0;
    
    result_arm = arm_depthwise_conv_s8(&ctx,&conv_params,&quant_params,&input_dims_conv27,actbuf1,&filter_dims_conv27,kernel_data_conv27,&bias_dims_conv27,bias_data_conv27,&output_dims_conv27,actbuf2);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv28,actbuf2,&filter_dims_conv28,mobilenet_v2_index_layer_28,&bias_dims_conv28,bias_data_conv28,&output_dims_conv28,lut_data,actbuf1);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv29,actbuf1,&filter_dims_conv29,mobilenet_v2_index_layer_29,&bias_dims_conv29,bias_data_conv29,&output_dims_conv29,lut_data,actbuf2);
    
    ctx.size = 0;
    
    result_arm = arm_depthwise_conv_s8(&ctx,&conv_params,&quant_params,&input_dims_conv30,actbuf2,&filter_dims_conv30,kernel_data_conv30,&bias_dims_conv30,bias_data_conv30,&output_dims_conv30,actbuf1);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv31,actbuf1,&filter_dims_conv31,mobilenet_v2_index_layer_31,&bias_dims_conv31,bias_data_conv31,&output_dims_conv31,lut_data,actbuf2);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv32,actbuf2,&filter_dims_conv32,mobilenet_v2_index_layer_32,&bias_dims_conv32,bias_data_conv32,&output_dims_conv32,lut_data,actbuf1);
    
    ctx.size = 0;
    
    result_arm = arm_depthwise_conv_s8(&ctx,&conv_params,&quant_params,&input_dims_conv33,actbuf1,&filter_dims_conv33,kernel_data_conv33,&bias_dims_conv33,bias_data_conv33,&output_dims_conv33,actbuf2);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv34,actbuf2,&filter_dims_conv34,mobilenet_v2_index_layer_34,&bias_dims_conv34,bias_data_conv34,&output_dims_conv34,lut_data,actbuf1);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv35,actbuf1,&filter_dims_conv35,mobilenet_v2_index_layer_35,&bias_dims_conv35,bias_data_conv35,&output_dims_conv35,lut_data,actbuf2);
    
    ctx.size = 0;
    
    result_arm = arm_depthwise_conv_s8(&ctx,&conv_params,&quant_params,&input_dims_conv36,actbuf2,&filter_dims_conv36,kernel_data_conv36,&bias_dims_conv36,bias_data_conv36,&output_dims_conv36,actbuf1);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv37,actbuf1,&filter_dims_conv37,mobilenet_v2_index_layer_37,&bias_dims_conv37,bias_data_conv37,&output_dims_conv37,lut_data,actbuf2);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv38,actbuf2,&filter_dims_conv38,mobilenet_v2_index_layer_38,&bias_dims_conv38,bias_data_conv38,&output_dims_conv38,lut_data,actbuf1);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv39,actbuf1,&filter_dims_conv39,mobilenet_v2_index_layer_39,&bias_dims_conv39,bias_data_conv39,&output_dims_conv39,lut_data,actbuf2);
    
    ctx.size = 0;
    
    result_arm = arm_depthwise_conv_s8(&ctx,&conv_params,&quant_params,&input_dims_conv40,actbuf2,&filter_dims_conv40,kernel_data_conv40,&bias_dims_conv40,bias_data_conv40,&output_dims_conv40,actbuf1);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv41,actbuf1,&filter_dims_conv41,mobilenet_v2_index_layer_41,&bias_dims_conv41,bias_data_conv41,&output_dims_conv41,lut_data,actbuf2);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv42,actbuf2,&filter_dims_conv42,mobilenet_v2_index_layer_42,&bias_dims_conv42,bias_data_conv42,&output_dims_conv42,lut_data,actbuf1);
    
    ctx.size = 0;
    
    result_arm = arm_depthwise_conv_s8(&ctx,&conv_params,&quant_params,&input_dims_conv43,actbuf1,&filter_dims_conv43,kernel_data_conv43,&bias_dims_conv43,bias_data_conv43,&output_dims_conv43,actbuf2);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv44,actbuf2,&filter_dims_conv44,mobilenet_v2_index_layer_44,&bias_dims_conv44,bias_data_conv44,&output_dims_conv44,lut_data,actbuf1);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv45,actbuf1,&filter_dims_conv45,mobilenet_v2_index_layer_45,&bias_dims_conv45,bias_data_conv45,&output_dims_conv45,lut_data,actbuf2);
    
    ctx.size = 0;
    
    result_arm = arm_depthwise_conv_s8(&stride2_conv_params,&conv_params,&quant_params,&input_dims_conv46,actbuf2,&filter_dims_conv46,kernel_data_conv46,&bias_dims_conv46,bias_data_conv46,&output_dims_conv46,actbuf1);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv47,actbuf1,&filter_dims_conv47,mobilenet_v2_index_layer_47,&bias_dims_conv47,bias_data_conv47,&output_dims_conv47,lut_data,actbuf2);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv48,actbuf2,&filter_dims_conv48,mobilenet_v2_index_layer_48,&bias_dims_conv48,bias_data_conv48,&output_dims_conv48,lut_data,actbuf1);
    
    ctx.size = 0;
    
    result_arm = arm_depthwise_conv_s8(&ctx,&conv_params,&quant_params,&input_dims_conv49,actbuf1,&filter_dims_conv49,kernel_data_conv49,&bias_dims_conv49,bias_data_conv49,&output_dims_conv49,actbuf2);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv50,actbuf2,&filter_dims_conv50,mobilenet_v2_index_layer_50,&bias_dims_conv50,bias_data_conv50,&output_dims_conv50,lut_data,actbuf1);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv51,actbuf1,&filter_dims_conv51,mobilenet_v2_index_layer_51,&bias_dims_conv51,bias_data_conv51,&output_dims_conv51,lut_data,actbuf2);
    
    ctx.size = 0;
    
    result_arm = arm_depthwise_conv_s8(&ctx,&conv_params,&quant_params,&input_dims_conv52,actbuf2,&filter_dims_conv52,kernel_data_conv52,&bias_dims_conv52,bias_data_conv52,&output_dims_conv52,actbuf1);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv53,actbuf1,&filter_dims_conv53,mobilenet_v2_index_layer_53,&bias_dims_conv53,bias_data_conv53,&output_dims_conv53,lut_data,actbuf2);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv54,actbuf2,&filter_dims_conv54,mobilenet_v2_index_layer_54,&bias_dims_conv54,bias_data_conv54,&output_dims_conv54,lut_data,actbuf1);
    
    ctx.size = 0;
    
    result_arm = arm_depthwise_conv_s8(&ctx,&conv_params,&quant_params,&input_dims_conv55,actbuf1,&filter_dims_conv55,kernel_data_conv55,&bias_dims_conv55,bias_data_conv55,&output_dims_conv55,actbuf2);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv56,actbuf2,&filter_dims_conv56,mobilenet_v2_index_layer_56,&bias_dims_conv56,bias_data_conv56,&output_dims_conv56,lut_data,actbuf1);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv57,actbuf1,&filter_dims_conv57,mobilenet_v2_index_layer_57,&bias_dims_conv57,bias_data_conv57,&output_dims_conv57,lut_data,actbuf2);
    
    ctx.size = 0;
    
    result_arm = lut_conv_zdim_v2_double_lookup(&ctx,&conv_params,&quant_params,&input_dims_conv58,actbuf2,&filter_dims_conv58,mobilenet_v2_index_layer_58,&bias_dims_conv58,bias_data_conv58,&output_dims_conv58,lut_data,actbuf1);
    free(actbuf1);
		free(actbuf2);
}

int main(){
	  HAL_Init();
    SystemClock_Config();
    while(1){
        conv_fw_mobilenet_v2();
    }
return 0;
}