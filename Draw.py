import numpy as np 
import matplotlib.pyplot as plt 
import math  

Hardware_cal = []
Hardware_bw = []


# Hardware Spec
Performance_HW = 11340 * pow(10,9) # GPU
# Performance_HW = 2260.8 * pow(10,9) # CPU
# BW_NAND = 560 * pow(10,6)
BW_DRAM = 484 * pow(10,9) # GPU
# BW_DRAM = 94 * pow(10,9) # CPU
# Hardware_intensity_NAND = Performance_HW / BW_NAND
Hardware_intensity_DRAM = Performance_HW / BW_DRAM

# The start point of line
Global_intensity = 0.01
# Global_NAND = Global_intensity * BW_NAND
Global_DRAM = Global_intensity * BW_DRAM


# print(f'Global: {Global_intensity}')
# print(f'Global_NAND: {Global_NAND}')
# print(f'Global_DRAM: {Global_DRAM}')
# print(f'NAND: {Hardware_intensity_NAND}')
# print(f'DRAM: {Hardware_intensity_DRAM}')

# NAND
# NAND_inten = []
# NAND_inten.append(Global_intensity)
# NAND_inten.append(Hardware_intensity_NAND)
# NAND_perf = []
# NAND_perf.append(Global_NAND)
# NAND_perf.append(Performance_HW)

# DRAM
DRAM_inten = []
DRAM_inten.append(Global_intensity)
DRAM_inten.append(Hardware_intensity_DRAM)
DRAM_perf = []
DRAM_perf.append(Global_DRAM)
DRAM_perf.append(Performance_HW)

# To plot the roof
Global_peak_perf = []
Global_peak_perf.append(Performance_HW)
Global_peak_perf.append(Performance_HW)
Global_peak_inten = []
Global_peak_inten.append(DRAM_inten[1])
Global_peak_inten.append(200)

# Application
sls_inten = 0.25
# conv_inten = 136.5
# fc_inten = 0.5
# vgg16_inten = 45.8

# Different Batch SIze
# fc_inten_1 = 0.5
# fc_inten_2 = 1
# fc_inten_4 = 2
# fc_inten_8 = 3.8
# fc_inten_16 = 7.5
# fc_inten_32 = 15.1
# fc_inten_64 = 30
# fc_inten_128 = 58.9
# sls_NAND_peak_perf = sls_inten * BW_NAND
sls_DRAM_peak_perf = min(sls_inten * BW_DRAM, Performance_HW)
# conv_DRAM_peak_perf = min(conv_inten * BW_DRAM,Performance_HW)
# vgg16_DRAM_peak_perf = min(vgg16_inten * BW_DRAM,Performance_HW)
# fc_DRAM_peak_perf = min(fc_inten * BW_DRAM,Performance_HW)
# sls_actual = 0.002 * pow(10,9)

# Real Performance
# conv_real_perf_gpu = 8517.75 * pow(10,9)
# conv_real_perf_cpu = 52.68 * pow(10,9)
# fc_real_perf_gpu = 177.82 * pow(10,9)
# fc_real_perf_cpu = 5.54 * pow(10,9)
sls_real_perf_gpu = 0.082 * pow(10,9)
sls_real_perf_cpu = 0.46 * pow(10,9)

# fc_DRAM_peak_perf_1 = min(fc_inten_1 * BW_DRAM, Performance_HW)
# fc_DRAM_peak_perf_2 = min(fc_inten_2 * BW_DRAM, Performance_HW)
# fc_DRAM_peak_perf_4 = min(fc_inten_4 * BW_DRAM, Performance_HW)
# fc_DRAM_peak_perf_8 = min(fc_inten_8 * BW_DRAM, Performance_HW)
# fc_DRAM_peak_perf_16 = min(fc_inten_16 * BW_DRAM, Performance_HW)
# fc_DRAM_peak_perf_32 = min(fc_inten_32 * BW_DRAM, Performance_HW)
# fc_DRAM_peak_perf_64 = min(fc_inten_64 * BW_DRAM, Performance_HW)
# fc_DRAM_peak_perf_128 = min(fc_inten_128 * BW_DRAM, Performance_HW)


# vgg16_real_perf_1 = 5525.09 * pow(10,9)
# vgg16_real_perf_1 = 94.24 * pow(10,9)
# fc_real_perf_1 = 5.54 * pow(10,9)
# fc_real_perf_2 = 9.58 * pow(10,9)
# fc_real_perf_4 = 9.62 * pow(10,9)
# fc_real_perf_8 = 12.63 * pow(10,9)
# fc_real_perf_16 = 32.7 * pow(10,9)
# fc_real_perf_32 = 56.91 * pow(10,9)
# fc_real_perf_64 = 91.91 * pow(10,9)
# fc_real_perf_128 = 131.34 * pow(10,9)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# line1, = ax.plot(NAND_inten, NAND_perf, color='blue', lw=2)
line2, = ax.plot(DRAM_inten, DRAM_perf, color='blue', lw=2)
line3, = ax.plot(Global_peak_inten, Global_peak_perf, color='blue', lw=2)
sls_p, = ax.plot(sls_inten, sls_DRAM_peak_perf, label='Peak Performance', marker='v', color='black')
sls_r, = ax.plot(sls_inten, sls_real_perf_cpu, label='Real Performance', marker='o', color='black')
# conv_p, = ax.plot(conv_inten, conv_DRAM_peak_perf, label='Peak Performance', marker='v', color='black')
# conv_r, = ax.plot(conv_inten, conv_real_perf_cpu, label='Real Performance', marker='o', color='black')
# fc_p, = ax.plot(fc_inten, fc_DRAM_peak_perf, label='Peak Performance', marker='v', color='black')
# fc_r, = ax.plot(fc_inten, fc_real_perf_gpu, label='Real Performance', marker='o', color='black')
# pt_p_1, = ax.plot(vgg16_inten, vgg16_real_perf_1, label='Real Performance', marker='o', color='black')
# pt_r_1, = ax.plot(vgg16_inten, vgg16_DRAM_peak_perf, label='Peak Performance', marker='v', color='black')
# pt_p_1, = ax.plot(fc_inten_1, fc_DRAM_peak_perf_1, label='Peak Performance', marker='v', color='black')
# pt_p_2, = ax.plot(fc_inten_2, fc_DRAM_peak_perf_2, marker='v', color='black')
# pt_p_4, = ax.plot(fc_inten_4, fc_DRAM_peak_perf_4, marker='v', color='black')
# pt_p_8, = ax.plot(fc_inten_8, fc_DRAM_peak_perf_8, marker='v', color='black')
# pt_p_16, = ax.plot(fc_inten_16, fc_DRAM_peak_perf_16, marker='v', color='black')
# pt_p_32, = ax.plot(fc_inten_32, fc_DRAM_peak_perf_32, marker='v', color='black')
# pt_p_64, = ax.plot(fc_inten_64, fc_DRAM_peak_perf_64, marker='v', color='black')
# pt_p_128, = ax.plot(fc_inten_128, fc_DRAM_peak_perf_128, marker='v', color='black')
# pt_r_1, = ax.plot(fc_inten_1, fc_real_perf_1, label='Real Performance', marker='o', color='black')
# pt_r_2, = ax.plot(fc_inten_2, fc_real_perf_2, marker='o', color='black')
# pt_r_4, = ax.plot(fc_inten_4, fc_real_perf_4, marker='o', color='black')
# pt_r_8, = ax.plot(fc_inten_8, fc_real_perf_8, marker='o', color='black')
# pt_r_16, = ax.plot(fc_inten_16, fc_real_perf_16, marker='o', color='black')
# pt_r_32, = ax.plot(fc_inten_32, fc_real_perf_32, marker='o', color='black')
# pt_r_64, = ax.plot(fc_inten_64, fc_real_perf_64, marker='o', color='black')
# pt_r_128, = ax.plot(fc_inten_128, fc_real_perf_128, marker='o', color='black')


# pt3, = ax.plot(fc_inten, fc_DRAM_peak_perf, label='FC', marker='x', color='black')
# pt2, = ax.plot(sls_inten, sls_actual, marker='o', color='black')
ax.set_yscale('log', base=10)
ax.set_xscale('log', base=10)
plt.xlim(0.1,200)
plt.ylim(pow(10,7), pow(10,14))
plt.xlabel('Operational Intensity (FLOP/Byte)', fontsize=10)
plt.ylabel('Performance (FLOPS)', fontsize=10)
plt.legend(loc='upper left')
# plt.show()
plt.savefig('Roofline_sls_GPU.png')