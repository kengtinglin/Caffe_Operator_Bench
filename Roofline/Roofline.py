import numpy as np 
import matplotlib.pyplot as plt 
import math
from utils.utils import config_parse
from Calculator import get_intensity


def Roofline(app_inten):
    Hardware_cal = []
    Hardware_bw = []

    # Hardware Spec
    Performance_HW = 11340 * pow(10,9) # GPU 1080Ti
    # Performance_HW = 2260.8 * pow(10,9) # CPU Intel 10940
    BW_DRAM = 484 * pow(10,9) # GPU
    # BW_DRAM = 94 * pow(10,9) # CPU
    # Hardware_intensity_NAND = Performance_HW / BW_NAND
    Hardware_intensity_DRAM = round(Performance_HW / BW_DRAM,2)

    # The start point of line
    Global_intensity = 0.01
    Global_DRAM = Global_intensity * BW_DRAM

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
    Global_peak_inten.append(app_inten+50)

    # Application
    app_theo_peak_perf = min(app_inten * BW_DRAM, Performance_HW)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    Roof_DRAM, = ax.plot(DRAM_inten, DRAM_perf, color='blue', lw=3)
    Roof_HWPeak, = ax.plot(Global_peak_inten, Global_peak_perf, color='blue', lw=3)
    app_peak, = ax.plot(app_inten, app_theo_peak_perf, label='Theoretical Peak Performance', marker='v', color='black', markersize=8)

    ax.set_yscale('log', base=10)
    ax.set_xscale('log', base=10)
    plt.xlim(0.1,200)
    plt.ylim(pow(10,7), pow(10,14))
    plt.xlabel('Operational Intensity (FLOP/Byte)', fontsize=10)
    plt.ylabel('Performance (FLOPS)', fontsize=10)
    plt.legend(loc='upper left')
    # plt.show()
    plt.savefig('Roofline_GPU.png')

if __name__ == "__main__":
    args = config_parse()
    Roofline(get_intensity(args))
