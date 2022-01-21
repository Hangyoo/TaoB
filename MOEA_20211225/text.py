def function(x1, x2, x3, x4, x5, x6):
    f1 = 1065.48 - 470.959 * x1 + 769.657 * x2 - 232.594 * x3 + 45.2741 * x4 - 146.575 * x5 + 19.2488 * x6 + \
         165.029 * x1 * x2 + 114.252 * x1 * x3 + 216.278 * x1 * x4 - 314.353 * x1 * x5 + 168.586 * x1 * x6 - \
         370.033 * x2 * x3 - 526.56 * x2 * x4 + 642.312 * x2 * x5 - 180.901 * x2 * x6 + 414.795 * x3 * x4 - \
         517.495 * x3 * x5 + 66.9567 * x3 * x6 + 669.121 * x4 * x5 - 302.959 * x4 * x6 + 277.075 * x5 * x6 - \
         48.2285 * x1 ** 2 - 80.8058 * x2 ** 2 + 197.09 * x3 ** 2 - 253.144 * x4 ** 2 - 315.906 * x5 ** 2 - 25.1113 * x6 ** 2

    # 出温风温
    f2 = 464.612 - 400.363 * x1 + 674.382 * x2 - 318.308 * x3 - 137.079 * x4 + 163.213 * x5 - 144.573 * x6 + \
         74.0726 * x1 * x2 + 202.806 * x1 * x3 - 18.3681 * x1 * x4 - 28.6151 * x1 * x5 + 3.61567 * x1 * x6 - \
         426.006 * x2 * x3 + 56.2006 * x2 * x4 + 37.6291 * x2 * x5 + 72.1038 * x2 * x6 - 23.1415 * x3 * x4 + \
         48.7089 * x3 * x5 - 152.869 * x3 * x6 - 196.325 * x4 * x5 + 126.504 * x4 * x6 + 89.7283 * x5 * x6 + \
         3.30492 * x1 ** 2 - 112.089 * x2 ** 2 + 259.391 * x3 ** 2 + 66.221 * x4 ** 2 - 9.25738 * x5 ** 2 - 35.7172 * x6 ** 2

    # 出口温度
    f3 = 332.285 - 3.76115 * x1 - 57.2402 * x2 - 7.03187 * x3 - 185.967 * x4 + 164.27 * x5 - 138.834 * x6 - \
         31.0463 * x1 * x2 + 63.3426 * x1 * x3 - 77.3782 * x1 * x4 - 12.2009 * x1 * x5 + 50.492 * x1 * x6 - \
         98.1579 * x2 * x3 + 117.233 * x2 * x4 + 45.8926 * x2 * x5 - 86.4151 * x2 * x6 + 9.31394 * x3 * x4 - \
         26.8711 * x3 * x5 - 12.3141 * x3 * x6 - 305.437 * x4 * x5 + 228.191 * x4 * x6 - 50.4836 * x5 * x6 + \
         9.53542 * x1 ** 2 + 32.4082 * x2 ** 2 + 32.4597 * x3 ** 2 + 67.2034 * x4 ** 2 + 132.538 * x5 ** 2 - 30.5995 * x6 ** 2

    print(f"f1:{f1},f2:{f2},f3:{f3}")

if __name__ == "__main__":

    function(2.6739990234375,2.7235595703125,2.7068359375,2.301806640625,2.061572265625,2.612109375)