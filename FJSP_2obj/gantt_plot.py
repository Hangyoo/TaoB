import matplotlib.pyplot as plt
from FJSP_2obj import decoding,readtext,gantt

'''
程序使用方法：
1.想绘制哪个问题的Gantt图，就将路径切换至那个问题的数据路径(即更新patch的地址)
2.将chorm的数据,换为你想绘制的点的编码
'''

patch = r'C:\Users\Hangyu\PycharmProjects\TaoB\FJSPMK\Benchmark\Mk07.fjs'
# patch = r'C:\Users\Hangyu\Desktop\JmetalTB\FJSP_2obj\realworld.fjs'

chrom = ([19, 4, 19, 12, 14, 7, 5, 1, 0, 8, 2, 14, 3, 18, 3, 7, 15, 15, 10, 1, 5, 16, 0, 4, 6, 1, 11, 10, 8, 10, 14, 8, 13, 14, 18, 11, 8, 1, 13, 10, 17, 12, 18, 18, 19, 0, 0, 9, 12, 19, 4, 19, 10, 9, 9, 12, 2, 8, 14, 13, 18, 2, 17, 4, 11, 5, 3, 9, 15, 17, 12, 13, 11, 16, 5, 6, 15, 6, 11, 16, 7, 6, 9, 3, 5, 15, 16, 3, 17, 6, 16, 13, 1, 7, 2, 4, 2, 0, 17, 7], [1, 0, 0, 0, 3, 0, 4, 0, 2, 2, 2, 0, 1, 0, 0, 1, 3, 3, 0, 1, 0, 1, 3, 2, 0, 0, 1, 1, 0, 1, 0, 1, 1, 2, 0, 1, 1, 2, 0, 0, 0, 0, 2, 3, 1, 1, 0, 1, 0, 4, 0, 0, 3, 0, 0, 1, 2, 0, 2, 2, 1, 0, 3, 0, 1, 0, 2, 3, 1, 0, 1, 1, 1, 3, 1, 1, 0, 0, 1, 0, 4, 3, 2, 1, 4, 0, 1, 1, 0, 1, 1, 3, 0, 0, 0, 0, 1, 0, 0, 3], [1, 1, 4, 0, 2, 2, 0, 1, 0, 4, 2, 3, 2, 3, 1, 2, 1, 1, 4, 0, 2, 1, 0, 1, 3, 2, 1, 2, 3, 4, 0, 2, 1, 3, 3, 2, 2, 2, 4, 1, 0, 4, 3, 0, 3, 4, 0, 1, 2, 0, 0, 1, 1, 4, 0, 4, 1, 1, 4, 4, 0, 0, 3, 4, 1, 4, 1, 3, 4, 3, 3, 1, 4, 4, 0, 2, 1, 0, 0, 4, 4, 0, 2, 3, 1, 2, 4, 1, 1, 3, 2, 4, 2, 2, 1, 3, 4, 1, 4, 3])


parameters = readtext.Readtext(patch).readtext()
# 绘制甘特图
gantt_data = decoding.translate_decoded_to_gantt(
    decoding.decodeBenchmark(parameters, chrom[0], chrom[1]))
title = "Flexible Job Shop Solution Processing Time (NSGA-II)"  # 甘特图title (英文)
print(gantt_data)
gantt.draw_chart(gantt_data, title, 15)  # 调节数字可以更改图片中标题字体
