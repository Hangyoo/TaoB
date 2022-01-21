import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class Draw:
    bound_x = []
    bound_y = []

    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.plt = plt

    def draw_points(self, x, y):
        self.ax.plot(x, y, 'ro')

    def set_xybd(self, x_bd, y_bd):
        self.ax.axis([x_bd[0], x_bd[1], y_bd[0], y_bd[1]])

    def draw_text(self, x , y, text, size = 8):
        self.ax.text(x, y, text, fontsize = size)

    def draw_line(self, point_from, point_to):
        line = [(point_from[0], point_from[1]), (point_to[0], point_to[1])]
        (line_xs, line_ys) = zip(*line)
        self.ax.add_line(Line2D(line_xs, line_ys, linewidth=1, color='blue'))

