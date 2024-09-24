from matplotlib.widgets import Slider, Button
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.ticker import FormatStrFormatter
import mpl_toolkits.axes_grid1
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np


class SqueezedNorm(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, mid=0, s1=2, s2=2, clip=False):
        self.vmin = vmin  # minimum value
        self.mid = mid  # middle value
        self.vmax = vmax  # maximum value
        self.s1 = s1
        self.s2 = s2
        def f(x, zero, vmax, s): return np.abs((x-zero)/(vmax-zero+0.0001))**(1./s+0.0001)*0.5
        self.g = lambda x, zero, vmin, vmax, s1, s2: f(x, zero, vmax, s1)*(x >= zero) - \
            f(x, zero, vmin, s2)*(x < zero)+0.5
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        r = self.g(value, self.mid, self.vmin, self.vmax, self.s1, self.s2)
        return np.ma.masked_array(r)


def MultipleImagesAndRectangle(fig, axes, data, vmin, vmax, norm, cmap='bwr', border=False, draw_rect=True, aspect='equal'):
    imgs = []
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()

    for i, ax in enumerate(axes):
        imgs.append(ax.imshow(data[i], vmin=vmin, vmax=vmax, aspect=aspect, cmap=cmap, norm=norm))
        ax.xaxis.set_ticklabels([])
        ax.axes.get_xaxis().set_ticks([])
        ax.yaxis.set_ticklabels([])
        ax.axes.get_yaxis().set_ticks([])
        if not border:
            ax.axis('off')
    return imgs


def show_heatmap(maps):
    threshold = 500
    m_min, m_max = maps.min(), maps.max()
    if m_max > threshold:
        m_max = threshold

    # This value depends on the input image, so set the "mid" value appropriately.
    # You can adjust this value by moving the slide bar.
    mid = m_max * 0.3  

    x = np.linspace(m_min, m_max, 100)
    norm = SqueezedNorm(vmin=m_min, vmax=m_max, mid=mid)

    plt.figure()
    im = plt.imshow(maps, vmin=m_min, vmax=m_max,
                    cmap='hot', norm=norm)

    cbar = plt.colorbar()

    midax = plt.axes([0.1, 0.04, 0.2, 0.03], facecolor="lightblue")
    s1ax = plt.axes([0.4, 0.04, 0.2, 0.03], facecolor="lightblue")
    s2ax = plt.axes([0.7, 0.04, 0.2, 0.03], facecolor="lightblue")

    mid = Slider(midax, 'Midpoint', x[0], x[-1], valinit=mid)
    s1 = Slider(s1ax, 'S1', 0.5, 6, valinit=1.7)
    s2 = Slider(s2ax, 'S2', 0.5, 6, valinit=4)

    def update(val):
        norm = SqueezedNorm(vmin=m_min, vmax=m_max, mid=mid.val, s1=s1.val, s2=s2.val)
        im.set_norm(norm)

    mid.on_changed(update)
    s1.on_changed(update)
    s2.on_changed(update)

    plt.show()


def main():
    if args.img_path:
        path = args.img_path
    else:
        path = "fig9_g_heatmap.npy"

    heatmap = np.load(path)
    show_heatmap(heatmap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generating Adversarial Patches')
    parser.add_argument('--img_path', type=str, default='')
    args = parser.parse_args()
    main()
