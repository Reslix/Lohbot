import matplotlib.pyplot  as plt


def imshow(img, im=None):
    plt.axis("off")
    if im is None:
        plt.ion()
        im = plt.imshow(img)
    else:
        im.set_data(img)
    plt.pause(.0003)
    return im


def close_plt():
    plt.ioff()


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()