import argparse
import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.collections import PathCollection

from modules.triangulation.load_image import image_to_1080p


class CorrespondenceSelector:
    def __init__(self, image1, image2):
        self.image1 = image1
        self.image2 = image2
        self.selected_index = 0
        self.selections = np.zeros([2, 8, 2])
        self.complete = False

    def keypress_callback(self, event: KeyEvent):
        if event.key is None:
            return

        print(event.key)

        if event.key in ["enter"]:
            self.complete = True
            return

        if event.key in [*"12345678"]:
            self.selected_index = int(event.key) - 1
            print("Selected point", event.key)
            return

    def click_callback(self, event: MouseEvent, left_axes, right_axes):
        if event.inaxes is None:
            return

        position = (event.xdata, event.ydata)
        side = (
            0
            if event.inaxes == left_axes
            else 1 if event.inaxes == right_axes else None
        )
        if side is None:
            return

        self.selections[side][self.selected_index] = position

    def select(self):
        fig, (left_axes, right_axes) = plt.subplots(1, 2)
        fig.set_figheight(4)
        fig.set_figwidth(12)

        left_axes: Axes
        right_axes: Axes
        left_axes.imshow(self.image1)
        right_axes.imshow(self.image2)
        left_axes.set_axis_off()
        right_axes.set_axis_off()
        fig.tight_layout()

        cid1 = fig.canvas.mpl_connect(
            "button_release_event",
            partial(self.click_callback, left_axes=left_axes, right_axes=right_axes),  # type: ignore
        )
        cid2 = fig.canvas.mpl_connect(
            "key_release_event",
            self.keypress_callback,  # type: ignore
        )

        # Rendering loop.
        while not self.complete:
            # Clear the list of artists.
            for artist in [*left_axes.get_children(), *right_axes.get_children()]:
                if isinstance(artist, PathCollection):
                    artist.remove()

            # Draw the points.
            for i, ax in enumerate([left_axes, right_axes]):
                ax.scatter(
                    [
                        *self.selections[i, : self.selected_index, 0],
                        *self.selections[i, self.selected_index + 1 :, 0],
                    ],
                    [
                        *self.selections[i, : self.selected_index, 1],
                        *self.selections[i, self.selected_index + 1 :, 1],
                    ],
                    c="blue",
                )
                ax.scatter(
                    self.selections[i, [self.selected_index], 0],
                    self.selections[i, [self.selected_index], 1],
                    c="red",
                )

            plt.pause(0.01)

        fig.canvas.mpl_disconnect(cid1)
        fig.canvas.mpl_disconnect(cid2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--left", required=True, help="Filename for left-side image."
    )
    parser.add_argument(
        "-r", "--right", required=True, help="Filename for right-side image."
    )
    args = parser.parse_args()

    base = os.path.abspath(os.path.dirname(__file__))

    left = image_to_1080p(np.array(PIL.Image.open(base + "/" + args.left)))
    right = image_to_1080p(np.array(PIL.Image.open(base + "/" + args.right)))

    selector = CorrespondenceSelector(left, right)
    selector.select()

    data = []
    for side in [0, 1]:
        for index, selection in enumerate(selector.selections[side]):
            data.append((side, index, selection[0], selection[1]))

    np.savetxt("selections.txt", np.array(data), delimiter=",", header="side,index,x,y")


if __name__ == "__main__":
    main()
