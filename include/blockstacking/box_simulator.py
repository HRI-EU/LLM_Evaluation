#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This module contains the BoxSimulator class for visualizing boxes in 2D space.
#
# Copyright (C) 2023, Honda Research Institute Europe GmbH.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     (1) Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#
#     (2) Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in
#     the documentation and/or other materials provided with the
#     distribution.
#
#     (3)The name of the author may not be used to
#     endorse or promote products derived from this software without
#     specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Authors: Joerg Deigmoeller <joerg.deigmoeller@honda-ri.de>

""" box_simulator.py """

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class BoxSimulator:
    """
    Provides functionality to simulate and visualize boxes in a 2D environment.
    """

    def __init__(self, headless: bool, sleep: int, fig_size=(7, 10)):
        """
        Initialize the simulator.


        Args:
            headless (bool): Run the simulation in headless mode.
            sleep (int): Sleep duration in seconds between redraws.
            fig_size (tuple): size of the figure plot.
        """
        self.headless = headless
        self.sleep = sleep
        self.colors = ["r", "g", "b", "y", "m", "c"]
        self.ax_text = None
        self.boxes = None

        if not self.headless:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=fig_size)
        self.title = ""
        self.text = ""

    def load_scene(self, boxes_dict: dict):
        """
        Load a scene configuration of boxes.

        Args:
            boxes_dict (dict): Dictionary containing box coordinates.
        """
        self.boxes = boxes_dict
        if not self.headless:
            self._draw_boxes()

    def update_box(self, label: str, new_coords: dict, render: bool):
        """
        Update a box's coordinates.

        Args:
            label (str): Box label.
            new_coords (dict): New coordinates for the box.
            render (bool): Whether to re-render the scene.
        """
        self.boxes[label] = new_coords
        if not self.headless and render:
            self._draw_boxes()

    def move_box_on_top(self, source_label: str, target_label: str, render: bool = True):
        """
        Move a box to the top of another box or location in the scene.

        Args:
            source_label (str): Label of the source box.
            target_label (str): Label of the destination location or box.
            render (bool): Indicates if the scene should be rendered after moving. Default is True.

        Returns:
            None
        """
        source_box = self.boxes[source_label]
        target_box = self.boxes[target_label]
        new_z = target_box["max"][2]
        height_source = source_box["max"][2] - source_box["min"][2]
        width_source = source_box["max"][0] - source_box["min"][0]
        depth_source = source_box["max"][1] - source_box["min"][1]

        # Position the source box on top of the target box
        source_box["min"] = [target_box["min"][0], target_box["min"][1], new_z]
        source_box["max"] = [
            target_box["min"][0] + width_source,
            target_box["min"][1] + depth_source,
            new_z + height_source,
        ]

        self.update_box(source_label, source_box, render)

    def _draw_boxes(self):
        """Draw boxes on the 2D canvas."""
        self.ax.clear()
        for idx, (label, box) in enumerate(self.boxes.items()):
            minx, _, minz = box["min"]
            maxx, _, maxz = box["max"]
            edgecolor = "none" if "table" in label else self.colors[idx % len(self.colors)]
            rect = patches.Rectangle(
                (minx, minz),
                maxx - minx,
                maxz - minz,
                linewidth=1,
                edgecolor=edgecolor,
                facecolor="none",
            )
            self.ax.add_patch(rect)
            centerx, centerz = (minx + maxx) / 2, (minz + maxz) / 2
            self.ax.text(
                centerx,
                centerz,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=edgecolor,
            )
        self._configure_axes()

    def _configure_axes(self):
        """Configure plot axes."""
        self.ax.set_title(self.title, fontsize=10)
        if self.ax_text is not None:
            self.ax_text.remove()
        self.ax_text = self.ax.text(-50, 190, self.text, va="top")
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(0, 200)
        self.ax.set_aspect("equal", "box")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Z")
        self.fig.canvas.flush_events()

    def set_title(self, title=""):
        """
        Set the title for the visualization.

        Args:
            title (str): The desired title for the visualization.
        """
        self.title = title
        if not self.headless:
            self._configure_axes()

    def set_text(self, text=""):
        """
        Set a descriptive text for the visualization.

        Args:
            text (str): The desired text to be displayed.
        """
        self.text = text
        if not self.headless:
            self._configure_axes()

    def close(self):
        """
        Close the visualization.
        """
        plt.close(self.fig)

    def get_boxes(self):
        """
        Get the current boxes in the scene.

        Returns:
            dict: The current configuration of boxes in the scene.
        """
        return self.boxes

    @staticmethod
    def close_event(event):
        """
        Close the visualization based on a trigger event.

        Args:
            event: The event triggering the closure.
        """
        plt.close(event.canvas.figure)

    def keep_display_open(self):
        """
        Keep the visualization display open until a key is pressed.
        """
        self.fig.canvas.mpl_connect("key_press_event", self.close_event)
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    SIMULATOR = BoxSimulator(headless=False, sleep=2)
    SAMPLE_BOXES = {
        "box1": {"min": [0, 0, 0], "max": [10, 10, 10]},
        "box2": {"min": [15, 0, 0], "max": [25, 10, 10]},
    }
    SIMULATOR.load_scene(SAMPLE_BOXES)
    plt.show()
