#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# spatial_reasoning.py
#
# This module provides functions for spatial reasoning related to box movements and positioning.
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

""" spatial_reasoning.py """

import trimesh
import numpy as np


def create_box(box_coords: dict, rot_x=0, rot_y=0, rot_z=0) -> trimesh.base.Trimesh:
    """
    Create a 3D box representation using the provided coordinates.

    Args:
        box_coords (dict): A dictionary with 'min' and 'max' keys specifying the box corners.
        rot_x (float): Rotation around the x-axis (default is 0).
        rot_y (float): Rotation around the y-axis (default is 0).
        rot_z (float): Rotation around the z-axis (default is 0).

    Returns:
        trimesh.base.Trimesh: The 3D box representation.
    """
    minx, miny, minz = box_coords["min"]
    maxx, maxy, maxz = box_coords["max"]
    center = [(minx + maxx) / 2, (miny + maxy) / 2, (minz + maxz) / 2]
    extents = [maxx - minx, maxy - miny, maxz - minz]
    box = trimesh.creation.box(extents=extents, transform=trimesh.transformations.translation_matrix(center))

    for angle, axis in zip([rot_x, rot_y, rot_z], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        rotation = trimesh.transformations.rotation_matrix(np.deg2rad(angle), axis, point=center)
        box.apply_transform(rotation)

    return box


# ... (previous content)


def _filter_boxes_by_condition(ref_box: trimesh.base.Trimesh, boxes_dict: dict, condition_fn) -> dict:
    """
    Filter boxes based on a specific condition.

    Args:
        ref_box (trimesh.base.Trimesh): Reference box mesh.
        boxes_dict (dict): Dictionary containing box labels and their coordinates.
        condition_fn (function): A function that defines the condition for filtering.

    Returns:
        dict: Boxes that satisfy the condition.
    """
    return {label: coords for label, coords in boxes_dict.items() if condition_fn(ref_box, create_box(coords))}


def get_boxes_left(ref_box, boxes_dict: dict) -> dict:
    """
    Get boxes that are positioned to the left of a reference box.

    Args:
        ref_box (dict): The reference box for which boxes on the left are desired.
        boxes_dict (dict): A dictionary of boxes with their coordinates.

    Returns:
        dict: Boxes that are to the left of the reference box.
    """
    ref_box_mesh = create_box(ref_box)

    def condition_fn(ref, box):
        return (
            box.bounds[1][0] < ref.bounds[0][0]
            and box.bounds[0][1] < ref.bounds[1][1]
            and box.bounds[1][1] > ref.bounds[0][1]
            and box.bounds[0][2] < ref.bounds[1][2]
            and box.bounds[1][2] > ref.bounds[0][2]
        )

    return _filter_boxes_by_condition(ref_box_mesh, boxes_dict, condition_fn)


def get_boxes_right(ref_box, boxes_dict: dict) -> dict:
    """
    Get boxes that are positioned to the right of a reference box.

    Args:
        ref_box (dict): The reference box for which boxes on the right are desired.
        boxes_dict (dict): A dictionary of boxes with their coordinates.

    Returns:
        dict: Boxes that are to the right of the reference box.
    """
    ref_box_mesh = create_box(ref_box)

    def condition_fn(ref, box):
        return (
            box.bounds[0][0] > ref.bounds[1][0]
            and box.bounds[0][1] < ref.bounds[1][1]
            and box.bounds[1][1] > ref.bounds[0][1]
            and box.bounds[0][2] < ref.bounds[1][2]
            and box.bounds[1][2] > ref.bounds[0][2]
        )

    return _filter_boxes_by_condition(ref_box_mesh, boxes_dict, condition_fn)


def get_boxes_below(ref_box, boxes_dict: dict) -> dict:
    """
    Get boxes that are positioned below a reference box.

    Args:
        ref_box (dict): The reference box for which boxes below are desired.
        boxes_dict (dict): A dictionary of boxes with their coordinates.

    Returns:
        dict: Boxes that are below the reference box.
    """
    ref_box_mesh = create_box(ref_box)

    def condition_fn(ref, box):
        return (
            box.bounds[1][2] < ref.bounds[0][2]
            and box.bounds[0][0] < ref.bounds[1][0]
            and box.bounds[1][0] > ref.bounds[0][0]
            and box.bounds[0][1] < ref.bounds[1][1]
            and box.bounds[1][1] > ref.bounds[0][1]
        )

    return _filter_boxes_by_condition(ref_box_mesh, boxes_dict, condition_fn)


def get_boxes_above(ref_box, boxes_dict: dict, var=0.1) -> dict:
    """
    Get boxes that are positioned above a reference box.

    Args:
        ref_box (dict): The reference box for which boxes above are desired.
        boxes_dict (dict): A dictionary of boxes with their coordinates.
        var (float): A variance value for considering the height. Default is 0.1.

    Returns:
        dict: Boxes that are above the reference box.
    """
    ref_box_mesh = create_box(ref_box)

    def condition_fn(ref, box):
        return (
            box.bounds[0][2] + var >= ref.bounds[1][2]
            and box.bounds[0][0] < ref.bounds[1][0]
            and box.bounds[1][0] > ref.bounds[0][0]
            and box.bounds[0][1] < ref.bounds[1][1]
            and box.bounds[1][1] > ref.bounds[0][1]
        )

    return _filter_boxes_by_condition(ref_box_mesh, boxes_dict, condition_fn)
