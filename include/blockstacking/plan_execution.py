#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# plan_execution.py
#
# This module contains functions for executing and evaluating plans related to box movements.
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

""" plan_executions.py """

import re
import time
import spatial_reasoning as sr


def _check_conditions(source: str, destination: str, simulator) -> str:
    """
    Check conditions before moving a box.

    Args:
        source (str): Label of the source box.
        destination (str): Label of the destination box or table.
        simulator (BoxSimulator): Instance of the BoxSimulator class.

    Returns:
        str: Error message if conditions aren't met, else None.
    """
    boxes = simulator.get_boxes()
    boxes_above_dest = sr.get_boxes_above(boxes[destination], boxes)
    box_can_be_moved = False
    if len(boxes_above_dest) == 0:
        box_can_be_moved = True
    elif len(boxes_above_dest) == 1:
        box_above_nb = re.findall(r"\d+", list(boxes_above_dest.keys())[0])
        src_box_nb = re.findall(r"\d+", source)
        if box_above_nb[0] == src_box_nb[0]:
            box_can_be_moved = True
    if box_can_be_moved:
        return None
    error_msg = (
        f"Cannot move '{source}' to {destination}: "
        f"'{', '.join(boxes_above_dest.keys())}' already positioned on top of it! "
        f"Plan aborted, please re-plan considering the current state."
    )
    return error_msg


def extract_numbers(s, keywords):
    """
    Extract numerical values based on specific keywords from a string.

    Args:
        s (str): The input string.
        keywords (list): List of keywords for which numbers should be extracted.

    Returns:
        int or None: The extracted number or None if no number is found.
    """
    numbers = []
    for keyword in keywords:
        pattern = r"{}\d{{1,2}}".format(keyword)
        compiled_pattern = re.compile(pattern)
        matches = compiled_pattern.findall(s)
        numbers += [int(m.replace(keyword, "")) for m in matches]

    return None if len(numbers) != 1 else numbers[0]


def move_box_and_above(source: str, destination: str, simulator):
    """
    Move the source box and any boxes that are on top of it to the destination.

    Args:
        source (str): Label of the source box.
        destination (str): Label of the destination box or table.
        simulator (BoxSimulator): Instance of the BoxSimulator class.

    Returns:
        str or None: An error message if an error occurs, else None.
    """
    box_src_nb = extract_numbers(source, ["b"])
    boxes = simulator.get_boxes()
    boxes_on_src = sr.get_boxes_above(boxes[source], boxes)
    boxes_on_dest = sr.get_boxes_above(boxes[destination], boxes)

    if destination in boxes_on_src:
        error = f"Evaluator: Cannot move {source} to {destination}: {destination} is already on {source}."
        return error
    if len(boxes_on_dest) > 0 and (list(boxes_on_dest.keys())[0] != source) and (destination != f"table{box_src_nb}"):
        error = (
            f"Evaluator: Cannot move '{source}' to {destination}: '{', '.join(boxes_on_dest.keys())}' "
            f"already positioned on top of it!"
        )
        return error
    if len(boxes_on_src) > 0:
        error = f"Evaluator: '{source}' is not clear: '{', '.join(boxes_on_src.keys())}' is positioned on top of it!"
        return error

    render = True if len(boxes_on_src) == 0 else False
    simulator.move_box_on_top(source, destination, render)

    if render:
        return None

    prev_box = source
    last_box = list(boxes_on_src.keys())[-1]
    for box_to_move in boxes_on_src:
        render = True if box_to_move == last_box else False
        simulator.move_box_on_top(box_to_move, prev_box, render=render)
        prev_box = box_to_move
    return None


def _execute_step(step: str, simulator) -> str:
    """
    Execute a specific step in the plan.

    Args:
        step (str): The plan step.
        simulator (BoxSimulator): Instance of the BoxSimulator class.

    Returns:
        str or None: An error message if an error occurs, else None.
    """
    match_on = re.match(r"move (b\d+) on (b\d+|table)", step)
    if match_on:
        source, destination = match_on.groups()
        if destination == "table":
            block_number = re.findall(r"\d+", source)[0]
            destination = f"table{block_number}"
        error = move_box_and_above(source, destination, simulator)
        return error
    return None


def _evaluate_goal(goal: str, simulator):
    """
    Evaluate the current state of the scene against a desired goal.

    Args:
        goal (str): The desired state.
        simulator (BoxSimulator): Instance of the BoxSimulator class.

    Returns:
        list: A list of error messages if the current state doesn't match the goal.
    """
    boxes = simulator.get_boxes()
    goal_conditions = goal.split("\n")
    errors = []
    for goal_condition in goal_conditions:
        match_on_top_of = re.match(r"(b\d+) should be on top of (b\d+)", goal_condition)
        if match_on_top_of:
            box_top, box_below = match_on_top_of.groups()
            boxes_on_top = list(sr.get_boxes_above(boxes[box_below], boxes).keys())
            if box_top not in boxes_on_top:
                errors.append(f"Evaluator: {box_top} is not on top of {box_below}")
    return errors


def execute_plan(plan: list, scene: str, goal: str, simulator, original: str) -> dict:
    """
    Execute a plan and evaluate it against a goal.

    Args:
        plan (list): List of steps in the plan.
        scene (str): Initial scene configuration.
        goal (str): Desired state at the end of the plan.
        simulator (BoxSimulator): Instance of the BoxSimulator class.
        original (str): Original configuration of the scene.

    Returns:
        dict: Results of plan execution including errors and steps taken.
    """
    simulator.load_scene(scene)
    step_counter = 0
    errors = []
    time.sleep(simulator.sleep)
    for step_ix in range(len(plan)):
        step = plan[step_ix]
        original_steps = original[step_ix]
        simulator.set_title("LLM: " + step + "\n" + "\n".join(original_steps))
        error = _execute_step(step, simulator)
        if error:
            errors.append(error)
            simulator.set_title(error)
            time.sleep(simulator.sleep)
            break
        step_counter += 1
        time.sleep(simulator.sleep)
    error = _evaluate_goal(goal, simulator)
    errors += error
    if errors:
        simulator.set_title(", ".join(error))
    else:
        simulator.set_title("Evaluator: goal successfully achieved")
    time.sleep(simulator.sleep)
    return {"errors": errors, "steps": step_counter}
