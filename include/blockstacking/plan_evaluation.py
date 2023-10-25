#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Plan Evaluation Tool
# ====================
#
# This module provides functionalities to evaluate various planning methods using
# the BoxSimulator. It can calculate success rates and print the results.
#
# This tool can be executed directly, allowing for arguments to determine
# the simulator mode, execution sleep time, and the break_at feature.
#
# Usage:
#     python plan_evaluation_tool.py [-v] [-s SLEEP] [-b BREAK_AT]
#
# Options:
#     -v          : Run the simulator in non-headless (visual) mode.
#     -s SLEEP    : Sleep duration in seconds between plan executions.
#     -b BREAK_AT : The domain at which to stop processing.
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

""" plan_evaluation.py """

import argparse
import copy
import os
import json

from box_simulator import BoxSimulator
from plan_execution import execute_plan

DEFAULT_METHODS = ["llm", "llm_ic", "llm_ic_pddl", "llm_step", "ChunksGPT4"]
DEFAULT_RUNS = ["run1", "run2", "run3"]


class PlanEvaluation:
    """Class responsible for evaluating and comparing plans."""

    def __init__(self, headless=True, sleep_duration=0, break_at=None):
        """
        Initialize the PlanEvaluation instance.

        :param headless: Boolean indicating if the simulation should run in headless mode.
        :type headless: bool
        :param sleep_duration: Duration in seconds to sleep between plan executions.
        :type sleep_duration: int
        :param break_at: The domain at which to stop processing.
        :type break_at: str or None
        """
        self.experiments_data = self._load_json_data(
            os.path.join("..", "..", "data", "ground_truths", "blockstacking.json")
        )
        self.plans_data = self._load_json_data(
            os.path.join(
                "..",
                "..",
                "data",
                "planning_results",
                "blockstacking",
                "plans-blockstacking.json",
            )
        )
        self.headless = headless
        self.break_at = break_at
        self.simulator = BoxSimulator(headless=headless, sleep=sleep_duration, fig_size=(7, 8))
        self.results = {}

    def get_results(self):
        """Get evaluation results"""
        return self.evaluate()

    def display_results(self, results_to_display):
        success_rates = self._calculate_success_rates(results_to_display)
        self._print_results(success_rates)

    @staticmethod
    def _load_json_data(filepath: str) -> dict:
        """Load JSON data from the given filepath."""
        with open(filepath, "r") as file:
            return json.load(file)

    @staticmethod
    def _calculate_success_rates(results: dict) -> dict:
        """Calculate success rates from the given results."""
        success_rates = {}
        for method, runs in results.items():
            successful_plans = sum(
                1 for domains in runs.values() for result in domains.values() if len(result["errors"]) == 0
            )
            total_plans = sum(len(domains) for domains in runs.values())
            success_rates[method] = successful_plans / total_plans if total_plans else 0.0
        return success_rates

    @staticmethod
    def _print_results(success_rates):
        """Print formatted success rates."""
        print(30 * "=", " RESULTS ", 30 * "=")
        for method, success_rate in success_rates.items():
            if method == "llm":
                print(f"LLM-As-P ({method}):\t\t", success_rate)
            elif method == "llm_ic_pddl":
                print(f"LLM+P ({method}):\t", success_rate)
            elif method == "ChunksGPT4":
                print(f"Ours ({method}):\t", success_rate)

    def evaluate(self, filter_methods=None, filter_runs=None):
        """Evaluate and execute plans using the simulator."""
        filter_methods = filter_methods or DEFAULT_METHODS
        filter_runs = filter_runs or DEFAULT_RUNS

        for method, runs in self.plans_data.items():
            if method in filter_methods:
                self.results[method] = {}
                for run, domains in runs.items():
                    if run in filter_runs:
                        self.results[method][run] = {}
                        for domain, plan_data in domains.items():
                            if self.break_at and domain == self.break_at:
                                break
                            plan, original, scene, goal = (
                                plan_data["revised"],
                                plan_data["original"],
                                self.experiments_data[domain]["scene3D"],
                                self.experiments_data[domain]["goal"],
                            )

                            print(20 * "=", f"{method} - {domain} - {run}", 20 * "=")
                            domain_text = self.experiments_data[domain]["domain"].replace("\n", "\n\t")
                            goal_text = self.experiments_data[domain]["goal"].replace("\n", "\n\t")
                            print(f"domain:\n\t{domain_text}")
                            print(f"goal:\n\t{goal_text}")
                            self.simulator.set_text(f"{method} - {domain} - {run} \n {goal}")
                            scene_mem = copy.deepcopy(scene)
                            result = execute_plan(plan, scene_mem, goal, self.simulator, original)
                            self.results[method][run][domain] = copy.deepcopy(result)
                            if result["errors"]:
                                for error in result["errors"]:
                                    print("\033[91m" + error + "\033[0m")
        if not self.headless:
            self.simulator.keep_display_open()
        return self.results


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Execute plans with optional settings.")
    PARSER.add_argument(
        "-v",
        dest="headless",
        action="store_false",
        default=True,
        help="Run the simulation for blockstacking.",
    )
    PARSER.add_argument(
        "-s",
        "--sleep",
        type=int,
        default=0,
        help="Sleep duration in seconds between plan executions.",
    )
    PARSER.add_argument(
        "-b",
        "--break_at",
        type=str,
        default="p06",
        help="The domain at which to stop processing. "
        "Use '-b None' if you like to run all "
        "experiments. Default is '-b p06'.",
    )

    ARGS = PARSER.parse_args()
    EVALUATION = PlanEvaluation(headless=ARGS.headless, sleep_duration=ARGS.sleep, break_at=ARGS.break_at)
    EVALUATION_RESULTS = EVALUATION.get_results()
    EVALUATION.display_results(EVALUATION_RESULTS)
