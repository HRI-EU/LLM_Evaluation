#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Evaluation of the final plans
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
# Authors: Felix Ocker <felix.ocker@honda-ri.de>
#
import copy
import json
import logging
import os
import statistics

import numpy as np
import pandas as pd
import scipy.stats as scistats

import plot_utilities

from collections import Counter
from dataclasses import dataclass
from typing import (
    Dict,
    List,
    Tuple,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Result:
    replans: int
    subplans: int
    executable: bool
    type: str
    experiment: str
    run: int
    final_state: dict = None
    correct: bool = False
    edit_distance: float = 0.0
    exec_time: float = None


@dataclass
class ExperimentGroundTruth:
    id: str
    solids: list
    liquids: list
    optional: list


class Evaluator:
    def __init__(
        self,
        missing_distance_weight: float = 1.0,
        superfluous_distance_weight: float = 0.2,
        time_base: float = 60,
        ml_replan_time_s: float = 0.1,
        hl_replan_time_s: float = 60,
    ) -> None:
        self.ordered_types: List[str] = [
            "nRnS",
            "nRwS",
            "F0",
            "F1",
            "F2",
            "SF0",
            "SF1",
            "SF2",
        ]
        self.mdw = missing_distance_weight
        self.sdw = superfluous_distance_weight
        self.color_palette = "viridis"
        self.time_base = time_base
        self.ml_replan_time_s = ml_replan_time_s
        self.hl_replan_time_s = hl_replan_time_s

    def load_several(self, data_dir: str, blacklist: List[str] = None) -> list:
        """Load data from all json files in the data directory"""
        json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
        if blacklist:
            json_files = [f for f in json_files if f not in blacklist]
        return [self.load(data_dir, jf) for jf in json_files]

    @staticmethod
    def load(data_dir: str, filename: str) -> dict:
        logger.info(f"Loading {filename}.")
        with open(data_dir + "/" + filename) as f:
            data = json.load(f)
        return data

    def load_ground_truth(self, gt_dir: str, gt_file: str) -> dict:
        gt_data = self.load(gt_dir, gt_file)
        gts = {
            gt_id: ExperimentGroundTruth(
                id=gt_id,
                solids=gt_data[gt_id]["ingredient"],
                liquids=gt_data[gt_id]["liquid"],
                optional=gt_data[gt_id]["optional"] if "optional" in gt_data[gt_id] else [],
            )
            for gt_id in gt_data
        }
        logger.info(f"Loaded {len(gts)} ground truth elements.")
        return gts

    @staticmethod
    def retrieve_relevant_data(data: dict) -> List[Result]:
        results = []
        feedbacktype_ = data["feedbackType"]
        for key, value in data.items():
            if key == "feedbackType":
                continue
            for run_, run_data in enumerate(value):
                res = Result(
                    replans=0,
                    subplans=0,
                    executable=False,
                    type=feedbacktype_,
                    experiment=key,
                    run=run_,
                )
                for elem in run_data["log"]:
                    if elem["type"] == "rePlan":
                        res.replans += 1
                    elif elem["type"] == "subPlan":
                        res.subplans += 1
                    elif elem["type"] == "evaluation":
                        if elem["result"] == "Success":
                            res.executable = True
                if run_data["finalState"] is None:
                    res.final_state = {"empty_tray": {"type": "tray", "holdsObject": []}}
                else:
                    res.final_state = json.loads(run_data["finalState"])
                logger.debug(f"Retrieved entry: {res}")
                results.append(res)
        return results

    @staticmethod
    def _generate_baseline(results: List[Result]) -> List[Result]:
        """Generate nRnS and nRS from the available data"""
        baseline = []
        for r in results:
            cp_nrns = copy.deepcopy(r)
            cp_nrns.type = "nRnS"
            if cp_nrns.replans > 0 or cp_nrns.subplans > 0:
                cp_nrns.executable = False
                cp_nrns.replans = 0
                cp_nrns.subplans = 0
                cp_nrns.final_state = {"empty_tray": {"type": "tray", "holdsObject": []}}

            cp_nrws = copy.deepcopy(r)
            cp_nrws.type = "nRwS"
            if cp_nrws.replans > 0:
                cp_nrws.executable = False
                cp_nrws.replans = 0
                cp_nrws.final_state = {"empty_tray": {"type": "tray", "holdsObject": []}}

            baseline.extend([cp_nrns, cp_nrws])
        return baseline

    def calc_statistics(
        self,
        data: List[Result],
        data_dir: str,
        plot_per_experiment: bool = False,
    ) -> None:
        # sort by feedback type
        sorted_by_type = {}
        for d in data:
            if d.type not in sorted_by_type:
                sorted_by_type[d.type] = [d]
            else:
                sorted_by_type[d.type].append(d)

        # remove experiment types for which there is no data
        self.ordered_types = [t for t in self.ordered_types if t in sorted_by_type]

        # calculate scores by setup
        for t in self.ordered_types:
            print(f"Setup: {t}")
            executable_rate = np.mean([d.executable for d in sorted_by_type[t]])
            print(f"Executable: {executable_rate}")
            correct_rate = np.mean([d.correct for d in sorted_by_type[t]])
            failed_invalid = len([d for d in sorted_by_type[t] if d.edit_distance == -1])
            print(
                f"Correct rate: {correct_rate} "
                f"({failed_invalid} failed due to invalid or incomplete responses, e.g., 0 or 2 glasses)"
            )
            # avg edit distance is messed up by values for invalid experiments
            # edit_distance = np.mean([d.edit_distance for d in sorted_by_type[t] if d.edit_distance != float("inf")])
            # print(f"Average edit distance (for valid results): {edit_distance}")

        # generate accumulated boxplots (merges all experiments in folder)
        all_replans, all_subplans = [], []
        all_replan_labels, all_subplan_labels = [], []
        for t in self.ordered_types:
            all_replan_labels.append(f"HLP {t}")
            all_replans.append([d.replans for d in sorted_by_type[t]])
            all_subplan_labels.append(f"MLP {t}")
            all_subplans.append([d.subplans for d in sorted_by_type[t]])
        plot_utilities.boxplot(
            data=all_replans + all_subplans,
            labels=all_replan_labels + all_subplan_labels,
            title=f"All experiments for {data_dir.split('/')[-1]}",
            filename="20230827-eval-boxplot",
            ylabel="Number of replans",
            vlines=[8.5],
        )

        # generate edit distance distribution
        edit_distances = [[d.type, d.edit_distance, d.correct, d.executable] for d in data]
        df = pd.DataFrame(edit_distances, columns=["type", "edit_distance", "correct", "executable"])
        show_distribution = False
        if show_distribution:
            plot_utilities.distribution(
                data=df,
                x="edit_distance",
                hue="type",
                hue_order=self.ordered_types,
                title=f"Edit distances for {data_dir.split('/')[-1]}",
                color_palette=self.color_palette,
            )
        plot_utilities.distribution_bars(
            data=df,
            x="edit_distance",
            stat="percent",
            hue="type",
            hue_order=self.ordered_types,
            binwidth=0.15,
            title=f"Edit distances for {data_dir.split('/')[-1]}",
            x_lim=(-0.5, 4),
            filename="20230827-eval-distplot",
            color_palette=self.color_palette,
        )
        plot_utilities.barchart(
            data=df,
            x="type",
            y="executable",
            x_order=self.ordered_types,
            title=f"Executability by type for {data_dir.split('/')[-1]}",
            filename="20230827-eval-executability",
            ylabel="Executability [%]",
            color_palette=self.color_palette,
            percent=True,
        )
        plot_utilities.barchart(
            data=df,
            x="type",
            y="correct",
            x_order=self.ordered_types,
            title=f"Correctness by type for {data_dir.split('/')[-1]}",
            filename="20230827-eval-correctness",
            ylabel="Correctness [%]",
            color_palette=self.color_palette,
            percent=True,
        )

        # correct rates for completed experiments only
        sum_correct_after = sum([d.correct for d in data if d.executable])
        sum_executable_after = len([d.executable for d in data if d.executable])
        avg_correct_after_executable = sum_correct_after / sum_executable_after * 100

        edit_distances_reduced = [[d.type, d.edit_distance, d.correct, d.executable] for d in data if d.executable]
        df_reduced = pd.DataFrame(
            edit_distances_reduced,
            columns=["type", "edit_distance", "correct", "executable"],
        )
        plot_utilities.barchart(
            data=df_reduced,
            x="type",
            y="correct",
            x_order=self.ordered_types,
            title=f"Correctness by type for {data_dir.split('/')[-1]} (executable only)",
            filename="20230827-eval-correctness-dependent",
            ylabel="Correctness [%]",
            axhline=avg_correct_after_executable,
            color_palette=self.color_palette,
            percent=True,
        )

        # timing results
        for d in data:
            d.exec_time = self.time_base + self.hl_replan_time_s * d.replans + self.ml_replan_time_s * d.subplans
        exec_times_reduced = [[d.type, d.exec_time] for d in data if d.executable]
        df_reduced = pd.DataFrame(exec_times_reduced, columns=["type", "exec_time"])
        plot_utilities.barchart(
            data=df_reduced,
            x="type",
            y="exec_time",
            x_order=self.ordered_types,
            title=f"Runtimes by type for {data_dir.split('/')[-1]} (executable only)",
            filename="20230827-eval-execution-times",
            ylabel="Runtime [s]",
            color_palette=self.color_palette,
        )

        # combined plot for executability and timing
        full_type_names = {
            "BL": "BL: Baseline",
            "M": "M: Mid-level planner",
            "H0": "H0: High-level planner, what",
            "H1": "H1: High-level planner, what + why",
            "H2": "H2: High-level planner, what + why + how",
            "MH0": "MH0: Mid- and high-level planner, what",
            "MH1": "MH1: Mid- and high-level planner, what + why",
            "MH2": "MH2: Mid- and high-level planner, what + why + how",
        }
        point_data = []
        for t in self.ordered_types:
            time_avg = statistics.mean([d.exec_time for d in data if d.executable and d.type == t])
            time_sem = scistats.sem([d.exec_time for d in data if d.executable and d.type == t])
            exec_rate = statistics.mean([d.executable for d in data if d.type == t]) * 100
            exec_sem = scistats.sem([d.executable for d in data if d.type == t]) * 100
            point_data.append([full_type_names[t], time_avg, time_sem, exec_rate, exec_sem])
        df_points = pd.DataFrame(
            point_data,
            columns=["Type", "time_avg", "time_sem", "exec_rate", "exec_sem"],
        )
        plot_utilities.scatter(
            data=df_points,
            x="time_avg",
            xlabel="Average runtime [s]",
            y="exec_rate",
            ylabel="Executability [%]",
            xlim=(55.0, 175.0),
            hue="Type",
            hue_order=[full_type_names[ot] for ot in self.ordered_types],
            title=f"Executability (higher is better) over runtime (lower is better).",
            filename="20230827-eval-execution-over-times",
            color_palette=self.color_palette,
            size=(4.5, 4.5),
        )

        # sort by experiment
        sorted_by_experiment = {}
        for d in data:
            if d.experiment not in sorted_by_experiment:
                sorted_by_experiment[d.experiment] = {d.type: [d]}
            else:
                if d.type not in sorted_by_experiment[d.experiment]:
                    sorted_by_experiment[d.experiment][d.type] = [d]
                else:
                    sorted_by_experiment[d.experiment][d.type].append(d)

        # calculate distributions by experiment
        if plot_per_experiment:
            for exp, exp_data in sorted_by_experiment.items():
                replans, subplans = [], []
                replan_labels, subplan_labels = [], []
                for t in exp_data:
                    replans.append([r.replans for r in exp_data[t]])
                    replan_labels.append(f"replan {t}")
                    replans.append([r.subplans for r in exp_data[t]])
                    subplan_labels.append(f"subplan {t}")
                plot_utilities.boxplot(
                    data=replans + subplans,
                    labels=replan_labels + subplan_labels,
                    title=f"Experiment {exp}",
                )

    def rename_types(
        self,
        data: List[Result],
        name_lookup: dict = None,
    ) -> List[Result]:
        default_lookup = {
            "nRnS": "BL",
            "nRwS": "M",
            "F0": "H0",
            "F1": "H1",
            "F2": "H2",
            "SF0": "MH0",
            "SF1": "MH1",
            "SF2": "MH2",
        }
        lookup = name_lookup if name_lookup else default_lookup
        self.ordered_types = list(lookup.values())
        for r in data:
            r.type = lookup[r.type]
        return data

    def check_correctness(
        self,
        final_state: dict,
        solids: list,
        liquids: list,
        optionals: list,
        exp_id: str,
    ) -> Tuple[bool, float]:
        """
        NOTE: assumes that there is either exactly one pizza or exactly one glass with a cocktail
        NOTE: distinguish pizza and cocktail scenario implicitly via occurrence of pizza_dough and glass types
        """
        # retrieve objects held by tray pizzas and glasses
        trays = [t for t in [o for o in final_state if "type" in final_state[o]] if final_state[t]["type"] == "tray"]
        assert len(trays) == 1, f"Issue with {exp_id}: number of trays included is not 1."
        things = final_state[trays[0]]["holdsObject"]
        if len(things) != 1:
            logger.info(f"{exp_id}: objects on tray ambiguous ({things}).")
            return False, -1
        thing = final_state[things[0]]
        res_cntr_sld = Counter(thing["holdsObject"])
        res_cntr_lqd = Counter(thing["holdsLiquid"])
        gt_cntr_sld = Counter(solids)
        gt_cntr_lqd = Counter(liquids)
        gt_cntr_opt = Counter(optionals)
        # RFE: extend this for more detailed analysis regarding first and second type errors

        # calc edit distance
        # NOTE: assumes that the amount per ingredient does not matter
        res_sld = set(res_cntr_sld.keys())
        res_lqd = set(res_cntr_lqd.keys())
        gt_sld = set(gt_cntr_sld.keys())
        gt_lqd = set(gt_cntr_lqd.keys())
        gt_opt = set(gt_cntr_opt.keys())
        missing = (gt_sld - res_sld) | (gt_lqd - res_lqd)
        superfluous = (res_sld | res_lqd) - (gt_sld | gt_lqd | gt_opt)
        edit_distance = self.mdw * len(missing) + self.sdw * len(superfluous)

        # calc correct
        correct = len(missing) == 0 and len(superfluous) == 0
        # correct = res_cntr_sld == gt_cntr_sld and res_cntr_lqd == gt_cntr_lqd
        logger.debug(
            f"Correct {exp_id}: {correct}.\n"
            f"Edit distance: {edit_distance}.\n"
            f"Missing: {missing}.\n"
            f"Superfluous: {superfluous}.\n"
            f"Result: {res_cntr_lqd}, {res_cntr_sld}\nGT: {gt_cntr_lqd}, {gt_cntr_sld}, {gt_cntr_opt}\n"
        )
        return correct, edit_distance

    def assess_correct(self, results: List[Result], ground_truths: Dict[str, ExperimentGroundTruth]) -> None:
        for res in results:
            gt = ground_truths[res.experiment]
            res_id = res.type + "-" + res.experiment + "-" + str(res.run)
            res.correct, res.edit_distance = self.check_correctness(
                res.final_state, gt.solids, gt.liquids, gt.optional, res_id
            )

    @staticmethod
    def provide_data_overview(results: List[Result], gt: dict) -> None:
        print(f"Number of experiments: {len(results)}")
        cocktail_names = list({r.experiment for r in results})
        cocktail_names.sort()
        print(f"Cocktails: {cocktail_names}")
        ingredients_ = [gt[e].solids + gt[e].liquids + gt[e].optional for e in gt if e in cocktail_names]
        ingredients = list({i for sublist in ingredients_ for i in sublist})
        ingredients.sort()
        print(f"Ingredients: {ingredients}")
        print(f"Number of elements in GT: {len([e for e in gt if e.startswith('02')])}")

    def evaluate(
        self,
        data_dir: str,
        ground_truth_dir: str,
        ground_truth_file: str,
        blacklist: List[str],
    ) -> None:
        """
        Convenience function for running the evaluation.
        """
        gt = self.load_ground_truth(ground_truth_dir, ground_truth_file)
        data = self.load_several(data_dir, blacklist=blacklist)
        results = [r for d in data for r in self.retrieve_relevant_data(d)]
        self.provide_data_overview(results, gt)
        baseline = self._generate_baseline(results)
        results.extend(baseline)
        results = self.rename_types(results)

        self.assess_correct(results=results, ground_truths=gt)
        self.calc_statistics(results, data_dir)


if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate(
        data_dir="../../data/planning_results/cocktails",
        ground_truth_dir="../../data/ground_truths",
        ground_truth_file="cocktails.json",
        blacklist=["example.json", "example2.json"],
    )
