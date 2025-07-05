#!/usr/bin/env python3
"""
Enhanced Street Fighter II Transformer Analytics System with Oscillation Analysis
Comprehensive analysis of transformer learning patterns, performance metrics,
and actionable recommendations for improvement.

Key enhancements:
1. Oscillation-based positioning analysis
2. Spatial control evaluation
3. Neutral game pattern recognition
4. Whiff bait detection analysis
5. Cross-attention weight analysis
6. Enhanced feature importance for 45-feature system
"""

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import warnings
import re

warnings.filterwarnings("ignore")

# Set up plotting style
plt.style.use("default")
sns.set_palette("husl")


class EnhancedStreetFighterAnalytics:
    """
    Enhanced analytics system for Street Fighter II with oscillation analysis
    """

    def __init__(self, analysis_dir="analysis_data", logs_dir="logs"):
        self.analysis_dir = Path(analysis_dir)
        self.logs_dir = Path(logs_dir)
        self.output_dir = Path("analytics_output")
        self.output_dir.mkdir(exist_ok=True)

        # Updated feature names from wrapper.py (45 total: 21 strategic + 12 oscillation + 12 button)
        self.strategic_feature_names = [
            "player_in_danger",  # 0
            "opponent_in_danger",  # 1
            "health_ratio",  # 2
            "combined_health_change",  # 3
            "player_damage_rate",  # 4
            "opponent_damage_rate",  # 5
            "player_corner_distance",  # 6
            "opponent_corner_distance",  # 7
            "player_near_corner",  # 8
            "opponent_near_corner",  # 9
            "center_control",  # 10
            "vertical_advantage",  # 11
            "space_control_from_oscillation",  # 12 - Enhanced with oscillation
            "optimal_spacing",  # 13
            "forward_pressure_from_oscillation",  # 14 - Enhanced with oscillation
            "defensive_movement",  # 15
            "close_combat_frequency",  # 16
            "enhanced_score_momentum",  # 17 - Key combo feature
            "status_difference",  # 18
            "agent_victories",  # 19
            "enemy_victories",  # 20
        ]

        # NEW: Oscillation feature names (12 features)
        self.oscillation_feature_names = [
            "player_oscillation_frequency",  # 21
            "opponent_oscillation_frequency",  # 22
            "player_oscillation_amplitude",  # 23
            "opponent_oscillation_amplitude",  # 24
            "space_control_score",  # 25
            "neutral_game_duration_ratio",  # 26
            "movement_aggression_ratio",  # 27
            "defensive_movement_ratio",  # 28
            "neutral_dance_ratio",  # 29
            "whiff_bait_frequency",  # 30
            "advantage_transition_frequency",  # 31
            "velocity_differential",  # 32
        ]

        # Previous button feature names (12 features)
        self.button_feature_names = [
            "prev_B_pressed",  # 33 - Light Kick
            "prev_Y_pressed",  # 34 - Light Punch
            "prev_SELECT_pressed",  # 35
            "prev_START_pressed",  # 36
            "prev_UP_pressed",  # 37
            "prev_DOWN_pressed",  # 38
            "prev_LEFT_pressed",  # 39
            "prev_RIGHT_pressed",  # 40
            "prev_A_pressed",  # 41 - Medium Kick
            "prev_X_pressed",  # 42 - Medium Punch
            "prev_L_pressed",  # 43 - Heavy Punch
            "prev_R_pressed",  # 44 - Heavy Kick
        ]

        # Combined feature names (45 total)
        self.all_feature_names = (
            self.strategic_feature_names
            + self.oscillation_feature_names
            + self.button_feature_names
        )

        # Enhanced action categories
        self.action_categories = {
            "movement": list(range(1, 9)),
            "light_attacks": list(range(9, 15)),
            "medium_attacks": list(range(15, 21)),
            "heavy_attacks": list(range(21, 27)),
            "jumping_attacks": list(range(27, 33)),
            "crouching_attacks": list(range(33, 39)),
            "special_moves": list(range(39, 51)),
            "defensive": [54, 55, 56],  # Updated from wrapper
            "neutral_game": list(range(1, 9)) + [54, 55, 56],  # Movement + defensive
            "combo_starters": list(range(15, 21))
            + list(range(21, 27)),  # Medium + heavy
        }

        # NEW: Oscillation analysis categories
        self.oscillation_categories = {
            "frequency_features": [21, 22],  # Player/opponent oscillation frequency
            "amplitude_features": [23, 24],  # Player/opponent oscillation amplitude
            "spatial_control": [
                25,
                12,
            ],  # Space control score + strategic space control
            "neutral_game": [
                26,
                29,
                31,
            ],  # Neutral duration, dance, advantage transitions
            "movement_intent": [27, 28],  # Aggression/defensive ratios
            "baiting_features": [30],  # Whiff bait frequency
            "velocity_tracking": [32],  # Velocity differential
        }

        self.data = {}
        self.insights = {}

    def load_all_data(self):
        """Load all available analysis and log data including oscillation logs"""
        print("📁 Loading all data files...")

        # Load JSON analysis files
        self._load_analysis_files()

        # Load log files including oscillation logs
        self._load_log_files()

        # Load oscillation-specific data
        self._load_oscillation_data()

        # Load checkpoint data
        self._load_checkpoint_data()

        print(f"✅ Data loading complete. Found:")
        for data_type, data in self.data.items():
            if isinstance(data, list):
                print(f"   {data_type}: {len(data)} entries")
            elif isinstance(data, dict):
                print(f"   {data_type}: {len(data)} files")
            else:
                print(f"   {data_type}: loaded")

    def _load_oscillation_data(self):
        """Load oscillation-specific analysis data"""
        self.data["oscillation_logs"] = []

        # Look for oscillation log files
        oscillation_patterns = [
            "oscillation_analysis_*.log",
            "oscillation_summary_*.txt",
            "enhanced_oscillation_*.log",
        ]

        for pattern in oscillation_patterns:
            files = list(self.analysis_dir.glob(pattern))
            for file_path in files:
                try:
                    with open(file_path, "r") as f:
                        content = f.read()
                        self.data["oscillation_logs"].append(
                            {
                                "file": str(file_path),
                                "content": content,
                                "type": "oscillation_log",
                            }
                        )
                    print(f"   ✓ Loaded oscillation data: {file_path.name}")
                except Exception as e:
                    print(f"   ✗ Error loading {file_path}: {e}")

    def _load_analysis_files(self):
        """Load transformer analysis JSON files"""
        analysis_files = []

        # Look for various analysis file patterns
        patterns = [
            "final_enhanced_analysis.json",
            "enhanced_transformer_analysis_*.json",
            "transformer_analysis_step_*.json",
            "transformer_analysis_win_*.json",
            "oscillation_enhanced_analysis_*.json",  # NEW
        ]

        for pattern in patterns:
            files = list(self.analysis_dir.glob(pattern))
            analysis_files.extend(files)

        self.data["analysis_files"] = []

        for file_path in analysis_files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    data["source_file"] = str(file_path)
                    self.data["analysis_files"].append(data)
                    print(f"   ✓ Loaded: {file_path.name}")
            except Exception as e:
                print(f"   ✗ Error loading {file_path}: {e}")

    def _load_log_files(self):
        """Load various log files from training including oscillation logs"""
        self.data["logs"] = {}

        # Training stats logs
        training_logs = list(self.logs_dir.glob("training_*.log"))
        self.data["logs"]["training_stats"] = self._parse_log_files(training_logs)

        # Enhanced oscillation logs from training
        enhanced_logs = list(self.logs_dir.glob("enhanced_oscillation_*.log"))
        self.data["logs"]["enhanced_oscillation"] = self._parse_log_files(enhanced_logs)

        # Performance logs
        performance_logs = list(self.logs_dir.glob("performance_*.log"))
        self.data["logs"]["performance"] = self._parse_log_files(performance_logs)

        print(f"   ✓ Loaded {len(training_logs)} training log files")
        print(f"   ✓ Loaded {len(enhanced_logs)} enhanced oscillation log files")
        print(f"   ✓ Loaded {len(performance_logs)} performance log files")

    def _parse_log_files(self, log_files):
        """Parse log files into structured data"""
        parsed_data = []

        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        # Try to parse different log formats
                        parsed_entry = self._parse_log_line(line, log_file.name)
                        if parsed_entry:
                            parsed_data.append(parsed_entry)

            except Exception as e:
                print(f"   ✗ Error parsing {log_file}: {e}")

        return parsed_data

    def _parse_log_line(self, line, source_file):
        """Parse individual log lines with enhanced parsing for oscillation data"""
        parsed = {"source_file": source_file, "raw_line": line}

        # Try to extract timestamp
        timestamp_match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", line)
        if timestamp_match:
            try:
                parsed["timestamp"] = datetime.fromisoformat(timestamp_match.group(1))
            except:
                pass

        # Extract training step
        step_match = re.search(r"Step:\s*(\d+)", line)
        if step_match:
            parsed["step"] = int(step_match.group(1))

        # Extract win rate
        win_rate_match = re.search(r"Win Rate:\s*([\d.]+)%", line)
        if win_rate_match:
            parsed["win_rate"] = float(win_rate_match.group(1)) / 100

        # Extract oscillation-specific metrics
        osc_freq_match = re.search(r"Player Oscillation Frequency:\s*([\d.]+)", line)
        if osc_freq_match:
            parsed["player_oscillation_frequency"] = float(osc_freq_match.group(1))

        space_control_match = re.search(r"Space Control Score:\s*([-\d.]+)", line)
        if space_control_match:
            parsed["space_control_score"] = float(space_control_match.group(1))

        neutral_game_match = re.search(r"Neutral Game Duration:\s*(\d+)", line)
        if neutral_game_match:
            parsed["neutral_game_duration"] = int(neutral_game_match.group(1))

        whiff_bait_match = re.search(r"Whiff Bait Attempts:\s*(\d+)", line)
        if whiff_bait_match:
            parsed["whiff_bait_attempts"] = int(whiff_bait_match.group(1))

        # Extract cross-attention weights
        attention_weights = {}
        for attention_type in ["Visual", "Strategy", "Oscillation", "Button"]:
            weight_match = re.search(rf"{attention_type}:\s*([\d.]+)", line)
            if weight_match:
                attention_weights[f"{attention_type.lower()}_attention"] = float(
                    weight_match.group(1)
                )

        if attention_weights:
            parsed["attention_weights"] = attention_weights

        return parsed

    def _load_checkpoint_data(self):
        """Load checkpoint data including oscillation-enhanced models"""
        self.data["checkpoints"] = []

        model_dirs = [
            "enhanced_oscillation_trained_models",
            "enhanced_trained_models",
            "trained_models",
        ]

        for model_dir in model_dirs:
            model_path = Path(model_dir)
            if model_path.exists():
                model_files = list(model_path.glob("*.zip"))
                for model_file in model_files:
                    self.data["checkpoints"].append(
                        {
                            "path": str(model_file),
                            "name": model_file.name,
                            "size_mb": model_file.stat().st_size / (1024 * 1024),
                            "modified": datetime.fromtimestamp(
                                model_file.stat().st_mtime
                            ),
                            "is_oscillation_enhanced": "oscillation"
                            in model_file.name.lower(),
                        }
                    )

    def analyze_oscillation_learning(self):
        """Comprehensive analysis of oscillation-based learning patterns"""
        print("\n🌊 ANALYZING OSCILLATION LEARNING PATTERNS...")

        # Extract oscillation data from logs
        oscillation_data = self._extract_oscillation_data()

        if not oscillation_data:
            print("❌ No oscillation data found!")
            return

        # Analyze different oscillation aspects
        self.insights["oscillation_frequency_analysis"] = (
            self._analyze_oscillation_frequency(oscillation_data)
        )
        self.insights["spatial_control_analysis"] = self._analyze_spatial_control(
            oscillation_data
        )
        self.insights["neutral_game_analysis"] = self._analyze_neutral_game_patterns(
            oscillation_data
        )
        self.insights["whiff_bait_analysis"] = self._analyze_whiff_bait_patterns(
            oscillation_data
        )
        self.insights["cross_attention_analysis"] = (
            self._analyze_cross_attention_weights(oscillation_data)
        )
        self.insights["oscillation_effectiveness"] = (
            self._assess_oscillation_effectiveness(oscillation_data)
        )

        print("✅ Oscillation learning analysis complete")

    def _extract_oscillation_data(self):
        """Extract oscillation data from various sources"""
        oscillation_data = []

        # From training logs
        for log_entry in self.data.get("logs", {}).get("enhanced_oscillation", []):
            if any(
                key in log_entry
                for key in [
                    "player_oscillation_frequency",
                    "space_control_score",
                    "neutral_game_duration",
                ]
            ):
                oscillation_data.append(
                    {
                        "source": "training_log",
                        "timestamp": log_entry.get("timestamp"),
                        "step": log_entry.get("step", 0),
                        "player_oscillation_frequency": log_entry.get(
                            "player_oscillation_frequency", 0
                        ),
                        "space_control_score": log_entry.get("space_control_score", 0),
                        "neutral_game_duration": log_entry.get(
                            "neutral_game_duration", 0
                        ),
                        "whiff_bait_attempts": log_entry.get("whiff_bait_attempts", 0),
                        "attention_weights": log_entry.get("attention_weights", {}),
                    }
                )

        # From analysis files
        for analysis_data in self.data.get("analysis_files", []):
            if "oscillation_metrics" in analysis_data:
                metrics = analysis_data["oscillation_metrics"]
                oscillation_data.append(
                    {
                        "source": "analysis_file",
                        "step": analysis_data.get("step", 0),
                        **metrics,
                    }
                )

        # Sort by step
        oscillation_data.sort(key=lambda x: x.get("step", 0))

        return oscillation_data

    def _analyze_oscillation_frequency(self, oscillation_data):
        """Analyze oscillation frequency patterns"""
        print("   📊 Analyzing oscillation frequency patterns...")

        if not oscillation_data:
            return {"status": "no_data"}

        frequencies = [
            d.get("player_oscillation_frequency", 0) for d in oscillation_data
        ]
        steps = [d.get("step", 0) for d in oscillation_data]

        if not any(frequencies):
            return {"status": "no_frequency_data"}

        # Calculate frequency statistics
        frequency_stats = {
            "mean_frequency": np.mean(frequencies),
            "max_frequency": np.max(frequencies),
            "min_frequency": np.min(frequencies),
            "std_frequency": np.std(frequencies),
            "frequency_trend": self._calculate_trend(frequencies),
            "optimal_frequency_range": self._determine_optimal_frequency_range(
                frequencies, oscillation_data
            ),
        }

        # Analyze frequency vs performance correlation
        win_rates = []
        for d in oscillation_data:
            # Try to find corresponding win rate data
            step = d.get("step", 0)
            win_rate = self._find_win_rate_for_step(step)
            if win_rate is not None:
                win_rates.append(win_rate)
            else:
                win_rates.append(0)

        if win_rates:
            frequency_performance_correlation = (
                np.corrcoef(frequencies, win_rates)[0, 1] if len(frequencies) > 1 else 0
            )
            frequency_stats["frequency_performance_correlation"] = (
                frequency_performance_correlation
            )

        return frequency_stats

    def _analyze_spatial_control(self, oscillation_data):
        """Analyze spatial control learning patterns"""
        print("   🎯 Analyzing spatial control patterns...")

        space_control_scores = [
            d.get("space_control_score", 0) for d in oscillation_data
        ]

        if not any(space_control_scores):
            return {"status": "no_spatial_data"}

        # Analyze spatial control evolution
        spatial_analysis = {
            "mean_space_control": np.mean(space_control_scores),
            "space_control_trend": self._calculate_trend(space_control_scores),
            "space_control_variance": np.var(space_control_scores),
            "positive_control_ratio": len([s for s in space_control_scores if s > 0])
            / len(space_control_scores),
            "dominant_control_instances": len(
                [s for s in space_control_scores if s > 0.5]
            ),
            "space_control_improvement": self._calculate_improvement_rate(
                space_control_scores
            ),
        }

        # Analyze space control phases
        if len(space_control_scores) >= 10:
            phases = self._split_into_phases(space_control_scores, 3)
            spatial_analysis["phase_analysis"] = {
                "early_phase_control": np.mean(phases[0]),
                "middle_phase_control": np.mean(phases[1]),
                "late_phase_control": np.mean(phases[2]),
                "phase_improvement": np.mean(phases[2]) - np.mean(phases[0]),
            }

        return spatial_analysis

    def _analyze_neutral_game_patterns(self, oscillation_data):
        """Analyze neutral game learning patterns"""
        print("   ⚔️ Analyzing neutral game patterns...")

        neutral_durations = [
            d.get("neutral_game_duration", 0) for d in oscillation_data
        ]

        if not any(neutral_durations):
            return {"status": "no_neutral_data"}

        # Analyze neutral game patterns
        neutral_analysis = {
            "avg_neutral_duration": np.mean(neutral_durations),
            "neutral_duration_trend": self._calculate_trend(neutral_durations),
            "neutral_consistency": 1
            / (1 + np.std(neutral_durations)),  # Higher is more consistent
            "short_neutral_ratio": len([d for d in neutral_durations if d < 30])
            / len(neutral_durations),
            "long_neutral_ratio": len([d for d in neutral_durations if d > 120])
            / len(neutral_durations),
            "optimal_neutral_range": self._determine_optimal_neutral_range(
                neutral_durations, oscillation_data
            ),
        }

        # Analyze neutral game effectiveness
        if len(neutral_durations) > 5:
            # Group by neutral duration ranges
            short_neutrals = [
                d for d in oscillation_data if d.get("neutral_game_duration", 0) < 60
            ]
            long_neutrals = [
                d for d in oscillation_data if d.get("neutral_game_duration", 0) >= 60
            ]

            if short_neutrals and long_neutrals:
                short_space_control = np.mean(
                    [d.get("space_control_score", 0) for d in short_neutrals]
                )
                long_space_control = np.mean(
                    [d.get("space_control_score", 0) for d in long_neutrals]
                )

                neutral_analysis["short_vs_long_effectiveness"] = {
                    "short_neutral_space_control": short_space_control,
                    "long_neutral_space_control": long_space_control,
                    "effectiveness_difference": long_space_control
                    - short_space_control,
                }

        return neutral_analysis

    def _analyze_whiff_bait_patterns(self, oscillation_data):
        """Analyze whiff baiting learning patterns"""
        print("   🎣 Analyzing whiff bait patterns...")

        whiff_attempts = [d.get("whiff_bait_attempts", 0) for d in oscillation_data]

        if not any(whiff_attempts):
            return {"status": "no_whiff_data"}

        # Analyze whiff bait evolution
        whiff_analysis = {
            "total_whiff_attempts": sum(whiff_attempts),
            "avg_whiff_per_game": np.mean(whiff_attempts),
            "whiff_attempt_trend": self._calculate_trend(whiff_attempts),
            "whiff_consistency": self._calculate_consistency(whiff_attempts),
            "whiff_learning_rate": self._calculate_learning_rate(whiff_attempts),
        }

        # Analyze whiff bait effectiveness
        if len(whiff_attempts) > 5:
            # Find correlation with space control
            space_scores = [d.get("space_control_score", 0) for d in oscillation_data]
            if len(space_scores) == len(whiff_attempts):
                whiff_effectiveness = (
                    np.corrcoef(whiff_attempts, space_scores)[0, 1]
                    if len(whiff_attempts) > 1
                    else 0
                )
                whiff_analysis["whiff_effectiveness_correlation"] = whiff_effectiveness

        return whiff_analysis

    def _analyze_cross_attention_weights(self, oscillation_data):
        """Analyze cross-attention weight distribution and evolution"""
        print("   🔍 Analyzing cross-attention weights...")

        attention_data = []
        for d in oscillation_data:
            weights = d.get("attention_weights", {})
            if weights:
                attention_data.append(weights)

        if not attention_data:
            return {"status": "no_attention_data"}

        # Analyze attention weight evolution
        attention_types = [
            "visual_attention",
            "strategy_attention",
            "oscillation_attention",
            "button_attention",
        ]
        attention_analysis = {}

        for attention_type in attention_types:
            weights = [d.get(attention_type, 0) for d in attention_data]
            if any(weights):
                attention_analysis[attention_type] = {
                    "mean_weight": np.mean(weights),
                    "weight_trend": self._calculate_trend(weights),
                    "weight_variance": np.var(weights),
                    "final_weight": weights[-1] if weights else 0,
                    "weight_stability": self._calculate_stability(weights),
                }

        # Analyze attention balance
        if len(attention_data) > 0:
            final_weights = attention_data[-1]
            total_weight = sum(final_weights.values())

            if total_weight > 0:
                attention_balance = {
                    name: weight / total_weight
                    for name, weight in final_weights.items()
                }
                attention_analysis["final_attention_balance"] = attention_balance

                # Determine if oscillation attention is adequately weighted
                oscillation_weight = attention_balance.get("oscillation_attention", 0)
                attention_analysis["oscillation_focus_adequacy"] = (
                    self._assess_oscillation_focus(oscillation_weight)
                )

        return attention_analysis

    def _assess_oscillation_effectiveness(self, oscillation_data):
        """Assess overall effectiveness of oscillation-based learning"""
        print("   📈 Assessing oscillation effectiveness...")

        if not oscillation_data:
            return {"status": "no_data"}

        effectiveness_metrics = {}

        # 1. Frequency optimization
        frequencies = [
            d.get("player_oscillation_frequency", 0) for d in oscillation_data
        ]
        if frequencies:
            # Optimal frequency range for fighting games is typically 1-3 Hz
            optimal_freq_instances = len([f for f in frequencies if 1.0 <= f <= 3.0])
            effectiveness_metrics["frequency_optimization"] = (
                optimal_freq_instances / len(frequencies)
            )

        # 2. Spatial control mastery
        space_scores = [d.get("space_control_score", 0) for d in oscillation_data]
        if space_scores:
            positive_control_ratio = len([s for s in space_scores if s > 0]) / len(
                space_scores
            )
            effectiveness_metrics["spatial_control_mastery"] = positive_control_ratio

        # 3. Neutral game management
        neutral_durations = [
            d.get("neutral_game_duration", 0) for d in oscillation_data
        ]
        if neutral_durations:
            # Good neutral game management should have moderate durations (30-90 frames)
            optimal_neutral_instances = len(
                [d for d in neutral_durations if 30 <= d <= 90]
            )
            effectiveness_metrics["neutral_game_management"] = (
                optimal_neutral_instances / len(neutral_durations)
            )

        # 4. Whiff bait utilization
        whiff_attempts = [d.get("whiff_bait_attempts", 0) for d in oscillation_data]
        if whiff_attempts:
            # Good whiff bait usage should show some attempts but not excessive
            active_whiff_instances = len([w for w in whiff_attempts if w > 0])
            effectiveness_metrics["whiff_bait_utilization"] = min(
                active_whiff_instances / len(whiff_attempts), 1.0
            )

        # 5. Overall oscillation integration
        if len(effectiveness_metrics) > 0:
            overall_effectiveness = np.mean(list(effectiveness_metrics.values()))
            effectiveness_metrics["overall_effectiveness"] = overall_effectiveness

            # Classification
            if overall_effectiveness >= 0.8:
                effectiveness_metrics["effectiveness_level"] = "excellent"
            elif overall_effectiveness >= 0.6:
                effectiveness_metrics["effectiveness_level"] = "good"
            elif overall_effectiveness >= 0.4:
                effectiveness_metrics["effectiveness_level"] = "developing"
            else:
                effectiveness_metrics["effectiveness_level"] = "needs_improvement"

        return effectiveness_metrics

    def analyze_enhanced_feature_importance(self):
        """Enhanced feature importance analysis for 45-feature system"""
        print("\n🧠 ANALYZING ENHANCED FEATURE IMPORTANCE (45 features)...")

        # Extract prediction data
        predictions = self._extract_prediction_data()

        if not predictions:
            print("❌ No prediction data found!")
            return

        # Analyze feature importance by category
        self.insights["strategic_feature_importance"] = (
            self._analyze_feature_category_importance(predictions, "strategic", 0, 21)
        )
        self.insights["oscillation_feature_importance"] = (
            self._analyze_feature_category_importance(
                predictions, "oscillation", 21, 33
            )
        )
        self.insights["button_feature_importance"] = (
            self._analyze_feature_category_importance(predictions, "button", 33, 45)
        )

        # Overall feature ranking
        self.insights["overall_feature_ranking"] = self._rank_all_features(predictions)

        print("✅ Enhanced feature importance analysis complete")

    def _analyze_feature_category_importance(
        self, predictions, category_name, start_idx, end_idx
    ):
        """Analyze importance of features in a specific category"""
        print(f"   🔍 Analyzing {category_name} features ({start_idx}-{end_idx-1})...")

        category_features = {}
        feature_names = self.all_feature_names[start_idx:end_idx]

        for pred in predictions:
            if (
                "strategic_features" in pred
                and len(pred["strategic_features"]) >= end_idx
            ):
                features = pred["strategic_features"][start_idx:end_idx]
                attack_timing = pred.get("attack_timing", 0)
                defend_timing = pred.get("defend_timing", 0)

                for i, feature_value in enumerate(features):
                    if i < len(feature_names):
                        feature_name = feature_names[i]
                        if feature_name not in category_features:
                            category_features[feature_name] = {
                                "values": [],
                                "attack_correlations": [],
                                "defend_correlations": [],
                            }

                        category_features[feature_name]["values"].append(feature_value)
                        category_features[feature_name]["attack_correlations"].append(
                            feature_value * attack_timing
                        )
                        category_features[feature_name]["defend_correlations"].append(
                            feature_value * defend_timing
                        )

        # Calculate importance metrics
        importance_metrics = {}
        for feature_name, data in category_features.items():
            values = np.array(data["values"])

            importance_metrics[feature_name] = {
                "mean_activation": np.mean(np.abs(values)),
                "variance": np.var(values),
                "max_activation": np.max(np.abs(values)),
                "activation_frequency": np.mean(np.abs(values) > 0.1),
                "attack_correlation": np.mean(data["attack_correlations"]),
                "defend_correlation": np.mean(data["defend_correlations"]),
                "predictive_power": np.mean(np.abs(values)) * np.var(values),
                "feature_stability": 1 / (1 + np.std(values)),  # Higher is more stable
            }

        # Rank features in this category
        ranked_features = sorted(
            importance_metrics.items(),
            key=lambda x: x[1]["predictive_power"],
            reverse=True,
        )

        return {
            "category_name": category_name,
            "feature_metrics": importance_metrics,
            "ranked_features": ranked_features,
            "category_summary": self._summarize_category_importance(
                importance_metrics, category_name
            ),
        }

    def _summarize_category_importance(self, importance_metrics, category_name):
        """Summarize importance findings for a feature category"""
        if not importance_metrics:
            return {"status": "no_data"}

        activations = [m["mean_activation"] for m in importance_metrics.values()]
        variances = [m["variance"] for m in importance_metrics.values()]

        return {
            "total_features": len(importance_metrics),
            "avg_activation": np.mean(activations),
            "activation_consistency": 1 / (1 + np.std(activations)),
            "high_activation_features": len([a for a in activations if a > 0.3]),
            "underutilized_features": len([a for a in activations if a < 0.1]),
            "category_effectiveness": np.mean(activations) * np.mean(variances),
        }

    def _rank_all_features(self, predictions):
        """Rank all 45 features by importance"""
        all_features = {}

        for pred in predictions:
            if "strategic_features" in pred and len(pred["strategic_features"]) >= 45:
                features = pred["strategic_features"]
                attack_timing = pred.get("attack_timing", 0)
                defend_timing = pred.get("defend_timing", 0)

                for i, feature_value in enumerate(features):
                    if i < len(self.all_feature_names):
                        feature_name = self.all_feature_names[i]
                        if feature_name not in all_features:
                            all_features[feature_name] = {
                                "values": [],
                                "attack_correlations": [],
                                "defend_correlations": [],
                                "feature_index": i,
                                "category": self._get_feature_category(i),
                            }

                        all_features[feature_name]["values"].append(feature_value)
                        all_features[feature_name]["attack_correlations"].append(
                            feature_value * attack_timing
                        )
                        all_features[feature_name]["defend_correlations"].append(
                            feature_value * defend_timing
                        )

        # Calculate comprehensive importance scores
        feature_rankings = {}
        for feature_name, data in all_features.items():
            values = np.array(data["values"])

            # Multi-factor importance score
            activation_score = np.mean(np.abs(values))
            variance_score = np.var(values)
            correlation_score = max(
                abs(np.mean(data["attack_correlations"])),
                abs(np.mean(data["defend_correlations"])),
            )

            # Weighted importance score
            importance_score = (
                activation_score * 0.4 + variance_score * 0.3 + correlation_score * 0.3
            )

            feature_rankings[feature_name] = {
                "importance_score": importance_score,
                "activation_score": activation_score,
                "variance_score": variance_score,
                "correlation_score": correlation_score,
                "category": data["category"],
                "feature_index": data["feature_index"],
            }

        # Sort by importance
        ranked_features = sorted(
            feature_rankings.items(),
            key=lambda x: x[1]["importance_score"],
            reverse=True,
        )

        return {
            "ranked_features": ranked_features,
            "top_10_features": ranked_features[:10],
            "bottom_10_features": ranked_features[-10:],
            "category_rankings": self._rank_categories_by_importance(feature_rankings),
        }

    def _get_feature_category(self, feature_index):
        """Get category name for a feature index"""
        if feature_index < 21:
            return "strategic"
        elif feature_index < 33:
            return "oscillation"
        else:
            return "button"

    def _rank_categories_by_importance(self, feature_rankings):
        """Rank feature categories by average importance"""
        category_scores = defaultdict(list)

        for feature_name, data in feature_rankings.items():
            category = data["category"]
            importance = data["importance_score"]
            category_scores[category].append(importance)

        category_rankings = {}
        for category, scores in category_scores.items():
            category_rankings[category] = {
                "avg_importance": np.mean(scores),
                "total_features": len(scores),
                "high_importance_features": len([s for s in scores if s > 0.3]),
                "category_consistency": 1 / (1 + np.std(scores)),
            }

        return sorted(
            category_rankings.items(),
            key=lambda x: x[1]["avg_importance"],
            reverse=True,
        )

    def generate_oscillation_recommendations(self):
        """Generate specific recommendations for oscillation-based improvements"""
        print("\n💡 GENERATING OSCILLATION-SPECIFIC RECOMMENDATIONS...")

        recommendations = {
            "oscillation_improvements": [],
            "spatial_control_improvements": [],
            "neutral_game_improvements": [],
            "attention_mechanism_improvements": [],
            "feature_engineering_improvements": [],
            "training_improvements": [],
        }

        # Analyze current oscillation performance
        self._recommend_oscillation_improvements(recommendations)
        self._recommend_spatial_control_improvements(recommendations)
        self._recommend_neutral_game_improvements(recommendations)
        self._recommend_attention_improvements(recommendations)
        self._recommend_oscillation_feature_improvements(recommendations)
        self._recommend_oscillation_training_improvements(recommendations)

        # Prioritize oscillation recommendations
        self._prioritize_oscillation_recommendations(recommendations)

        self.insights["oscillation_recommendations"] = recommendations
        print("✅ Oscillation recommendations generated")

    def _recommend_oscillation_improvements(self, recommendations):
        """Recommend improvements for oscillation frequency and amplitude"""
        osc_analysis = self.insights.get("oscillation_frequency_analysis", {})

        if osc_analysis.get("status") == "no_data":
            recommendations["oscillation_improvements"].append(
                {
                    "type": "data_collection",
                    "issue": "No oscillation data found",
                    "suggestion": "Ensure oscillation tracking is enabled in wrapper",
                    "implementation": "Check OscillationTracker initialization and logging",
                    "priority": "critical",
                }
            )
            return

        mean_freq = osc_analysis.get("mean_frequency", 0)
        freq_trend = osc_analysis.get("frequency_trend", 0)

        # Optimal frequency range for fighting games is 1-3 Hz
        if mean_freq < 1.0:
            recommendations["oscillation_improvements"].append(
                {
                    "type": "frequency_tuning",
                    "issue": f"Low oscillation frequency ({mean_freq:.2f} Hz)",
                    "suggestion": "Increase movement sensitivity in oscillation detection",
                    "implementation": [
                        "Reduce oscillation detection threshold in OscillationTracker",
                        "Adjust direction change sensitivity",
                        "Consider shorter history window for faster detection",
                    ],
                    "priority": "high",
                }
            )
        elif mean_freq > 4.0:
            recommendations["oscillation_improvements"].append(
                {
                    "type": "frequency_tuning",
                    "issue": f"High oscillation frequency ({mean_freq:.2f} Hz)",
                    "suggestion": "Reduce noise in oscillation detection",
                    "implementation": [
                        "Increase minimum movement threshold",
                        "Add movement smoothing filter",
                        "Implement oscillation validation",
                    ],
                    "priority": "high",
                }
            )

        if freq_trend < -0.1:
            recommendations["oscillation_improvements"].append(
                {
                    "type": "learning_stability",
                    "issue": "Declining oscillation frequency over time",
                    "suggestion": "Investigate why oscillation patterns are degrading",
                    "implementation": [
                        "Add oscillation pattern rewards",
                        "Implement oscillation consistency bonuses",
                        "Check for overfitting in movement patterns",
                    ],
                    "priority": "high",
                }
            )

        # Frequency-performance correlation
        freq_perf_corr = osc_analysis.get("frequency_performance_correlation", 0)
        if abs(freq_perf_corr) < 0.1:
            recommendations["oscillation_improvements"].append(
                {
                    "type": "correlation_improvement",
                    "issue": "Weak correlation between oscillation frequency and performance",
                    "suggestion": "Enhance oscillation feature engineering",
                    "implementation": [
                        "Add oscillation effectiveness metrics",
                        "Implement context-aware oscillation scoring",
                        "Include opponent oscillation interaction features",
                    ],
                    "priority": "medium",
                }
            )

    def _recommend_spatial_control_improvements(self, recommendations):
        """Recommend improvements for spatial control analysis"""
        spatial_analysis = self.insights.get("spatial_control_analysis", {})

        if spatial_analysis.get("status") == "no_spatial_data":
            recommendations["spatial_control_improvements"].append(
                {
                    "type": "data_collection",
                    "issue": "No spatial control data found",
                    "suggestion": "Ensure spatial control calculation is working",
                    "implementation": "Check _calculate_space_control method in OscillationTracker",
                    "priority": "critical",
                }
            )
            return

        mean_control = spatial_analysis.get("mean_space_control", 0)
        control_trend = spatial_analysis.get("space_control_trend", 0)
        positive_ratio = spatial_analysis.get("positive_control_ratio", 0)

        if mean_control < 0.1:
            recommendations["spatial_control_improvements"].append(
                {
                    "type": "control_enhancement",
                    "issue": f"Low spatial control score ({mean_control:.3f})",
                    "suggestion": "Enhance spatial control calculation",
                    "implementation": [
                        "Increase center control bonus weight",
                        "Add corner pressure detection",
                        "Implement stage positioning rewards",
                        "Add opponent cornering bonus",
                    ],
                    "priority": "high",
                }
            )

        if positive_ratio < 0.4:
            recommendations["spatial_control_improvements"].append(
                {
                    "type": "control_consistency",
                    "issue": f"Low positive control ratio ({positive_ratio:.2f})",
                    "suggestion": "Improve spatial control consistency",
                    "implementation": [
                        "Add spatial control momentum tracking",
                        "Implement control state persistence",
                        "Add positional advantage memory",
                    ],
                    "priority": "high",
                }
            )

        if control_trend < -0.05:
            recommendations["spatial_control_improvements"].append(
                {
                    "type": "control_degradation",
                    "issue": "Declining spatial control over time",
                    "suggestion": "Investigate spatial control learning degradation",
                    "implementation": [
                        "Add spatial control regularization",
                        "Implement control consistency rewards",
                        "Check for spatial control overfitting",
                    ],
                    "priority": "high",
                }
            )

        # Phase analysis recommendations
        phase_analysis = spatial_analysis.get("phase_analysis", {})
        if phase_analysis:
            improvement = phase_analysis.get("phase_improvement", 0)
            if improvement < 0.05:
                recommendations["spatial_control_improvements"].append(
                    {
                        "type": "learning_progression",
                        "issue": "Minimal spatial control improvement across training phases",
                        "suggestion": "Implement progressive spatial control curriculum",
                        "implementation": [
                            "Start with basic center control",
                            "Add corner avoidance training",
                            "Implement advanced positioning strategies",
                            "Add opponent pressure resistance training",
                        ],
                        "priority": "medium",
                    }
                )

    def _recommend_neutral_game_improvements(self, recommendations):
        """Recommend improvements for neutral game analysis"""
        neutral_analysis = self.insights.get("neutral_game_analysis", {})

        if neutral_analysis.get("status") == "no_neutral_data":
            recommendations["neutral_game_improvements"].append(
                {
                    "type": "data_collection",
                    "issue": "No neutral game data found",
                    "suggestion": "Ensure neutral game detection is working",
                    "implementation": "Check neutral game state tracking in OscillationTracker",
                    "priority": "critical",
                }
            )
            return

        avg_duration = neutral_analysis.get("avg_neutral_duration", 0)
        consistency = neutral_analysis.get("neutral_consistency", 0)
        short_ratio = neutral_analysis.get("short_neutral_ratio", 0)
        long_ratio = neutral_analysis.get("long_neutral_ratio", 0)

        # Optimal neutral game duration is typically 30-90 frames
        if avg_duration < 30:
            recommendations["neutral_game_improvements"].append(
                {
                    "type": "neutral_duration",
                    "issue": f"Very short neutral game duration ({avg_duration:.1f} frames)",
                    "suggestion": "Encourage longer neutral game engagement",
                    "implementation": [
                        "Add neutral game persistence rewards",
                        "Reduce aggressive action bonuses",
                        "Implement patience training scenarios",
                    ],
                    "priority": "high",
                }
            )
        elif avg_duration > 120:
            recommendations["neutral_game_improvements"].append(
                {
                    "type": "neutral_duration",
                    "issue": f"Very long neutral game duration ({avg_duration:.1f} frames)",
                    "suggestion": "Encourage more decisive neutral game actions",
                    "implementation": [
                        "Add neutral game breakthrough rewards",
                        "Implement opportunity recognition training",
                        "Add neutral game timeout penalties",
                    ],
                    "priority": "medium",
                }
            )

        if consistency < 0.5:
            recommendations["neutral_game_improvements"].append(
                {
                    "type": "neutral_consistency",
                    "issue": f"Low neutral game consistency ({consistency:.3f})",
                    "suggestion": "Improve neutral game pattern recognition",
                    "implementation": [
                        "Add neutral game state features",
                        "Implement neutral game pattern rewards",
                        "Add consistent neutral game bonuses",
                    ],
                    "priority": "high",
                }
            )

        # Effectiveness analysis
        effectiveness = neutral_analysis.get("short_vs_long_effectiveness", {})
        if effectiveness:
            effectiveness_diff = effectiveness.get("effectiveness_difference", 0)
            if abs(effectiveness_diff) > 0.2:
                recommendations["neutral_game_improvements"].append(
                    {
                        "type": "neutral_effectiveness",
                        "issue": f"Significant effectiveness difference between short/long neutral games",
                        "suggestion": "Balance neutral game duration strategies",
                        "implementation": [
                            "Add duration-adaptive neutral game rewards",
                            "Implement context-sensitive neutral game training",
                            "Add neutral game outcome analysis",
                        ],
                        "priority": "medium",
                    }
                )

    def _recommend_attention_improvements(self, recommendations):
        """Recommend improvements for cross-attention mechanisms"""
        attention_analysis = self.insights.get("cross_attention_analysis", {})

        if attention_analysis.get("status") == "no_attention_data":
            recommendations["attention_mechanism_improvements"].append(
                {
                    "type": "data_collection",
                    "issue": "No cross-attention data found",
                    "suggestion": "Ensure cross-attention logging is enabled",
                    "implementation": "Check attention weight logging in training callback",
                    "priority": "critical",
                }
            )
            return

        # Analyze attention balance
        attention_balance = attention_analysis.get("final_attention_balance", {})
        if attention_balance:
            oscillation_weight = attention_balance.get("oscillation_attention", 0)
            visual_weight = attention_balance.get("visual_attention", 0)
            strategy_weight = attention_balance.get("strategy_attention", 0)
            button_weight = attention_balance.get("button_attention", 0)

            # Oscillation attention should be significant but not overwhelming
            if oscillation_weight < 0.15:
                recommendations["attention_mechanism_improvements"].append(
                    {
                        "type": "attention_balance",
                        "issue": f"Low oscillation attention weight ({oscillation_weight:.3f})",
                        "suggestion": "Increase oscillation attention importance",
                        "implementation": [
                            "Add oscillation attention regularization",
                            "Implement oscillation attention boosting",
                            "Add oscillation-specific attention heads",
                        ],
                        "priority": "high",
                    }
                )
            elif oscillation_weight > 0.6:
                recommendations["attention_mechanism_improvements"].append(
                    {
                        "type": "attention_balance",
                        "issue": f"Excessive oscillation attention weight ({oscillation_weight:.3f})",
                        "suggestion": "Rebalance attention weights",
                        "implementation": [
                            "Add attention weight regularization",
                            "Implement attention diversity loss",
                            "Add multi-modal attention balancing",
                        ],
                        "priority": "high",
                    }
                )

            # Visual attention should remain important
            if visual_weight < 0.2:
                recommendations["attention_mechanism_improvements"].append(
                    {
                        "type": "visual_attention",
                        "issue": f"Low visual attention weight ({visual_weight:.3f})",
                        "suggestion": "Maintain visual attention importance",
                        "implementation": [
                            "Add visual attention minimum constraints",
                            "Implement visual-oscillation attention coupling",
                            "Add visual feature enhancement",
                        ],
                        "priority": "medium",
                    }
                )

        # Analyze attention stability
        for attention_type in [
            "oscillation_attention",
            "visual_attention",
            "strategy_attention",
            "button_attention",
        ]:
            attention_data = attention_analysis.get(attention_type, {})
            if attention_data:
                stability = attention_data.get("weight_stability", 0)
                if stability < 0.5:
                    recommendations["attention_mechanism_improvements"].append(
                        {
                            "type": "attention_stability",
                            "issue": f"Unstable {attention_type} ({stability:.3f})",
                            "suggestion": f"Improve {attention_type} stability",
                            "implementation": [
                                "Add attention weight smoothing",
                                "Implement attention momentum",
                                "Add attention stability regularization",
                            ],
                            "priority": "medium",
                        }
                    )

    def _recommend_oscillation_feature_improvements(self, recommendations):
        """Recommend improvements for oscillation feature engineering"""

        # Analyze oscillation feature importance
        osc_importance = self.insights.get("oscillation_feature_importance", {})
        if osc_importance:
            ranked_features = osc_importance.get("ranked_features", [])

            # Check for underutilized oscillation features
            underutilized = [
                f for f in ranked_features if f[1]["mean_activation"] < 0.1
            ]
            if underutilized:
                recommendations["feature_engineering_improvements"].append(
                    {
                        "type": "underutilized_oscillation_features",
                        "issue": f"{len(underutilized)} oscillation features are underutilized",
                        "suggestion": "Enhance underutilized oscillation features",
                        "features": [f[0] for f in underutilized],
                        "implementation": [
                            "Review feature calculation methods",
                            "Add feature normalization",
                            "Implement feature interaction terms",
                            "Add feature importance-based weighting",
                        ],
                        "priority": "medium",
                    }
                )

        # Suggest new oscillation features
        recommendations["feature_engineering_improvements"].extend(
            [
                {
                    "type": "new_oscillation_feature",
                    "suggestion": "Add oscillation pattern recognition features",
                    "rationale": "Detect specific oscillation patterns (e.g., dash-dancing, wave-dashing)",
                    "implementation": [
                        "Add pattern matching algorithms",
                        "Implement oscillation signature detection",
                        "Add pattern frequency analysis",
                    ],
                    "priority": "high",
                },
                {
                    "type": "new_oscillation_feature",
                    "suggestion": "Add oscillation prediction features",
                    "rationale": "Predict opponent oscillation patterns",
                    "implementation": [
                        "Add opponent oscillation modeling",
                        "Implement oscillation prediction network",
                        "Add oscillation counter-strategy features",
                    ],
                    "priority": "high",
                },
                {
                    "type": "new_oscillation_feature",
                    "suggestion": "Add oscillation effectiveness metrics",
                    "rationale": "Measure oscillation success rate",
                    "implementation": [
                        "Track oscillation outcomes",
                        "Add oscillation-to-attack conversion rates",
                        "Implement oscillation ROI analysis",
                    ],
                    "priority": "medium",
                },
                {
                    "type": "new_oscillation_feature",
                    "suggestion": "Add multi-scale oscillation analysis",
                    "rationale": "Analyze oscillation at different time scales",
                    "implementation": [
                        "Add short-term oscillation (1-2 seconds)",
                        "Add medium-term oscillation (3-5 seconds)",
                        "Add long-term oscillation (full round)",
                        "Implement temporal oscillation fusion",
                    ],
                    "priority": "medium",
                },
            ]
        )

    def _recommend_oscillation_training_improvements(self, recommendations):
        """Recommend training improvements specific to oscillation learning"""

        # Analyze oscillation effectiveness
        osc_effectiveness = self.insights.get("oscillation_effectiveness", {})
        if osc_effectiveness:
            effectiveness_level = osc_effectiveness.get(
                "effectiveness_level", "unknown"
            )

            if effectiveness_level in ["needs_improvement", "developing"]:
                recommendations["training_improvements"].append(
                    {
                        "type": "oscillation_curriculum",
                        "issue": f"Oscillation effectiveness level: {effectiveness_level}",
                        "suggestion": "Implement oscillation-focused curriculum learning",
                        "implementation": [
                            "Stage 1: Basic oscillation patterns (20% of training)",
                            "Stage 2: Spatial control through oscillation (30% of training)",
                            "Stage 3: Advanced oscillation strategies (25% of training)",
                            "Stage 4: Oscillation-based neutral game (25% of training)",
                        ],
                        "priority": "high",
                    }
                )

        # Reward shaping for oscillation
        recommendations["training_improvements"].extend(
            [
                {
                    "type": "oscillation_reward_shaping",
                    "suggestion": "Enhanced oscillation reward engineering",
                    "implementation": [
                        "Add oscillation frequency optimization rewards",
                        "Implement spatial control progression bonuses",
                        "Add neutral game oscillation rewards",
                        "Implement oscillation effectiveness bonuses",
                    ],
                    "priority": "high",
                },
                {
                    "type": "oscillation_exploration",
                    "suggestion": "Oscillation-guided exploration",
                    "implementation": [
                        "Add oscillation pattern entropy bonuses",
                        "Implement oscillation diversity rewards",
                        "Add oscillation innovation bonuses",
                    ],
                    "priority": "medium",
                },
                {
                    "type": "oscillation_validation",
                    "suggestion": "Add oscillation learning validation",
                    "implementation": [
                        "Implement oscillation pattern tests",
                        "Add oscillation effectiveness benchmarks",
                        "Create oscillation learning metrics",
                    ],
                    "priority": "medium",
                },
            ]
        )

    def _prioritize_oscillation_recommendations(self, recommendations):
        """Prioritize oscillation-specific recommendations"""

        # Count priority levels
        priority_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        for category in recommendations:
            if isinstance(recommendations[category], list):
                for rec in recommendations[category]:
                    priority = rec.get("priority", "low")
                    priority_counts[priority] += 1

                # Sort by priority within category
                recommendations[category].sort(
                    key=lambda x: {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(
                        x.get("priority", "low"), 0
                    ),
                    reverse=True,
                )

        recommendations["priority_summary"] = priority_counts

    def create_oscillation_visualizations(self):
        """Create visualizations specific to oscillation analysis"""
        print("\n📊 CREATING OSCILLATION VISUALIZATIONS...")

        plots_dir = self.output_dir / "oscillation_plots"
        plots_dir.mkdir(exist_ok=True)

        # Create oscillation-specific plots
        self._plot_oscillation_frequency_evolution(plots_dir)
        self._plot_spatial_control_analysis(plots_dir)
        self._plot_neutral_game_patterns(plots_dir)
        self._plot_cross_attention_weights(plots_dir)
        self._plot_oscillation_feature_importance(plots_dir)
        self._plot_oscillation_effectiveness_dashboard(plots_dir)

        print(f"✅ Oscillation visualizations saved to {plots_dir}")

    def _plot_oscillation_frequency_evolution(self, plots_dir):
        """Plot oscillation frequency evolution over training"""
        osc_data = self._extract_oscillation_data()
        if not osc_data:
            return

        frequencies = [d.get("player_oscillation_frequency", 0) for d in osc_data]
        steps = [d.get("step", 0) for d in osc_data]

        if not any(frequencies):
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Frequency over time
        ax1.plot(
            steps,
            frequencies,
            "b-",
            linewidth=2,
            alpha=0.7,
            label="Oscillation Frequency",
        )
        ax1.axhline(
            y=1.0, color="g", linestyle="--", alpha=0.5, label="Optimal Min (1.0 Hz)"
        )
        ax1.axhline(
            y=3.0, color="g", linestyle="--", alpha=0.5, label="Optimal Max (3.0 Hz)"
        )
        ax1.fill_between(
            steps, 1.0, 3.0, alpha=0.1, color="green", label="Optimal Range"
        )

        # Add moving average
        if len(frequencies) > 10:
            window = max(10, len(frequencies) // 20)
            moving_avg = (
                pd.Series(frequencies).rolling(window=window, center=True).mean()
            )
            ax1.plot(
                steps,
                moving_avg,
                "r-",
                linewidth=3,
                label=f"{window}-step Moving Average",
            )

        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Oscillation Frequency (Hz)")
        ax1.set_title("Oscillation Frequency Evolution During Training")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Frequency distribution
        ax2.hist(frequencies, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        ax2.axvline(x=1.0, color="g", linestyle="--", alpha=0.7, label="Optimal Min")
        ax2.axvline(x=3.0, color="g", linestyle="--", alpha=0.7, label="Optimal Max")
        ax2.axvline(
            x=np.mean(frequencies),
            color="r",
            linestyle="-",
            linewidth=2,
            label=f"Mean: {np.mean(frequencies):.2f}",
        )
        ax2.set_xlabel("Oscillation Frequency (Hz)")
        ax2.set_ylabel("Frequency Count")
        ax2.set_title("Oscillation Frequency Distribution")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            plots_dir / "oscillation_frequency_evolution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_spatial_control_analysis(self, plots_dir):
        """Plot spatial control analysis"""
        osc_data = self._extract_oscillation_data()
        if not osc_data:
            return

        space_scores = [d.get("space_control_score", 0) for d in osc_data]
        steps = [d.get("step", 0) for d in osc_data]

        if not any(space_scores):
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Space control over time
        ax1.plot(steps, space_scores, "g-", linewidth=2, alpha=0.7)
        ax1.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax1.fill_between(
            steps,
            0,
            space_scores,
            where=np.array(space_scores) > 0,
            alpha=0.3,
            color="green",
            label="Positive Control",
        )
        ax1.fill_between(
            steps,
            0,
            space_scores,
            where=np.array(space_scores) < 0,
            alpha=0.3,
            color="red",
            label="Negative Control",
        )
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Space Control Score")
        ax1.set_title("Spatial Control Evolution")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Space control distribution
        ax2.hist(
            space_scores, bins=30, alpha=0.7, color="lightgreen", edgecolor="black"
        )
        ax2.axvline(x=0, color="k", linestyle="-", alpha=0.5)
        ax2.axvline(
            x=np.mean(space_scores),
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(space_scores):.3f}",
        )
        ax2.set_xlabel("Space Control Score")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Space Control Distribution")
        ax2.legend()
        ax2.grid(alpha=0.3)

        # Positive vs negative control over time
        positive_ratios = []
        window_size = max(50, len(space_scores) // 20)
        for i in range(window_size, len(space_scores)):
            window = space_scores[i - window_size : i]
            positive_ratio = len([s for s in window if s > 0]) / len(window)
            positive_ratios.append(positive_ratio)

        if positive_ratios:
            ax3.plot(steps[window_size:], positive_ratios, "purple", linewidth=2)
            ax3.axhline(
                y=0.5, color="k", linestyle="--", alpha=0.5, label="50% Threshold"
            )
            ax3.set_xlabel("Training Step")
            ax3.set_ylabel("Positive Control Ratio")
            ax3.set_title(f"Positive Control Ratio (Rolling {window_size}-step Window)")
            ax3.legend()
            ax3.grid(alpha=0.3)

        # Space control effectiveness by phase
        if len(space_scores) > 100:
            phases = self._split_into_phases(space_scores, 5)
            phase_names = ["Early", "Mid-Early", "Middle", "Mid-Late", "Late"]
            phase_means = [np.mean(phase) for phase in phases]

            colors = ["red", "orange", "yellow", "lightgreen", "green"]
            bars = ax4.bar(phase_names, phase_means, color=colors, alpha=0.7)
            ax4.axhline(y=0, color="k", linestyle="-", alpha=0.3)
            ax4.set_ylabel("Average Space Control Score")
            ax4.set_title("Space Control by Training Phase")
            ax4.grid(axis="y", alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, phase_means):
                height = bar.get_height()
                ax4.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        plt.tight_layout()
        plt.savefig(
            plots_dir / "spatial_control_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_neutral_game_patterns(self, plots_dir):
        """Plot neutral game pattern analysis"""
        osc_data = self._extract_oscillation_data()
        if not osc_data:
            return

        neutral_durations = [d.get("neutral_game_duration", 0) for d in osc_data]
        steps = [d.get("step", 0) for d in osc_data]

        if not any(neutral_durations):
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Neutral game duration over time
        ax1.plot(steps, neutral_durations, "orange", linewidth=2, alpha=0.7)
        ax1.axhline(
            y=30, color="g", linestyle="--", alpha=0.5, label="Optimal Min (30 frames)"
        )
        ax1.axhline(
            y=90, color="g", linestyle="--", alpha=0.5, label="Optimal Max (90 frames)"
        )
        ax1.fill_between(steps, 30, 90, alpha=0.1, color="green", label="Optimal Range")
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Neutral Game Duration (frames)")
        ax1.set_title("Neutral Game Duration Evolution")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Duration distribution
        ax2.hist(
            neutral_durations, bins=25, alpha=0.7, color="orange", edgecolor="black"
        )
        ax2.axvline(x=30, color="g", linestyle="--", alpha=0.7, label="Optimal Min")
        ax2.axvline(x=90, color="g", linestyle="--", alpha=0.7, label="Optimal Max")
        ax2.axvline(
            x=np.mean(neutral_durations),
            color="r",
            linestyle="-",
            linewidth=2,
            label=f"Mean: {np.mean(neutral_durations):.1f}",
        )
        ax2.set_xlabel("Neutral Game Duration (frames)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Neutral Game Duration Distribution")
        ax2.legend()
        ax2.grid(alpha=0.3)

        # Categorize neutral game lengths
        short_neutral = len([d for d in neutral_durations if d < 30])
        optimal_neutral = len([d for d in neutral_durations if 30 <= d <= 90])
        long_neutral = len([d for d in neutral_durations if d > 90])

        categories = [
            "Short\n(<30 frames)",
            "Optimal\n(30-90 frames)",
            "Long\n(>90 frames)",
        ]
        counts = [short_neutral, optimal_neutral, long_neutral]
        colors = ["red", "green", "blue"]

        ax3.pie(
            counts, labels=categories, colors=colors, autopct="%1.1f%%", startangle=90
        )
        ax3.set_title("Neutral Game Duration Categories")

        # Neutral game effectiveness (duration vs space control)
        space_scores = [d.get("space_control_score", 0) for d in osc_data]
        if len(space_scores) == len(neutral_durations):
            ax4.scatter(neutral_durations, space_scores, alpha=0.6, color="purple")
            ax4.axvline(x=30, color="g", linestyle="--", alpha=0.5)
            ax4.axvline(x=90, color="g", linestyle="--", alpha=0.5)
            ax4.axhline(y=0, color="k", linestyle="-", alpha=0.3)
            ax4.set_xlabel("Neutral Game Duration (frames)")
            ax4.set_ylabel("Space Control Score")
            ax4.set_title("Neutral Game Duration vs Space Control")
            ax4.grid(alpha=0.3)

            # Add correlation coefficient
            if len(neutral_durations) > 1:
                corr = np.corrcoef(neutral_durations, space_scores)[0, 1]
                ax4.text(
                    0.05,
                    0.95,
                    f"Correlation: {corr:.3f}",
                    transform=ax4.transAxes,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

        plt.tight_layout()
        plt.savefig(
            plots_dir / "neutral_game_patterns.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_cross_attention_weights(self, plots_dir):
        """Plot cross-attention weight evolution"""
        osc_data = self._extract_oscillation_data()
        if not osc_data:
            return

        # Extract attention weights
        attention_data = []
        steps = []
        for d in osc_data:
            weights = d.get("attention_weights", {})
            if weights:
                attention_data.append(weights)
                steps.append(d.get("step", 0))

        if not attention_data:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Extract individual attention series
        attention_types = [
            "visual_attention",
            "strategy_attention",
            "oscillation_attention",
            "button_attention",
        ]
        colors = ["blue", "green", "red", "orange"]

        for i, (attention_type, color) in enumerate(zip(attention_types, colors)):
            weights = [d.get(attention_type, 0) for d in attention_data]
            if any(weights):
                ax1.plot(
                    steps,
                    weights,
                    color=color,
                    linewidth=2,
                    label=attention_type.replace("_", " ").title(),
                    alpha=0.8,
                )

        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Attention Weight")
        ax1.set_title("Cross-Attention Weight Evolution")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Final attention distribution
        if attention_data:
            final_weights = attention_data[-1]
            labels = [name.replace("_", " ").title() for name in final_weights.keys()]
            values = list(final_weights.values())

            ax2.pie(
                values, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
            )
            ax2.set_title("Final Attention Weight Distribution")

        # Attention balance over time
        balance_scores = []
        for weights in attention_data:
            # Calculate attention balance (entropy-based)
            values = np.array(list(weights.values()))
            if np.sum(values) > 0:
                normalized = values / np.sum(values)
                entropy = -np.sum(normalized * np.log(normalized + 1e-10))
                balance_scores.append(entropy)
            else:
                balance_scores.append(0)

        if balance_scores:
            ax3.plot(steps, balance_scores, "purple", linewidth=2)
            ax3.axhline(
                y=np.log(4),
                color="k",
                linestyle="--",
                alpha=0.5,
                label=f"Perfect Balance: {np.log(4):.2f}",
            )
            ax3.set_xlabel("Training Step")
            ax3.set_ylabel("Attention Balance (Entropy)")
            ax3.set_title("Attention Balance Evolution")
            ax3.legend()
            ax3.grid(alpha=0.3)

        # Oscillation attention focus
        osc_weights = [d.get("oscillation_attention", 0) for d in attention_data]
        if osc_weights:
            ax4.plot(
                steps, osc_weights, "red", linewidth=3, label="Oscillation Attention"
            )
            ax4.axhline(
                y=0.15, color="g", linestyle="--", alpha=0.7, label="Minimum Threshold"
            )
            ax4.axhline(
                y=0.6,
                color="orange",
                linestyle="--",
                alpha=0.7,
                label="Maximum Threshold",
            )
            ax4.fill_between(
                steps, 0.15, 0.6, alpha=0.1, color="green", label="Optimal Range"
            )
            ax4.set_xlabel("Training Step")
            ax4.set_ylabel("Oscillation Attention Weight")
            ax4.set_title("Oscillation Attention Focus")
            ax4.legend()
            ax4.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            plots_dir / "cross_attention_weights.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_oscillation_feature_importance(self, plots_dir):
        """Plot oscillation feature importance analysis"""
        osc_importance = self.insights.get("oscillation_feature_importance", {})
        if not osc_importance or not osc_importance.get("ranked_features"):
            return

        ranked_features = osc_importance["ranked_features"]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Top oscillation features by importance
        top_features = ranked_features[:8]
        feature_names = [f[0].replace("_", " ").title() for f in top_features]
        importance_scores = [f[1]["predictive_power"] for f in top_features]

        bars = ax1.barh(range(len(feature_names)), importance_scores, color="skyblue")
        ax1.set_yticks(range(len(feature_names)))
        ax1.set_yticklabels(feature_names, fontsize=10)
        ax1.set_xlabel("Predictive Power")
        ax1.set_title("Top Oscillation Features by Importance")
        ax1.grid(axis="x", alpha=0.3)

        # Feature activation levels
        activations = [f[1]["mean_activation"] for f in top_features]
        ax2.barh(range(len(feature_names)), activations, color="lightgreen")
        ax2.set_yticks(range(len(feature_names)))
        ax2.set_yticklabels(feature_names, fontsize=10)
        ax2.set_xlabel("Mean Activation")
        ax2.set_title("Top Oscillation Features by Activation")
        ax2.grid(axis="x", alpha=0.3)

        # Feature correlations
        attack_corrs = [f[1]["attack_correlation"] for f in top_features]
        defend_corrs = [f[1]["defend_correlation"] for f in top_features]

        x = np.arange(len(feature_names))
        width = 0.35

        ax3.barh(
            x - width / 2,
            attack_corrs,
            width,
            label="Attack Correlation",
            color="red",
            alpha=0.7,
        )
        ax3.barh(
            x + width / 2,
            defend_corrs,
            width,
            label="Defend Correlation",
            color="blue",
            alpha=0.7,
        )
        ax3.set_yticks(x)
        ax3.set_yticklabels(feature_names, fontsize=10)
        ax3.set_xlabel("Correlation with Predictions")
        ax3.set_title("Feature Correlations with Attack/Defend")
        ax3.legend()
        ax3.grid(axis="x", alpha=0.3)

        # Feature utilization distribution
        all_activations = [f[1]["mean_activation"] for f in ranked_features]
        ax4.hist(all_activations, bins=15, alpha=0.7, color="orange", edgecolor="black")
        ax4.axvline(
            x=0.1, color="r", linestyle="--", alpha=0.7, label="Underutilized Threshold"
        )
        ax4.axvline(
            x=0.3, color="g", linestyle="--", alpha=0.7, label="Well-utilized Threshold"
        )
        ax4.set_xlabel("Mean Activation")
        ax4.set_ylabel("Number of Features")
        ax4.set_title("Oscillation Feature Utilization Distribution")
        ax4.legend()
        ax4.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            plots_dir / "oscillation_feature_importance.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_oscillation_effectiveness_dashboard(self, plots_dir):
        """Create a comprehensive oscillation effectiveness dashboard"""
        effectiveness = self.insights.get("oscillation_effectiveness", {})
        if not effectiveness or effectiveness.get("status") == "no_data":
            return

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # Overall effectiveness gauge
        ax1 = fig.add_subplot(gs[0, :2])
        overall_eff = effectiveness.get("overall_effectiveness", 0)
        effectiveness_level = effectiveness.get("effectiveness_level", "unknown")

        # Create gauge chart
        categories = ["Needs Improvement", "Developing", "Good", "Excellent"]
        values = [0.4, 0.6, 0.8, 1.0]
        colors = ["red", "orange", "yellow", "green"]

        wedges, texts = ax1.pie(
            [0.4, 0.2, 0.2, 0.2],
            colors=colors,
            startangle=180,
            counterclock=False,
            wedgeprops=dict(width=0.3),
        )

        # Add needle
        angle = 180 - (overall_eff * 180)
        ax1.arrow(
            0,
            0,
            0.7 * np.cos(np.radians(angle)),
            0.7 * np.sin(np.radians(angle)),
            head_width=0.05,
            head_length=0.05,
            fc="black",
            ec="black",
        )

        ax1.set_title(
            f"Overall Oscillation Effectiveness: {effectiveness_level.title()}\n"
            f"Score: {overall_eff:.3f}",
            fontsize=14,
            fontweight="bold",
        )

        # Individual effectiveness metrics
        ax2 = fig.add_subplot(gs[0, 2:])
        metrics = [
            "frequency_optimization",
            "spatial_control_mastery",
            "neutral_game_management",
            "whiff_bait_utilization",
        ]
        metric_values = [effectiveness.get(m, 0) for m in metrics]
        metric_labels = [m.replace("_", " ").title() for m in metrics]

        bars = ax2.bar(
            metric_labels,
            metric_values,
            color=["blue", "green", "orange", "purple"],
            alpha=0.7,
        )
        ax2.set_ylabel("Effectiveness Score")
        ax2.set_title("Individual Effectiveness Metrics")
        ax2.set_ylim(0, 1)
        ax2.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Oscillation data trends
        osc_data = self._extract_oscillation_data()
        if osc_data:
            ax3 = fig.add_subplot(gs[1, :2])
            frequencies = [d.get("player_oscillation_frequency", 0) for d in osc_data]
            space_scores = [d.get("space_control_score", 0) for d in osc_data]
            steps = [d.get("step", 0) for d in osc_data]

            if frequencies and space_scores:
                ax3_twin = ax3.twinx()

                line1 = ax3.plot(
                    steps, frequencies, "b-", linewidth=2, label="Oscillation Frequency"
                )
                line2 = ax3_twin.plot(
                    steps, space_scores, "g-", linewidth=2, label="Space Control"
                )

                ax3.set_xlabel("Training Step")
                ax3.set_ylabel("Oscillation Frequency (Hz)", color="b")
                ax3_twin.set_ylabel("Space Control Score", color="g")
                ax3.set_title("Oscillation Metrics Over Training")

                # Combine legends
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax3.legend(lines, labels, loc="upper left")
                ax3.grid(alpha=0.3)

        # Effectiveness improvement recommendations
        ax4 = fig.add_subplot(gs[1, 2:])
        recommendations = self.insights.get("oscillation_recommendations", {})
        if recommendations:
            # Count recommendations by priority
            priority_counts = recommendations.get("priority_summary", {})
            priorities = list(priority_counts.keys())
            counts = list(priority_counts.values())
            colors = {
                "critical": "red",
                "high": "orange",
                "medium": "yellow",
                "low": "green",
            }
            bar_colors = [colors.get(p, "gray") for p in priorities]

            ax4.bar(priorities, counts, color=bar_colors, alpha=0.7)
            ax4.set_ylabel("Number of Recommendations")
            ax4.set_title("Improvement Recommendations by Priority")
            ax4.grid(axis="y", alpha=0.3)

        # Feature category effectiveness
        ax5 = fig.add_subplot(gs[2, :2])
        feature_rankings = self.insights.get("overall_feature_ranking", {})
        if feature_rankings:
            category_rankings = feature_rankings.get("category_rankings", [])
            if category_rankings:
                categories = [c[0] for c in category_rankings]
                avg_importance = [c[1]["avg_importance"] for c in category_rankings]

                bars = ax5.bar(
                    categories,
                    avg_importance,
                    color=["blue", "red", "green"],
                    alpha=0.7,
                )
                ax5.set_ylabel("Average Feature Importance")
                ax5.set_title("Feature Category Importance Ranking")
                ax5.grid(axis="y", alpha=0.3)

        # Oscillation learning progression
        ax6 = fig.add_subplot(gs[2, 2:])
        if osc_data and len(osc_data) > 20:
            # Split into phases and show progression
            phases = self._split_into_phases(osc_data, 5)
            phase_names = ["Early", "Mid-Early", "Middle", "Mid-Late", "Late"]

            # Calculate average effectiveness for each phase
            phase_effectiveness = []
            for phase in phases:
                phase_freq = [d.get("player_oscillation_frequency", 0) for d in phase]
                phase_space = [d.get("space_control_score", 0) for d in phase]

                # Simple effectiveness score
                freq_eff = (
                    len([f for f in phase_freq if 1.0 <= f <= 3.0]) / len(phase_freq)
                    if phase_freq
                    else 0
                )
                space_eff = (
                    len([s for s in phase_space if s > 0]) / len(phase_space)
                    if phase_space
                    else 0
                )
                combined_eff = (freq_eff + space_eff) / 2
                phase_effectiveness.append(combined_eff)

            ax6.plot(
                phase_names,
                phase_effectiveness,
                "o-",
                linewidth=3,
                markersize=8,
                color="purple",
            )
            ax6.set_ylabel("Phase Effectiveness")
            ax6.set_title("Oscillation Learning Progression")
            ax6.grid(alpha=0.3)
            ax6.set_ylim(0, 1)

        plt.suptitle("Oscillation Analysis Dashboard", fontsize=16, fontweight="bold")
        plt.savefig(
            plots_dir / "oscillation_effectiveness_dashboard.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def generate_enhanced_report(self):
        """Generate comprehensive report with oscillation analysis"""
        print("\n📋 GENERATING ENHANCED OSCILLATION REPORT...")

        report = {
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_type": "enhanced_oscillation_analysis",
                "feature_system": "45_features_strategic_oscillation_button",
                "data_sources": {
                    "analysis_files": len(self.data.get("analysis_files", [])),
                    "oscillation_logs": len(self.data.get("oscillation_logs", [])),
                    "training_logs": sum(
                        len(logs) for logs in self.data.get("logs", {}).values()
                    ),
                    "checkpoints": len(self.data.get("checkpoints", [])),
                },
            },
            "executive_summary": self._generate_enhanced_executive_summary(),
            "oscillation_analysis": {
                "frequency_analysis": self.insights.get(
                    "oscillation_frequency_analysis", {}
                ),
                "spatial_control_analysis": self.insights.get(
                    "spatial_control_analysis", {}
                ),
                "neutral_game_analysis": self.insights.get("neutral_game_analysis", {}),
                "whiff_bait_analysis": self.insights.get("whiff_bait_analysis", {}),
                "cross_attention_analysis": self.insights.get(
                    "cross_attention_analysis", {}
                ),
                "oscillation_effectiveness": self.insights.get(
                    "oscillation_effectiveness", {}
                ),
            },
            "feature_analysis": {
                "strategic_features": self.insights.get(
                    "strategic_feature_importance", {}
                ),
                "oscillation_features": self.insights.get(
                    "oscillation_feature_importance", {}
                ),
                "button_features": self.insights.get("button_feature_importance", {}),
                "overall_ranking": self.insights.get("overall_feature_ranking", {}),
            },
            "recommendations": {
                "oscillation_specific": self.insights.get(
                    "oscillation_recommendations", {}
                ),
                "general_improvements": self.insights.get("recommendations", {}),
            },
            "visualizations_created": [
                "oscillation_frequency_evolution.png",
                "spatial_control_analysis.png",
                "neutral_game_patterns.png",
                "cross_attention_weights.png",
                "oscillation_feature_importance.png",
                "oscillation_effectiveness_dashboard.png",
            ],
        }

        # Save comprehensive report
        report_path = self.output_dir / "enhanced_oscillation_analysis_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Generate enhanced markdown summary
        self._generate_enhanced_markdown_summary(report)

        print(f"✅ Enhanced oscillation report saved to {report_path}")
        return report

    def _generate_enhanced_executive_summary(self):
        """Generate executive summary with oscillation focus"""
        summary = {
            "overall_status": "unknown",
            "oscillation_effectiveness": "unknown",
            "key_oscillation_findings": [],
            "critical_oscillation_issues": [],
            "top_oscillation_recommendations": [],
            "feature_category_ranking": [],
        }

        # Determine oscillation effectiveness
        osc_effectiveness = self.insights.get("oscillation_effectiveness", {})
        if osc_effectiveness:
            summary["oscillation_effectiveness"] = osc_effectiveness.get(
                "effectiveness_level", "unknown"
            )
            overall_eff = osc_effectiveness.get("overall_effectiveness", 0)

            if overall_eff >= 0.8:
                summary["overall_status"] = "excellent_oscillation"
            elif overall_eff >= 0.6:
                summary["overall_status"] = "good_oscillation"
            elif overall_eff >= 0.4:
                summary["overall_status"] = "developing_oscillation"
            else:
                summary["overall_status"] = "needs_oscillation_improvement"

        # Key oscillation findings
        freq_analysis = self.insights.get("oscillation_frequency_analysis", {})
        if freq_analysis and freq_analysis.get("status") != "no_data":
            mean_freq = freq_analysis.get("mean_frequency", 0)
            summary["key_oscillation_findings"].append(
                f"Average oscillation frequency: {mean_freq:.2f} Hz"
            )

        spatial_analysis = self.insights.get("spatial_control_analysis", {})
        if spatial_analysis and spatial_analysis.get("status") != "no_spatial_data":
            mean_control = spatial_analysis.get("mean_space_control", 0)
            positive_ratio = spatial_analysis.get("positive_control_ratio", 0)
            summary["key_oscillation_findings"].append(
                f"Spatial control score: {mean_control:.3f} (positive control: {positive_ratio:.1%})"
            )

        neutral_analysis = self.insights.get("neutral_game_analysis", {})
        if neutral_analysis and neutral_analysis.get("status") != "no_neutral_data":
            avg_duration = neutral_analysis.get("avg_neutral_duration", 0)
            summary["key_oscillation_findings"].append(
                f"Average neutral game duration: {avg_duration:.1f} frames"
            )

        # Feature category ranking
        feature_rankings = self.insights.get("overall_feature_ranking", {})
        if feature_rankings:
            category_rankings = feature_rankings.get("category_rankings", [])
            summary["feature_category_ranking"] = [
                f"{cat[0].title()}: {cat[1]['avg_importance']:.3f}"
                for cat in category_rankings
            ]

        # Critical issues and recommendations
        osc_recommendations = self.insights.get("oscillation_recommendations", {})
        if osc_recommendations:
            critical_issues = []
            top_recommendations = []

            for category in osc_recommendations:
                if isinstance(osc_recommendations[category], list):
                    for rec in osc_recommendations[category][:3]:  # Top 3 per category
                        if rec.get("priority") in ["critical", "high"]:
                            issue = rec.get("issue", rec.get("suggestion", ""))
                            if issue:
                                critical_issues.append(f"[{category}] {issue}")

                            suggestion = rec.get("suggestion", "")
                            if suggestion:
                                top_recommendations.append(f"[{category}] {suggestion}")

            summary["critical_oscillation_issues"] = critical_issues[:5]
            summary["top_oscillation_recommendations"] = top_recommendations[:5]

        return summary

    def _generate_enhanced_markdown_summary(self, report):
        """Generate enhanced markdown summary with oscillation focus"""

        markdown_content = f"""# Enhanced Street Fighter II Oscillation Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

**Overall Status:** {report['executive_summary']['overall_status'].replace('_', ' ').title()}
**Oscillation Effectiveness:** {report['executive_summary']['oscillation_effectiveness'].replace('_', ' ').title()}

### Key Oscillation Findings
"""

        for finding in report["executive_summary"]["key_oscillation_findings"]:
            markdown_content += f"- {finding}\n"

        markdown_content += "\n### Feature Category Ranking\n"
        for ranking in report["executive_summary"]["feature_category_ranking"]:
            markdown_content += f"- {ranking}\n"

        markdown_content += "\n### Critical Oscillation Issues\n"
        for issue in report["executive_summary"]["critical_oscillation_issues"]:
            markdown_content += f"- ❌ {issue}\n"

        markdown_content += "\n### Top Oscillation Recommendations\n"
        for rec in report["executive_summary"]["top_oscillation_recommendations"]:
            markdown_content += f"- 💡 {rec}\n"

        # Detailed oscillation analysis
        markdown_content += "\n## Detailed Oscillation Analysis\n\n"

        # Frequency analysis
        freq_analysis = report["oscillation_analysis"]["frequency_analysis"]
        if freq_analysis and freq_analysis.get("status") != "no_data":
            markdown_content += "### Oscillation Frequency Analysis\n"
            markdown_content += f"- **Mean Frequency:** {freq_analysis.get('mean_frequency', 0):.2f} Hz\n"
            markdown_content += f"- **Frequency Range:** {freq_analysis.get('min_frequency', 0):.2f} - {freq_analysis.get('max_frequency', 0):.2f} Hz\n"
            markdown_content += f"- **Frequency Trend:** {freq_analysis.get('frequency_trend', 0):+.3f}\n"

            freq_perf_corr = freq_analysis.get("frequency_performance_correlation", 0)
            if freq_perf_corr:
                markdown_content += (
                    f"- **Performance Correlation:** {freq_perf_corr:+.3f}\n"
                )
            markdown_content += "\n"

        # Spatial control analysis
        spatial_analysis = report["oscillation_analysis"]["spatial_control_analysis"]
        if spatial_analysis and spatial_analysis.get("status") != "no_spatial_data":
            markdown_content += "### Spatial Control Analysis\n"
            markdown_content += f"- **Mean Space Control:** {spatial_analysis.get('mean_space_control', 0):+.3f}\n"
            markdown_content += f"- **Positive Control Ratio:** {spatial_analysis.get('positive_control_ratio', 0):.1%}\n"
            markdown_content += f"- **Control Trend:** {spatial_analysis.get('space_control_trend', 0):+.3f}\n"
            markdown_content += f"- **Dominant Control Instances:** {spatial_analysis.get('dominant_control_instances', 0)}\n"

            phase_analysis = spatial_analysis.get("phase_analysis", {})
            if phase_analysis:
                markdown_content += f"- **Phase Improvement:** {phase_analysis.get('phase_improvement', 0):+.3f}\n"
            markdown_content += "\n"

        # Neutral game analysis
        neutral_analysis = report["oscillation_analysis"]["neutral_game_analysis"]
        if neutral_analysis and neutral_analysis.get("status") != "no_neutral_data":
            markdown_content += "### Neutral Game Analysis\n"
            markdown_content += f"- **Average Duration:** {neutral_analysis.get('avg_neutral_duration', 0):.1f} frames\n"
            markdown_content += f"- **Neutral Consistency:** {neutral_analysis.get('neutral_consistency', 0):.3f}\n"
            markdown_content += f"- **Short Neutral Ratio:** {neutral_analysis.get('short_neutral_ratio', 0):.1%}\n"
            markdown_content += f"- **Long Neutral Ratio:** {neutral_analysis.get('long_neutral_ratio', 0):.1%}\n"
            markdown_content += "\n"

        # Cross-attention analysis
        attention_analysis = report["oscillation_analysis"]["cross_attention_analysis"]
        if (
            attention_analysis
            and attention_analysis.get("status") != "no_attention_data"
        ):
            markdown_content += "### Cross-Attention Analysis\n"
            final_balance = attention_analysis.get("final_attention_balance", {})
            if final_balance:
                markdown_content += "**Final Attention Distribution:**\n"
                for attention_type, weight in final_balance.items():
                    markdown_content += (
                        f"- {attention_type.replace('_', ' ').title()}: {weight:.1%}\n"
                    )

            osc_focus = attention_analysis.get("oscillation_focus_adequacy", {})
            if osc_focus:
                markdown_content += f"- **Oscillation Focus Adequacy:** {osc_focus.get('status', 'unknown')}\n"
            markdown_content += "\n"

        # Feature analysis summary
        markdown_content += "## Feature Analysis Summary\n\n"

        feature_analysis = report["feature_analysis"]
        overall_ranking = feature_analysis.get("overall_ranking", {})
        if overall_ranking:
            top_features = overall_ranking.get("top_10_features", [])
            if top_features:
                markdown_content += "### Top 10 Most Important Features\n"
                for i, (feature_name, data) in enumerate(top_features, 1):
                    importance = data["importance_score"]
                    category = data["category"]
                    markdown_content += (
                        f"{i}. **{feature_name}** ({category}): {importance:.3f}\n"
                    )
                markdown_content += "\n"

        # Oscillation-specific recommendations
        markdown_content += "## Oscillation-Specific Recommendations\n\n"

        osc_recommendations = report["recommendations"]["oscillation_specific"]

        # High priority oscillation improvements
        high_priority_recs = []
        for category in osc_recommendations:
            if isinstance(osc_recommendations[category], list):
                for rec in osc_recommendations[category]:
                    if rec.get("priority") in ["critical", "high"]:
                        high_priority_recs.append((category, rec))

        if high_priority_recs:
            markdown_content += "### High Priority Oscillation Improvements\n\n"
            for category, rec in high_priority_recs[:8]:  # Top 8
                priority_emoji = {"critical": "🔴", "high": "🟠"}.get(
                    rec.get("priority", "high"), "🟡"
                )
                cat_name = category.replace("_", " ").title()
                suggestion = rec.get("suggestion", "Unknown")
                markdown_content += f"#### {priority_emoji} [{cat_name}] {suggestion}\n"

                if "issue" in rec:
                    markdown_content += f"**Issue:** {rec['issue']}\n\n"

                if "implementation" in rec:
                    impl = rec["implementation"]
                    if isinstance(impl, list):
                        markdown_content += "**Implementation:**\n"
                        for item in impl:
                            markdown_content += f"- {item}\n"
                    else:
                        markdown_content += f"**Implementation:** {impl}\n"
                    markdown_content += "\n"

        # Feature engineering recommendations
        feature_recs = osc_recommendations.get("feature_engineering_improvements", [])
        new_feature_recs = [
            r for r in feature_recs if r.get("type") == "new_oscillation_feature"
        ]

        if new_feature_recs:
            markdown_content += "### Suggested New Oscillation Features\n\n"
            for i, rec in enumerate(new_feature_recs, 1):
                suggestion = rec.get("suggestion", "Unknown")
                rationale = rec.get("rationale", "N/A")
                markdown_content += f"#### {i}. {suggestion}\n"
                markdown_content += f"**Rationale:** {rationale}\n"

                if "implementation" in rec:
                    impl = rec["implementation"]
                    if isinstance(impl, list):
                        markdown_content += "**Implementation:**\n"
                        for item in impl:
                            markdown_content += f"- {item}\n"
                    else:
                        markdown_content += f"**Implementation:** {impl}\n"
                markdown_content += "\n"

        # Training improvements
        training_recs = osc_recommendations.get("training_improvements", [])
        if training_recs:
            markdown_content += "### Training Improvements\n\n"
            for rec in training_recs[:3]:  # Top 3
                suggestion = rec.get("suggestion", "Unknown")
                markdown_content += f"#### {suggestion}\n"

                if "implementation" in rec:
                    impl = rec["implementation"]
                    if isinstance(impl, list):
                        markdown_content += "**Implementation:**\n"
                        for item in impl:
                            markdown_content += f"- {item}\n"
                    else:
                        markdown_content += f"**Implementation:** {impl}\n"
                markdown_content += "\n"

        # Data sources and visualizations
        markdown_content += f"""
## Data Sources Analyzed

- Analysis Files: {report['metadata']['data_sources']['analysis_files']}
- Oscillation Logs: {report['metadata']['data_sources']['oscillation_logs']}
- Training Log Entries: {report['metadata']['data_sources']['training_logs']}
- Model Checkpoints: {report['metadata']['data_sources']['checkpoints']}

## Oscillation Visualizations Generated

"""
        for viz in report["visualizations_created"]:
            markdown_content += f"- 📊 `oscillation_plots/{viz}`\n"

        markdown_content += f"""
## Implementation Roadmap

### Phase 1: Critical Issues (Immediate)
- Address critical oscillation data collection issues
- Fix any broken oscillation tracking mechanisms
- Implement essential oscillation feature improvements

### Phase 2: High Priority Improvements (1-2 weeks)
- Enhance oscillation frequency detection and optimization
- Improve spatial control calculation methods
- Implement oscillation-specific training rewards
- Add new oscillation features for pattern recognition

### Phase 3: Medium Priority Enhancements (2-4 weeks)
- Optimize cross-attention balance for oscillation features
- Implement oscillation prediction capabilities
- Add advanced neutral game analysis features
- Create oscillation effectiveness benchmarks

### Phase 4: Long-term Optimizations (1-2 months)
- Implement multi-scale oscillation analysis
- Add opponent oscillation modeling
- Create oscillation-based curriculum learning
- Develop oscillation ROI analysis systems

## Monitoring and Validation

### Key Metrics to Track
1. **Oscillation Frequency:** Target range 1.0-3.0 Hz
2. **Spatial Control Score:** Target positive trend with >40% positive instances
3. **Neutral Game Duration:** Target 30-90 frames average
4. **Cross-Attention Balance:** Oscillation attention 15-60% of total
5. **Feature Utilization:** All oscillation features >0.1 activation

### Validation Tests
- Oscillation pattern recognition accuracy
- Spatial control effectiveness correlation
- Neutral game management consistency
- Cross-attention stability over training
- Feature importance progression

---
*Enhanced Oscillation Analysis Report generated by Street Fighter II Analytics System*
"""

        # Save enhanced markdown report
        markdown_path = self.output_dir / "enhanced_oscillation_analysis_summary.md"
        with open(markdown_path, "w") as f:
            f.write(markdown_content)

        print(f"✅ Enhanced markdown summary saved to {markdown_path}")

    # Helper methods for oscillation analysis
    def _find_win_rate_for_step(self, step):
        """Find win rate data for a specific training step"""
        for log_entry in self.data.get("logs", {}).get("training_stats", []):
            if log_entry.get("step") == step and "win_rate" in log_entry:
                return log_entry["win_rate"]
        return None

    def _determine_optimal_frequency_range(self, frequencies, oscillation_data):
        """Determine optimal oscillation frequency range based on performance"""
        if not frequencies or len(frequencies) < 10:
            return {"min": 1.0, "max": 3.0, "confidence": "low"}

        # Group by frequency ranges and analyze performance
        freq_ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 10)]
        range_performance = {}

        for range_min, range_max in freq_ranges:
            range_data = [
                d
                for d, f in zip(oscillation_data, frequencies)
                if range_min <= f < range_max
            ]

            if range_data:
                # Calculate average space control for this range
                space_scores = [d.get("space_control_score", 0) for d in range_data]
                avg_performance = np.mean(space_scores) if space_scores else 0
                range_performance[f"{range_min}-{range_max}"] = {
                    "avg_performance": avg_performance,
                    "sample_size": len(range_data),
                }

        # Find best performing range
        best_range = max(
            range_performance.items(), key=lambda x: x[1]["avg_performance"]
        )

        return {
            "optimal_range": best_range[0],
            "performance_score": best_range[1]["avg_performance"],
            "confidence": "high" if best_range[1]["sample_size"] > 20 else "medium",
        }

    def _determine_optimal_neutral_range(self, neutral_durations, oscillation_data):
        """Determine optimal neutral game duration range"""
        if not neutral_durations or len(neutral_durations) < 10:
            return {"min": 30, "max": 90, "confidence": "low"}

        # Group by duration ranges
        duration_ranges = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 300)]
        range_performance = {}

        for range_min, range_max in duration_ranges:
            range_data = [
                d
                for d, dur in zip(oscillation_data, neutral_durations)
                if range_min <= dur < range_max
            ]

            if range_data:
                space_scores = [d.get("space_control_score", 0) for d in range_data]
                avg_performance = np.mean(space_scores) if space_scores else 0
                range_performance[f"{range_min}-{range_max}"] = {
                    "avg_performance": avg_performance,
                    "sample_size": len(range_data),
                }

        best_range = max(
            range_performance.items(), key=lambda x: x[1]["avg_performance"]
        )

        return {
            "optimal_range": best_range[0],
            "performance_score": best_range[1]["avg_performance"],
            "confidence": "high" if best_range[1]["sample_size"] > 15 else "medium",
        }

    def _assess_oscillation_focus(self, oscillation_weight):
        """Assess if oscillation attention weight is adequate"""
        if oscillation_weight < 0.15:
            return {
                "status": "insufficient",
                "recommendation": "increase_oscillation_focus",
            }
        elif oscillation_weight > 0.6:
            return {
                "status": "excessive",
                "recommendation": "balance_attention_weights",
            }
        else:
            return {"status": "adequate", "recommendation": "maintain_current_balance"}

    def _calculate_consistency(self, values):
        """Calculate consistency score (inverse of coefficient of variation)"""
        if not values or len(values) < 2:
            return 0
        mean_val = np.mean(values)
        if mean_val == 0:
            return 0
        cv = np.std(values) / mean_val
        return 1 / (1 + cv)

    def _calculate_learning_rate(self, values):
        """Calculate learning rate (improvement per unit time)"""
        if not values or len(values) < 2:
            return 0
        return (values[-1] - values[0]) / len(values)

    def _calculate_stability(self, values):
        """Calculate stability score"""
        if not values or len(values) < 2:
            return 0
        return 1 / (1 + np.std(values))

    def _split_into_phases(self, data, num_phases):
        """Split data into equal phases"""
        if len(data) < num_phases:
            return [data]

        phase_size = len(data) // num_phases
        phases = []
        for i in range(num_phases):
            start_idx = i * phase_size
            end_idx = (i + 1) * phase_size if i < num_phases - 1 else len(data)
            phases.append(data[start_idx:end_idx])

        return phases

    def _extract_prediction_data(self):
        """Extract prediction data from analysis files"""
        all_predictions = []

        for analysis_data in self.data.get("analysis_files", []):
            if "recent_predictions" in analysis_data:
                predictions = analysis_data["recent_predictions"]
                for pred in predictions:
                    pred["source"] = "analysis_file"
                all_predictions.extend(predictions)

            if "predictions" in analysis_data:
                predictions = analysis_data["predictions"]
                for pred in predictions:
                    pred["source"] = "analysis_file"
                all_predictions.extend(predictions)

        all_predictions.sort(key=lambda x: x.get("step", 0))
        return all_predictions

    def _calculate_trend(self, values):
        """Calculate trend direction (-1 to 1)"""
        if len(values) < 2:
            return 0
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        return np.clip(z[0], -1, 1)

    def _calculate_improvement_rate(self, values):
        """Calculate rate of improvement"""
        if len(values) < 2:
            return 0
        return (values[-1] - values[0]) / len(values)


def main():
    """Main enhanced analysis execution"""
    parser = argparse.ArgumentParser(
        description="Enhanced Street Fighter II Oscillation Analytics"
    )
    parser.add_argument(
        "--analysis-dir",
        default="analysis_data",
        help="Directory containing analysis files",
    )
    parser.add_argument(
        "--logs-dir", default="logs", help="Directory containing log files"
    )
    parser.add_argument(
        "--output-dir", default="analytics_output", help="Output directory for results"
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument(
        "--oscillation-focus",
        action="store_true",
        help="Focus analysis on oscillation features",
    )

    args = parser.parse_args()

    print("🌊 ENHANCED STREET FIGHTER II OSCILLATION ANALYTICS")
    print("=" * 60)

    # Initialize enhanced analytics system
    analytics = EnhancedStreetFighterAnalytics(
        analysis_dir=args.analysis_dir, logs_dir=args.logs_dir
    )
    analytics.output_dir = Path(args.output_dir)
    analytics.output_dir.mkdir(exist_ok=True)

    try:
        # Load all available data
        analytics.load_all_data()

        # Perform oscillation-focused analysis
        analytics.analyze_oscillation_learning()
        analytics.analyze_enhanced_feature_importance()
        analytics.generate_oscillation_recommendations()

        # Create visualizations (unless disabled)
        if not args.no_plots:
            analytics.create_oscillation_visualizations()

        # Generate comprehensive report
        report = analytics.generate_enhanced_report()

        # Print enhanced summary to console
        print("\n" + "=" * 60)
        print("🌊 ENHANCED OSCILLATION ANALYSIS COMPLETE")
        print("=" * 60)

        exec_summary = report["executive_summary"]
        print(
            f"📊 Overall Status: {exec_summary['overall_status'].replace('_', ' ').title()}"
        )
        print(
            f"🌊 Oscillation Effectiveness: {exec_summary['oscillation_effectiveness'].replace('_', ' ').title()}"
        )

        if exec_summary["key_oscillation_findings"]:
            print("\n🔍 Key Oscillation Findings:")
            for finding in exec_summary["key_oscillation_findings"]:
                print(f"  • {finding}")

        if exec_summary["feature_category_ranking"]:
            print("\n🏆 Feature Category Ranking:")
            for ranking in exec_summary["feature_category_ranking"]:
                print(f"  • {ranking}")

        if exec_summary["critical_oscillation_issues"]:
            print("\n❌ Critical Oscillation Issues:")
            for issue in exec_summary["critical_oscillation_issues"][:3]:
                print(f"  • {issue}")

        if exec_summary["top_oscillation_recommendations"]:
            print("\n💡 Top Oscillation Recommendations:")
            for i, rec in enumerate(
                exec_summary["top_oscillation_recommendations"][:3], 1
            ):
                print(f"  {i}. {rec}")

        # Print specific oscillation insights
        print("\n" + "=" * 60)
        print("🌊 DETAILED OSCILLATION INSIGHTS")
        print("=" * 60)

        osc_analysis = report["oscillation_analysis"]

        # Frequency analysis
        freq_analysis = osc_analysis["frequency_analysis"]
        if freq_analysis and freq_analysis.get("status") != "no_data":
            print(f"\n📊 Oscillation Frequency Analysis:")
            print(
                f"  • Mean Frequency: {freq_analysis.get('mean_frequency', 0):.2f} Hz"
            )
            print(
                f"  • Frequency Range: {freq_analysis.get('min_frequency', 0):.2f} - {freq_analysis.get('max_frequency', 0):.2f} Hz"
            )
            print(
                f"  • Frequency Trend: {freq_analysis.get('frequency_trend', 0):+.3f}"
            )

            optimal_range = freq_analysis.get("optimal_frequency_range", {})
            if optimal_range:
                print(
                    f"  • Optimal Range: {optimal_range.get('optimal_range', 'N/A')} Hz"
                )

        # Spatial control analysis
        spatial_analysis = osc_analysis["spatial_control_analysis"]
        if spatial_analysis and spatial_analysis.get("status") != "no_spatial_data":
            print(f"\n🎯 Spatial Control Analysis:")
            print(
                f"  • Mean Space Control: {spatial_analysis.get('mean_space_control', 0):+.3f}"
            )
            print(
                f"  • Positive Control Ratio: {spatial_analysis.get('positive_control_ratio', 0):.1%}"
            )
            print(
                f"  • Control Trend: {spatial_analysis.get('space_control_trend', 0):+.3f}"
            )

        # Effectiveness assessment
        effectiveness = osc_analysis["oscillation_effectiveness"]
        if effectiveness and effectiveness.get("status") != "no_data":
            print(f"\n📈 Oscillation Effectiveness:")
            print(
                f"  • Overall Effectiveness: {effectiveness.get('overall_effectiveness', 0):.3f}"
            )
            print(
                f"  • Effectiveness Level: {effectiveness.get('effectiveness_level', 'unknown').title()}"
            )

            metrics = [
                "frequency_optimization",
                "spatial_control_mastery",
                "neutral_game_management",
                "whiff_bait_utilization",
            ]
            for metric in metrics:
                if metric in effectiveness:
                    print(
                        f"  • {metric.replace('_', ' ').title()}: {effectiveness[metric]:.3f}"
                    )

        # Print file locations
        print(f"\n📁 Output Files:")
        print(
            f"  • Comprehensive Report: {analytics.output_dir}/enhanced_oscillation_analysis_report.json"
        )
        print(
            f"  • Summary Report: {analytics.output_dir}/enhanced_oscillation_analysis_summary.md"
        )
        if not args.no_plots:
            print(f"  • Visualizations: {analytics.output_dir}/oscillation_plots/")

        print(f"\n✅ Enhanced oscillation analysis complete!")
        print(f"🎯 Focus on high-priority oscillation improvements for best results.")

    except KeyboardInterrupt:
        print("\n⏹️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
