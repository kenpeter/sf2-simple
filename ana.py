#!/usr/bin/env python3
"""
Street Fighter II Transformer Analytics System
Comprehensive analysis of transformer learning patterns, performance metrics,
and actionable recommendations for improvement.

Based on the wrapper.py strategic features and training logs.
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

warnings.filterwarnings("ignore")

# Set up plotting style
plt.style.use("default")
sns.set_palette("husl")


class StreetFighterAnalytics:
    """
    Comprehensive analytics system for Street Fighter II transformer learning
    """

    def __init__(self, analysis_dir="analysis_data", logs_dir="logs"):
        self.analysis_dir = Path(analysis_dir)
        self.logs_dir = Path(logs_dir)
        self.output_dir = Path("analytics_output")
        self.output_dir.mkdir(exist_ok=True)

        # Strategic feature names from wrapper.py (33 total: 21 strategic + 12 button)
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
            "position_stability",  # 12
            "optimal_spacing",  # 13
            "forward_pressure",  # 14
            "defensive_movement",  # 15
            "close_combat_frequency",  # 16
            "enhanced_score_momentum",  # 17 - Key combo feature
            "status_difference",  # 18
            "agent_victories",  # 19
            "enemy_victories",  # 20
        ]

        # Button feature names from wrapper.py (12 buttons - PREVIOUS actions)
        self.button_feature_names = [
            "prev_B_pressed",  # 21 - Light Kick
            "prev_Y_pressed",  # 22 - Light Punch
            "prev_SELECT_pressed",  # 23
            "prev_START_pressed",  # 24
            "prev_UP_pressed",  # 25
            "prev_DOWN_pressed",  # 26
            "prev_LEFT_pressed",  # 27
            "prev_RIGHT_pressed",  # 28
            "prev_A_pressed",  # 29 - Medium Kick
            "prev_X_pressed",  # 30 - Medium Punch
            "prev_L_pressed",  # 31 - Heavy Punch
            "prev_R_pressed",  # 32 - Heavy Kick
        ]

        # Combined feature names (33 total)
        self.all_feature_names = (
            self.strategic_feature_names + self.button_feature_names
        )

        # Action categories from wrapper.py
        self.action_categories = {
            "movement": list(range(1, 9)),  # Basic movements
            "light_attacks": list(range(9, 15)),  # Light attacks
            "medium_attacks": list(range(15, 21)),  # Medium attacks
            "heavy_attacks": list(range(21, 27)),  # Heavy attacks
            "jumping_attacks": list(range(27, 33)),  # Jumping attacks
            "crouching_attacks": list(range(33, 39)),  # Crouching attacks
            "special_moves": list(range(39, 51)),  # Special move motions
            "defensive": list(range(101, 107)),  # Defensive options
            "combo_starters": list(range(107, 115)),  # Combo starters
        }

        self.data = {}
        self.insights = {}

    def load_all_data(self):
        """Load all available analysis and log data"""
        print("📁 Loading all data files...")

        # Load JSON analysis files
        self._load_analysis_files()

        # Load log files
        self._load_log_files()

        # Load any additional data
        self._load_checkpoint_data()

        print(f"✅ Data loading complete. Found:")
        for data_type, data in self.data.items():
            if isinstance(data, list):
                print(f"   {data_type}: {len(data)} entries")
            elif isinstance(data, dict):
                print(f"   {data_type}: {len(data)} files")
            else:
                print(f"   {data_type}: loaded")

    def _load_analysis_files(self):
        """Load transformer analysis JSON files"""
        analysis_files = []

        # Look for various analysis file patterns
        patterns = [
            "final_enhanced_analysis.json",
            "enhanced_transformer_analysis_*.json",
            "transformer_analysis_step_*.json",
            "transformer_analysis_win_*.json",
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
        """Load various log files from training"""
        self.data["logs"] = {}

        # Training stats logs
        training_logs = list(self.logs_dir.glob("training_stats_*.log"))
        self.data["logs"]["training_stats"] = self._parse_log_files(training_logs)

        # Performance logs
        performance_logs = list(self.logs_dir.glob("performance_*.log"))
        self.data["logs"]["performance"] = self._parse_log_files(performance_logs)

        # Milestone logs
        milestone_logs = list(self.logs_dir.glob("milestones_*.log"))
        self.data["logs"]["milestones"] = self._parse_log_files(milestone_logs)

        # Learning rate logs
        lr_logs = list(self.logs_dir.glob("learning_rate_*.log"))
        self.data["logs"]["learning_rate"] = self._parse_log_files(lr_logs)

        print(f"   ✓ Loaded {len(training_logs)} training log files")
        print(f"   ✓ Loaded {len(performance_logs)} performance log files")
        print(f"   ✓ Loaded {len(milestone_logs)} milestone log files")
        print(f"   ✓ Loaded {len(lr_logs)} learning rate log files")

    def _parse_log_files(self, log_files):
        """Parse log files into structured data"""
        parsed_data = []

        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        if " - " in line:
                            timestamp_str, content = line.strip().split(" - ", 1)
                            try:
                                timestamp = datetime.fromisoformat(timestamp_str)
                                parsed_data.append(
                                    {
                                        "timestamp": timestamp,
                                        "content": content,
                                        "source_file": log_file.name,
                                    }
                                )
                            except:
                                # If timestamp parsing fails, still capture content
                                parsed_data.append(
                                    {
                                        "timestamp": None,
                                        "content": line.strip(),
                                        "source_file": log_file.name,
                                    }
                                )
            except Exception as e:
                print(f"   ✗ Error parsing {log_file}: {e}")

        return parsed_data

    def _load_checkpoint_data(self):
        """Load any checkpoint or model data if available"""
        # Look for model files and extract metadata
        self.data["checkpoints"] = []

        model_dirs = ["enhanced_trained_models", "trained_models"]
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
                        }
                    )

    def analyze_transformer_learning(self):
        """Comprehensive analysis of what the transformer learned"""
        print("\n🧠 ANALYZING TRANSFORMER LEARNING PATTERNS...")

        # Extract prediction data
        predictions = self._extract_prediction_data()

        if not predictions:
            print("❌ No prediction data found!")
            return

        # Analyze different aspects
        self.insights["feature_importance"] = self._analyze_feature_importance(
            predictions
        )
        self.insights["temporal_patterns"] = self._analyze_temporal_patterns(
            predictions
        )
        self.insights["combo_learning"] = self._analyze_combo_learning(predictions)
        self.insights["strategic_evolution"] = self._analyze_strategic_evolution(
            predictions
        )
        self.insights["action_correlation"] = self._analyze_action_correlation(
            predictions
        )

        print("✅ Transformer learning analysis complete")

    def _extract_prediction_data(self):
        """Extract and combine prediction data from all sources"""
        all_predictions = []

        # From analysis files
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

        # Sort by step if available
        all_predictions.sort(key=lambda x: x.get("step", 0))

        return all_predictions

    def _analyze_feature_importance(self, predictions):
        """Analyze which features the transformer learned to value most"""
        print("   🔍 Analyzing feature importance...")

        if not predictions:
            return {}

        # Collect strategic features across all predictions
        feature_values = defaultdict(list)
        feature_correlations = defaultdict(list)

        for pred in predictions:
            if "strategic_features" in pred:
                features = pred["strategic_features"]
                attack_timing = pred.get("attack_timing", 0)
                defend_timing = pred.get("defend_timing", 0)

                # Ensure we have the right number of features (33)
                if len(features) != 33:
                    continue

                for i, feature_value in enumerate(features):
                    if i < len(self.all_feature_names):
                        feature_name = self.all_feature_names[i]
                        feature_values[feature_name].append(feature_value)

                        # Correlation with predictions
                        feature_correlations[f"{feature_name}_attack_corr"].append(
                            feature_value * attack_timing
                        )
                        feature_correlations[f"{feature_name}_defend_corr"].append(
                            feature_value * defend_timing
                        )

        # Calculate importance metrics
        importance_metrics = {}

        for feature_name in self.all_feature_names:
            if feature_name in feature_values:
                values = np.array(feature_values[feature_name])

                importance_metrics[feature_name] = {
                    "mean_activation": np.mean(np.abs(values)),
                    "variance": np.var(values),
                    "max_activation": np.max(np.abs(values)),
                    "activation_frequency": np.mean(np.abs(values) > 0.1),
                    "attack_correlation": np.mean(
                        feature_correlations.get(f"{feature_name}_attack_corr", [0])
                    ),
                    "defend_correlation": np.mean(
                        feature_correlations.get(f"{feature_name}_defend_corr", [0])
                    ),
                }

        # Rank features by importance
        ranked_features = sorted(
            importance_metrics.items(),
            key=lambda x: x[1]["mean_activation"] * x[1]["variance"],
            reverse=True,
        )

        return {
            "feature_metrics": importance_metrics,
            "ranked_features": ranked_features[:10],  # Top 10
            "total_features_analyzed": len(importance_metrics),
        }

    def _analyze_temporal_patterns(self, predictions):
        """Analyze how transformer predictions evolved over time"""
        print("   📈 Analyzing temporal learning patterns...")

        if len(predictions) < 100:
            return {}

        # Group predictions by training progress
        sorted_predictions = sorted(predictions, key=lambda x: x.get("step", 0))

        # Split into phases
        total_steps = len(sorted_predictions)
        phase_size = total_steps // 5  # 5 phases

        phases = {
            "early": sorted_predictions[:phase_size],
            "mid_early": sorted_predictions[phase_size : 2 * phase_size],
            "middle": sorted_predictions[2 * phase_size : 3 * phase_size],
            "mid_late": sorted_predictions[3 * phase_size : 4 * phase_size],
            "late": sorted_predictions[4 * phase_size :],
        }

        phase_analysis = {}

        for phase_name, phase_predictions in phases.items():
            if not phase_predictions:
                continue

            attack_timings = [p.get("attack_timing", 0) for p in phase_predictions]
            defend_timings = [p.get("defend_timing", 0) for p in phase_predictions]

            phase_analysis[phase_name] = {
                "avg_attack_confidence": np.mean(attack_timings),
                "avg_defend_confidence": np.mean(defend_timings),
                "attack_variance": np.var(attack_timings),
                "defend_variance": np.var(defend_timings),
                "prediction_count": len(phase_predictions),
                "high_confidence_predictions": len(
                    [
                        p
                        for p in phase_predictions
                        if p.get("attack_timing", 0) > 0.8
                        or p.get("defend_timing", 0) > 0.8
                    ]
                ),
            }

        return {
            "phase_analysis": phase_analysis,
            "learning_trend": self._calculate_learning_trend(sorted_predictions),
            "confidence_evolution": self._analyze_confidence_evolution(
                sorted_predictions
            ),
        }

    def _analyze_combo_learning(self, predictions):
        """Analyze how well the transformer learned combo patterns"""
        print("   🔥 Analyzing combo learning patterns...")

        combo_predictions = []

        # Extract combo-related predictions
        for pred in predictions:
            combo_context = pred.get("combo_context", {})
            current_combo = combo_context.get("current_combo", 0)

            if current_combo > 0:  # Active combo
                combo_predictions.append(
                    {
                        "combo_count": current_combo,
                        "attack_timing": pred.get("attack_timing", 0),
                        "defend_timing": pred.get("defend_timing", 0),
                        "score_momentum": (
                            pred.get("strategic_features", [0] * 18)[17]
                            if len(pred.get("strategic_features", [])) > 17
                            else 0
                        ),
                        "step": pred.get("step", 0),
                    }
                )

        if not combo_predictions:
            return {"status": "no_combo_data"}

        # Analyze combo patterns
        combo_stats = {
            "total_combo_predictions": len(combo_predictions),
            "max_combo_detected": max([p["combo_count"] for p in combo_predictions]),
            "avg_combo_length": np.mean([p["combo_count"] for p in combo_predictions]),
            "combo_attack_confidence": np.mean(
                [p["attack_timing"] for p in combo_predictions]
            ),
            "combo_defend_confidence": np.mean(
                [p["defend_timing"] for p in combo_predictions]
            ),
        }

        # Combo length distribution
        combo_lengths = [p["combo_count"] for p in combo_predictions]
        combo_distribution = Counter(combo_lengths)

        # Attack confidence by combo length
        confidence_by_length = defaultdict(list)
        for pred in combo_predictions:
            confidence_by_length[pred["combo_count"]].append(pred["attack_timing"])

        confidence_analysis = {
            length: np.mean(confidences)
            for length, confidences in confidence_by_length.items()
        }

        return {
            "combo_stats": combo_stats,
            "combo_distribution": dict(combo_distribution),
            "confidence_by_combo_length": confidence_analysis,
            "combo_learning_effectiveness": self._assess_combo_learning_effectiveness(
                combo_predictions
            ),
        }

    def _analyze_strategic_evolution(self, predictions):
        """Analyze how strategic understanding evolved"""
        print("   🎯 Analyzing strategic evolution...")

        if len(predictions) < 50:
            return {}

        # Key strategic features to track
        key_features = [
            "enhanced_score_momentum",  # Index 17
            "optimal_spacing",  # Index 13
            "center_control",  # Index 10
            "forward_pressure",  # Index 14
            "defensive_movement",  # Index 15
        ]

        feature_indices = {
            name: self.all_feature_names.index(name)
            for name in key_features
            if name in self.all_feature_names
        }

        # Track evolution over time
        sorted_predictions = sorted(predictions, key=lambda x: x.get("step", 0))

        evolution_data = {}

        for feature_name, feature_idx in feature_indices.items():
            feature_values = []
            steps = []

            for pred in sorted_predictions:
                features = pred.get("strategic_features", [])
                if len(features) > feature_idx:
                    feature_values.append(features[feature_idx])
                    steps.append(pred.get("step", 0))

            if feature_values:
                # Calculate moving averages
                window_size = min(50, len(feature_values) // 10)
                if window_size > 0:
                    moving_avg = (
                        pd.Series(feature_values)
                        .rolling(window=window_size)
                        .mean()
                        .tolist()
                    )
                    evolution_data[feature_name] = {
                        "values": feature_values,
                        "steps": steps,
                        "moving_average": moving_avg,
                        "trend": self._calculate_trend(feature_values),
                        "improvement_rate": self._calculate_improvement_rate(
                            feature_values
                        ),
                    }

        return evolution_data

    def _analyze_action_correlation(self, predictions):
        """Analyze correlation between actions and successful predictions"""
        print("   🎮 Analyzing action correlation patterns...")

        # Extract action patterns
        action_patterns = defaultdict(list)

        for pred in predictions:
            action = pred.get("action", "unknown")
            attack_timing = pred.get("attack_timing", 0)
            defend_timing = pred.get("defend_timing", 0)

            action_patterns[action].append(
                {
                    "attack_timing": attack_timing,
                    "defend_timing": defend_timing,
                    "confidence": max(attack_timing, defend_timing),
                }
            )

        # Analyze by action category
        category_analysis = {}

        for category, action_indices in self.action_categories.items():
            category_predictions = []

            for pred in predictions:
                # Check if action belongs to this category
                action_str = pred.get("action", "")
                if any(
                    f"ACTION_{idx}" in action_str or str(idx) in action_str
                    for idx in action_indices
                ):
                    category_predictions.append(pred)

            if category_predictions:
                attack_timings = [
                    p.get("attack_timing", 0) for p in category_predictions
                ]
                defend_timings = [
                    p.get("defend_timing", 0) for p in category_predictions
                ]

                category_analysis[category] = {
                    "count": len(category_predictions),
                    "avg_attack_confidence": np.mean(attack_timings),
                    "avg_defend_confidence": np.mean(defend_timings),
                    "effectiveness_score": np.mean(
                        [max(a, d) for a, d in zip(attack_timings, defend_timings)]
                    ),
                }

        return {
            "action_patterns": dict(action_patterns),
            "category_analysis": category_analysis,
            "most_effective_categories": sorted(
                category_analysis.items(),
                key=lambda x: x[1]["effectiveness_score"],
                reverse=True,
            )[:5],
        }

    def analyze_performance_metrics(self):
        """Analyze overall performance and identify improvement areas"""
        print("\n📊 ANALYZING PERFORMANCE METRICS...")

        # Extract performance data from various sources
        self.insights["win_rate_analysis"] = self._analyze_win_rates()
        self.insights["damage_analysis"] = self._analyze_damage_patterns()
        self.insights["learning_efficiency"] = self._analyze_learning_efficiency()

        print("✅ Performance analysis complete")

    def _analyze_win_rates(self):
        """Analyze win rate progression and patterns"""
        win_rate_data = []

        # Extract win rate data from performance logs
        for log_entry in self.data.get("logs", {}).get("performance", []):
            content = log_entry["content"]
            if "Win Rate:" in content:
                try:
                    # Parse win rate from log entries
                    parts = content.split("Win Rate:")[1].split()[0]
                    win_rate = float(parts.replace("%", "").replace("(", "")) / 100

                    win_rate_data.append(
                        {
                            "timestamp": log_entry["timestamp"],
                            "win_rate": win_rate,
                            "content": content,
                        }
                    )
                except:
                    continue

        if not win_rate_data:
            return {"status": "no_win_rate_data"}

        # Calculate win rate trends
        win_rates = [d["win_rate"] for d in win_rate_data]

        return {
            "progression": win_rate_data,
            "final_win_rate": win_rates[-1] if win_rates else 0,
            "peak_win_rate": max(win_rates) if win_rates else 0,
            "improvement_trend": self._calculate_trend(win_rates),
            "volatility": np.std(win_rates) if len(win_rates) > 1 else 0,
        }

    def _analyze_damage_patterns(self):
        """Analyze damage dealing and receiving patterns"""
        damage_data = []

        # Extract damage data from analysis files
        for analysis_data in self.data.get("analysis_files", []):
            if "performance_summary" in analysis_data:
                perf = analysis_data["performance_summary"]
                damage_data.append(
                    {
                        "avg_damage_per_round": perf.get("avg_damage_per_round", 0),
                        "defensive_efficiency": perf.get("defensive_efficiency", 0),
                        "total_rounds": perf.get("total_rounds", 0),
                    }
                )

        if not damage_data:
            return {"status": "no_damage_data"}

        return {
            "avg_damage_per_round": np.mean(
                [d["avg_damage_per_round"] for d in damage_data]
            ),
            "defensive_efficiency": np.mean(
                [d["defensive_efficiency"] for d in damage_data]
            ),
            "damage_consistency": np.std(
                [d["avg_damage_per_round"] for d in damage_data]
            ),
        }

    def _analyze_learning_efficiency(self):
        """Analyze how efficiently the model is learning"""
        efficiency_metrics = {}

        # Check milestone progression
        milestone_data = self.data.get("logs", {}).get("milestones", [])
        if milestone_data:
            milestones = []
            for entry in milestone_data:
                if "MILESTONE ACHIEVED" in entry["content"]:
                    try:
                        # Extract milestone percentage and step
                        content = entry["content"]
                        milestone_pct = float(content.split("%")[0].split()[-1])
                        step = int(content.split("step")[1].split()[0].replace(",", ""))
                        milestones.append({"milestone": milestone_pct, "step": step})
                    except:
                        continue

            if milestones:
                efficiency_metrics["milestone_progression"] = milestones
                efficiency_metrics["learning_speed"] = self._calculate_learning_speed(
                    milestones
                )

        # Check learning rate adaptations
        lr_adaptations = len(self.data.get("logs", {}).get("learning_rate", []))
        efficiency_metrics["lr_adaptations"] = lr_adaptations

        return efficiency_metrics

    def generate_improvement_recommendations(self):
        """Generate actionable recommendations for improvement"""
        print("\n💡 GENERATING IMPROVEMENT RECOMMENDATIONS...")

        recommendations = {
            "feature_improvements": [],
            "training_improvements": [],
            "architecture_improvements": [],
            "hyperparameter_improvements": [],
            "priority_level": {},
        }

        # Analyze current performance
        self._recommend_feature_improvements(recommendations)
        self._recommend_training_improvements(recommendations)
        self._recommend_architecture_improvements(recommendations)
        self._recommend_hyperparameter_improvements(recommendations)

        # Prioritize recommendations
        self._prioritize_recommendations(recommendations)

        self.insights["recommendations"] = recommendations
        print("✅ Recommendations generated")

    def _recommend_feature_improvements(self, recommendations):
        """Recommend feature engineering improvements"""
        feature_analysis = self.insights.get("feature_importance", {})

        if not feature_analysis:
            return

        ranked_features = feature_analysis.get("ranked_features", [])

        # Check for underutilized features
        if ranked_features:
            bottom_features = ranked_features[-5:]  # Bottom 5 features

            for feature_name, metrics in bottom_features:
                if metrics["mean_activation"] < 0.1:
                    recommendations["feature_improvements"].append(
                        {
                            "type": "underutilized_feature",
                            "feature": feature_name,
                            "issue": f'Low activation ({metrics["mean_activation"]:.3f})',
                            "suggestion": f"Consider enhancing {feature_name} calculation or removing if consistently unused",
                            "priority": "medium",
                        }
                    )

        # Check combo feature performance
        combo_analysis = self.insights.get("combo_learning", {})
        if combo_analysis and combo_analysis.get("combo_stats"):
            combo_stats = combo_analysis["combo_stats"]

            if combo_stats.get("max_combo_detected", 0) < 3:
                recommendations["feature_improvements"].append(
                    {
                        "type": "combo_detection",
                        "issue": "Low maximum combo detection",
                        "suggestion": "Improve combo detection parameters in StrategicFeatureTracker",
                        "details": "Consider reducing COMBO_TIMEOUT_FRAMES or MIN_SCORE_INCREASE_FOR_HIT",
                        "priority": "high",
                    }
                )

            if combo_stats.get("combo_attack_confidence", 0) < 0.6:
                recommendations["feature_improvements"].append(
                    {
                        "type": "combo_confidence",
                        "issue": "Low attack confidence during combos",
                        "suggestion": "Enhance score momentum calculation for better combo prediction",
                        "priority": "high",
                    }
                )

        # Suggest new features
        recommendations["feature_improvements"].extend(
            [
                {
                    "type": "new_feature",
                    "suggestion": "Add hit/block stun detection feature",
                    "rationale": "Important for frame advantage understanding",
                    "implementation": "Track frame data changes in game state",
                    "priority": "high",
                },
                {
                    "type": "new_feature",
                    "suggestion": "Add projectile tracking feature",
                    "rationale": "Critical for fireball game and zoning",
                    "implementation": "Computer vision detection of projectiles on screen",
                    "priority": "medium",
                },
                {
                    "type": "new_feature",
                    "suggestion": "Add meter/super tracking feature",
                    "rationale": "Essential for high-level play decisions",
                    "implementation": "Parse game state for special meter values",
                    "priority": "medium",
                },
            ]
        )

    def _recommend_training_improvements(self, recommendations):
        """Recommend training process improvements"""

        # Analyze win rate progression
        win_analysis = self.insights.get("win_rate_analysis", {})

        if win_analysis and win_analysis.get("final_win_rate"):
            final_wr = win_analysis["final_win_rate"]

            if final_wr < 0.5:
                recommendations["training_improvements"].append(
                    {
                        "type": "training_time",
                        "issue": f"Low final win rate ({final_wr:.1%})",
                        "suggestion": "Increase total training timesteps significantly",
                        "details": "Consider 25M+ timesteps for complex fighting game mastery",
                        "priority": "high",
                    }
                )

            volatility = win_analysis.get("volatility", 0)
            if volatility > 0.1:
                recommendations["training_improvements"].append(
                    {
                        "type": "stability",
                        "issue": f"High win rate volatility ({volatility:.3f})",
                        "suggestion": "Implement more stable learning rate schedule",
                        "priority": "medium",
                    }
                )

        # Check learning efficiency
        efficiency = self.insights.get("learning_efficiency", {})
        if efficiency.get("lr_adaptations", 0) > 10:
            recommendations["training_improvements"].append(
                {
                    "type": "learning_rate",
                    "issue": "Frequent learning rate adaptations detected",
                    "suggestion": "Implement curriculum learning with staged difficulty",
                    "details": "Start with easier opponents, gradually increase difficulty",
                    "priority": "medium",
                }
            )

        # Recommend curriculum learning
        recommendations["training_improvements"].extend(
            [
                {
                    "type": "curriculum_learning",
                    "suggestion": "Implement staged training curriculum",
                    "details": [
                        "Stage 1: Focus on basic movements and attacks (30% of training)",
                        "Stage 2: Introduce combo training with rewards (40% of training)",
                        "Stage 3: Advanced strategy and counter-play (30% of training)",
                    ],
                    "priority": "high",
                },
                {
                    "type": "reward_shaping",
                    "suggestion": "Enhanced reward engineering",
                    "details": [
                        "Increase combo multiplier rewards",
                        "Add frame advantage rewards",
                        "Implement spacing optimization rewards",
                        "Add anti-air and defensive bonuses",
                    ],
                    "priority": "high",
                },
                {
                    "type": "self_play",
                    "suggestion": "Implement self-play training",
                    "details": "Train against previous versions of the model for diverse strategies",
                    "priority": "medium",
                },
            ]
        )

    def _recommend_architecture_improvements(self, recommendations):
        """Recommend neural network architecture improvements"""

        # Analyze transformer performance
        temporal_analysis = self.insights.get("temporal_patterns", {})

        if temporal_analysis:
            phase_analysis = temporal_analysis.get("phase_analysis", {})

            # Check learning progression
            if phase_analysis:
                late_phase = phase_analysis.get("late", {})
                early_phase = phase_analysis.get("early", {})

                if late_phase and early_phase:
                    late_confidence = late_phase.get("avg_attack_confidence", 0)
                    early_confidence = early_phase.get("avg_attack_confidence", 0)

                    improvement = late_confidence - early_confidence

                    if improvement < 0.1:
                        recommendations["architecture_improvements"].append(
                            {
                                "type": "transformer_capacity",
                                "issue": "Limited learning progression in transformer",
                                "suggestion": "Increase transformer model capacity",
                                "details": [
                                    "Increase d_model from 384 to 512",
                                    "Add more transformer layers (8-10)",
                                    "Increase attention heads to 16",
                                    "Add residual connections in tactical predictor",
                                ],
                                "priority": "high",
                            }
                        )

        # Architecture recommendations based on fighting game requirements
        recommendations["architecture_improvements"].extend(
            [
                {
                    "type": "attention_mechanism",
                    "suggestion": "Add cross-attention between visual and strategic features",
                    "rationale": "Better integration of visual patterns with strategic understanding",
                    "implementation": "Separate encoders with cross-attention fusion",
                    "priority": "high",
                },
                {
                    "type": "temporal_modeling",
                    "suggestion": "Add LSTM layer before transformer",
                    "rationale": "Better capture of frame-by-frame dependencies",
                    "implementation": "BiLSTM -> Transformer -> Tactical Predictor",
                    "priority": "medium",
                },
                {
                    "type": "multi_head_prediction",
                    "suggestion": "Separate prediction heads for different aspects",
                    "details": [
                        "Attack timing head",
                        "Defend timing head",
                        "Combo opportunity head",
                        "Spacing optimization head",
                        "Counter-attack head",
                    ],
                    "priority": "high",
                },
                {
                    "type": "feature_fusion",
                    "suggestion": "Implement learned feature fusion instead of concatenation",
                    "details": "Use attention-based fusion of visual and strategic features",
                    "priority": "medium",
                },
            ]
        )

    def _recommend_hyperparameter_improvements(self, recommendations):
        """Recommend hyperparameter tuning improvements"""

        # Learning rate recommendations
        win_analysis = self.insights.get("win_rate_analysis", {})
        if win_analysis:
            trend = win_analysis.get("improvement_trend", 0)

            if trend < 0:
                recommendations["hyperparameter_improvements"].append(
                    {
                        "type": "learning_rate",
                        "issue": "Declining performance trend detected",
                        "suggestion": "Reduce learning rate and add warmup",
                        "details": "Use linear warmup for first 10% of training, then cosine decay",
                        "priority": "high",
                    }
                )

        # PPO-specific recommendations
        recommendations["hyperparameter_improvements"].extend(
            [
                {
                    "type": "ppo_clip_range",
                    "suggestion": "Adaptive clip range based on performance",
                    "current": "0.2 (fixed)",
                    "recommended": "Start at 0.3, decay to 0.1",
                    "rationale": "Larger early exploration, more conservative later",
                    "priority": "medium",
                },
                {
                    "type": "batch_size",
                    "suggestion": "Increase batch size for more stable gradients",
                    "current": "128",
                    "recommended": "256-512 for fighting games",
                    "rationale": "Fighting games benefit from larger batch sizes",
                    "priority": "medium",
                },
                {
                    "type": "gae_lambda",
                    "suggestion": "Tune GAE lambda for fighting game dynamics",
                    "current": "0.95",
                    "recommended": "0.98-0.99",
                    "rationale": "Longer-term dependencies important in fighting games",
                    "priority": "low",
                },
                {
                    "type": "entropy_coefficient",
                    "suggestion": "Dynamic entropy coefficient schedule",
                    "current": "0.005 (fixed)",
                    "recommended": "Start at 0.01, decay to 0.001",
                    "rationale": "Encourage exploration early, exploit later",
                    "priority": "medium",
                },
            ]
        )

    def _prioritize_recommendations(self, recommendations):
        """Assign priority levels to all recommendations"""

        priority_counts = {"high": 0, "medium": 0, "low": 0}

        for category in [
            "feature_improvements",
            "training_improvements",
            "architecture_improvements",
            "hyperparameter_improvements",
        ]:
            for rec in recommendations[category]:
                priority = rec.get("priority", "low")
                priority_counts[priority] += 1

        recommendations["priority_level"] = priority_counts

        # Sort recommendations by priority within each category
        for category in recommendations:
            if isinstance(recommendations[category], list):
                recommendations[category].sort(
                    key=lambda x: {"high": 3, "medium": 2, "low": 1}.get(
                        x.get("priority", "low"), 0
                    ),
                    reverse=True,
                )

    def create_visualizations(self):
        """Create comprehensive visualizations of the analysis"""
        print("\n📊 CREATING VISUALIZATIONS...")

        # Create plots directory
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Feature importance plot
        self._plot_feature_importance(plots_dir)

        # Learning progression plot
        self._plot_learning_progression(plots_dir)

        # Combo analysis plot
        self._plot_combo_analysis(plots_dir)

        # Performance trends plot
        self._plot_performance_trends(plots_dir)

        # Action effectiveness plot
        self._plot_action_effectiveness(plots_dir)

        print(f"✅ Visualizations saved to {plots_dir}")

    def _plot_feature_importance(self, plots_dir):
        """Plot feature importance analysis"""
        feature_analysis = self.insights.get("feature_importance", {})

        if not feature_analysis or not feature_analysis.get("ranked_features"):
            return

        # Top 15 features
        top_features = feature_analysis["ranked_features"][:15]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Mean activation plot
        feature_names = [f[0] for f in top_features]
        activations = [f[1]["mean_activation"] for f in top_features]

        ax1.barh(range(len(feature_names)), activations, color="skyblue")
        ax1.set_yticks(range(len(feature_names)))
        ax1.set_yticklabels(feature_names, fontsize=10)
        ax1.set_xlabel("Mean Activation")
        ax1.set_title("Top 15 Features by Mean Activation")
        ax1.grid(axis="x", alpha=0.3)

        # Correlation with predictions
        attack_corrs = [f[1]["attack_correlation"] for f in top_features]
        defend_corrs = [f[1]["defend_correlation"] for f in top_features]

        x = np.arange(len(feature_names))
        width = 0.35

        ax2.barh(
            x - width / 2,
            attack_corrs,
            width,
            label="Attack Correlation",
            color="red",
            alpha=0.7,
        )
        ax2.barh(
            x + width / 2,
            defend_corrs,
            width,
            label="Defend Correlation",
            color="blue",
            alpha=0.7,
        )

        ax2.set_yticks(x)
        ax2.set_yticklabels(feature_names, fontsize=10)
        ax2.set_xlabel("Correlation with Predictions")
        ax2.set_title("Feature Correlation with Attack/Defend Predictions")
        ax2.legend()
        ax2.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / "feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_learning_progression(self, plots_dir):
        """Plot learning progression over time"""
        temporal_analysis = self.insights.get("temporal_patterns", {})

        if not temporal_analysis or not temporal_analysis.get("phase_analysis"):
            return

        phase_analysis = temporal_analysis["phase_analysis"]

        phases = list(phase_analysis.keys())
        attack_confidences = [
            phase_analysis[p]["avg_attack_confidence"] for p in phases
        ]
        defend_confidences = [
            phase_analysis[p]["avg_defend_confidence"] for p in phases
        ]

        fig, ax = plt.subplots(figsize=(12, 6))

        x = range(len(phases))

        ax.plot(
            x,
            attack_confidences,
            "ro-",
            label="Attack Confidence",
            linewidth=2,
            markersize=8,
        )
        ax.plot(
            x,
            defend_confidences,
            "bo-",
            label="Defend Confidence",
            linewidth=2,
            markersize=8,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(phases)
        ax.set_ylabel("Average Confidence")
        ax.set_title("Learning Progression Across Training Phases")
        ax.legend()
        ax.grid(alpha=0.3)

        # Add trend lines
        z_attack = np.polyfit(x, attack_confidences, 1)
        p_attack = np.poly1d(z_attack)
        ax.plot(x, p_attack(x), "r--", alpha=0.8, label="Attack Trend")

        z_defend = np.polyfit(x, defend_confidences, 1)
        p_defend = np.poly1d(z_defend)
        ax.plot(x, p_defend(x), "b--", alpha=0.8, label="Defend Trend")

        plt.tight_layout()
        plt.savefig(
            plots_dir / "learning_progression.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_combo_analysis(self, plots_dir):
        """Plot combo learning analysis"""
        combo_analysis = self.insights.get("combo_learning", {})

        if not combo_analysis or combo_analysis.get("status") == "no_combo_data":
            return

        combo_distribution = combo_analysis.get("combo_distribution", {})
        confidence_by_length = combo_analysis.get("confidence_by_combo_length", {})

        if not combo_distribution:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Combo length distribution
        lengths = list(combo_distribution.keys())
        counts = list(combo_distribution.values())

        ax1.bar(lengths, counts, color="orange", alpha=0.7)
        ax1.set_xlabel("Combo Length")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Combo Length Distribution")
        ax1.grid(axis="y", alpha=0.3)

        # Confidence by combo length
        if confidence_by_length:
            conf_lengths = list(confidence_by_length.keys())
            confidences = list(confidence_by_length.values())

            ax2.plot(conf_lengths, confidences, "go-", linewidth=2, markersize=8)
            ax2.set_xlabel("Combo Length")
            ax2.set_ylabel("Average Attack Confidence")
            ax2.set_title("Attack Confidence by Combo Length")
            ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / "combo_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_performance_trends(self, plots_dir):
        """Plot overall performance trends"""
        win_analysis = self.insights.get("win_rate_analysis", {})

        if not win_analysis or win_analysis.get("status") == "no_win_rate_data":
            return

        progression = win_analysis.get("progression", [])

        if not progression:
            return

        timestamps = [p["timestamp"] for p in progression if p["timestamp"]]
        win_rates = [p["win_rate"] for p in progression]

        if not timestamps:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(timestamps, win_rates, "b-", linewidth=2, alpha=0.7)
        ax.scatter(timestamps, win_rates, color="blue", s=30, alpha=0.8)

        # Add moving average
        if len(win_rates) > 5:
            window_size = max(3, len(win_rates) // 10)
            moving_avg = (
                pd.Series(win_rates).rolling(window=window_size, center=True).mean()
            )
            ax.plot(
                timestamps,
                moving_avg,
                "r-",
                linewidth=3,
                label=f"{window_size}-point Moving Average",
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Win Rate")
        ax.set_title("Win Rate Progression Over Time")
        ax.grid(alpha=0.3)
        ax.legend()

        # Format x-axis
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(plots_dir / "performance_trends.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_action_effectiveness(self, plots_dir):
        """Plot action category effectiveness"""
        action_analysis = self.insights.get("action_correlation", {})

        if not action_analysis or not action_analysis.get("category_analysis"):
            return

        category_analysis = action_analysis["category_analysis"]

        categories = list(category_analysis.keys())
        effectiveness_scores = [
            category_analysis[cat]["effectiveness_score"] for cat in categories
        ]
        attack_confidences = [
            category_analysis[cat]["avg_attack_confidence"] for cat in categories
        ]
        defend_confidences = [
            category_analysis[cat]["avg_defend_confidence"] for cat in categories
        ]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Effectiveness scores
        bars = ax1.bar(categories, effectiveness_scores, color="green", alpha=0.7)
        ax1.set_ylabel("Effectiveness Score")
        ax1.set_title("Action Category Effectiveness")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, score in zip(bars, effectiveness_scores):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Attack vs Defend confidence
        x = np.arange(len(categories))
        width = 0.35

        ax2.bar(
            x - width / 2,
            attack_confidences,
            width,
            label="Attack Confidence",
            color="red",
            alpha=0.7,
        )
        ax2.bar(
            x + width / 2,
            defend_confidences,
            width,
            label="Defend Confidence",
            color="blue",
            alpha=0.7,
        )

        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, rotation=45)
        ax2.set_ylabel("Average Confidence")
        ax2.set_title("Attack vs Defend Confidence by Action Category")
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            plots_dir / "action_effectiveness.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report"""
        print("\n📋 GENERATING COMPREHENSIVE REPORT...")

        report = {
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "data_sources": {
                    "analysis_files": len(self.data.get("analysis_files", [])),
                    "log_files": sum(
                        len(logs) for logs in self.data.get("logs", {}).values()
                    ),
                    "checkpoints": len(self.data.get("checkpoints", [])),
                },
            },
            "executive_summary": self._generate_executive_summary(),
            "detailed_analysis": self.insights,
            "visualizations_created": [
                "feature_importance.png",
                "learning_progression.png",
                "combo_analysis.png",
                "performance_trends.png",
                "action_effectiveness.png",
            ],
        }

        # Save comprehensive report
        report_path = self.output_dir / "comprehensive_analysis_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Generate markdown summary
        self._generate_markdown_summary(report)

        print(f"✅ Comprehensive report saved to {report_path}")

        return report

    def _generate_executive_summary(self):
        """Generate executive summary of findings"""
        summary = {
            "overall_status": "unknown",
            "key_findings": [],
            "critical_issues": [],
            "top_recommendations": [],
        }

        # Determine overall status
        win_analysis = self.insights.get("win_rate_analysis", {})
        if win_analysis and win_analysis.get("final_win_rate"):
            final_wr = win_analysis["final_win_rate"]

            if final_wr >= 0.7:
                summary["overall_status"] = "excellent"
            elif final_wr >= 0.5:
                summary["overall_status"] = "good"
            elif final_wr >= 0.3:
                summary["overall_status"] = "developing"
            else:
                summary["overall_status"] = "needs_improvement"

        # Key findings
        feature_analysis = self.insights.get("feature_importance", {})
        if feature_analysis and feature_analysis.get("ranked_features"):
            top_feature = feature_analysis["ranked_features"][0]
            summary["key_findings"].append(
                f"Most important learned feature: {top_feature[0]} "
                f"(activation: {top_feature[1]['mean_activation']:.3f})"
            )

        combo_analysis = self.insights.get("combo_learning", {})
        if combo_analysis and combo_analysis.get("combo_stats"):
            max_combo = combo_analysis["combo_stats"].get("max_combo_detected", 0)
            summary["key_findings"].append(
                f"Maximum combo length detected: {max_combo} hits"
            )

        # Critical issues
        recommendations = self.insights.get("recommendations", {})
        if recommendations:
            high_priority = []
            for category in [
                "feature_improvements",
                "training_improvements",
                "architecture_improvements",
                "hyperparameter_improvements",
            ]:
                for rec in recommendations.get(category, []):
                    if rec.get("priority") == "high":
                        high_priority.append(rec)

            summary["critical_issues"] = [
                rec.get("issue", rec.get("suggestion", "")) for rec in high_priority[:3]
            ]
            summary["top_recommendations"] = [
                rec.get("suggestion", "") for rec in high_priority[:5]
            ]

        return summary

    def _generate_markdown_summary(self, report):
        """Generate markdown summary report"""

        markdown_content = f"""# Street Fighter II Transformer Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

**Overall Status:** {report['executive_summary']['overall_status'].replace('_', ' ').title()}

### Key Findings
"""

        for finding in report["executive_summary"]["key_findings"]:
            markdown_content += f"- {finding}\n"

        markdown_content += "\n### Critical Issues\n"
        for issue in report["executive_summary"]["critical_issues"]:
            markdown_content += f"- ❌ {issue}\n"

        markdown_content += "\n### Top Recommendations\n"
        for rec in report["executive_summary"]["top_recommendations"]:
            markdown_content += f"- 💡 {rec}\n"

        # Add detailed sections
        recommendations = self.insights.get("recommendations", {})

        if recommendations.get("feature_improvements"):
            markdown_content += "\n## Feature Improvements\n\n"
            for rec in recommendations["feature_improvements"][:5]:
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                    rec.get("priority", "low"), "⚪"
                )
                markdown_content += (
                    f"### {priority_emoji} {rec.get('suggestion', 'Unknown')}\n"
                )
                markdown_content += f"**Issue:** {rec.get('issue', 'N/A')}\n\n"
                if "details" in rec:
                    if isinstance(rec["details"], list):
                        for detail in rec["details"]:
                            markdown_content += f"- {detail}\n"
                    else:
                        markdown_content += f"{rec['details']}\n"
                markdown_content += "\n"

        if recommendations.get("training_improvements"):
            markdown_content += "\n## Training Improvements\n\n"
            for rec in recommendations["training_improvements"][:3]:
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                    rec.get("priority", "low"), "⚪"
                )
                markdown_content += (
                    f"### {priority_emoji} {rec.get('suggestion', 'Unknown')}\n"
                )
                if "details" in rec:
                    if isinstance(rec["details"], list):
                        for detail in rec["details"]:
                            markdown_content += f"- {detail}\n"
                    else:
                        markdown_content += f"{rec['details']}\n"
                markdown_content += "\n"

        markdown_content += f"""
## Data Sources Analyzed

- Analysis Files: {report['metadata']['data_sources']['analysis_files']}
- Log Entries: {report['metadata']['data_sources']['log_files']}
- Model Checkpoints: {report['metadata']['data_sources']['checkpoints']}

## Visualizations Generated

"""
        for viz in report["visualizations_created"]:
            markdown_content += f"- 📊 `plots/{viz}`\n"

        markdown_content += f"""
## Next Steps

1. **High Priority:** Address critical issues listed above
2. **Medium Priority:** Implement suggested feature improvements  
3. **Long Term:** Consider architecture enhancements
4. **Monitoring:** Set up automated performance tracking

---
*Report generated by Street Fighter II Analytics System*
"""

        # Save markdown report
        markdown_path = self.output_dir / "analysis_summary.md"
        with open(markdown_path, "w") as f:
            f.write(markdown_content)

        print(f"✅ Markdown summary saved to {markdown_path}")

    # Helper methods for calculations
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

    def _calculate_learning_trend(self, predictions):
        """Calculate overall learning trend"""
        if len(predictions) < 10:
            return {}

        steps = [p.get("step", 0) for p in predictions]
        attack_timings = [p.get("attack_timing", 0) for p in predictions]
        defend_timings = [p.get("defend_timing", 0) for p in predictions]

        return {
            "attack_trend": self._calculate_trend(attack_timings),
            "defend_trend": self._calculate_trend(defend_timings),
            "steps_analyzed": len(predictions),
        }

    def _analyze_confidence_evolution(self, predictions):
        """Analyze how confidence evolved over training"""
        if len(predictions) < 20:
            return {}

        # Split into early and late phases
        mid_point = len(predictions) // 2
        early_predictions = predictions[:mid_point]
        late_predictions = predictions[mid_point:]

        early_attack = np.mean([p.get("attack_timing", 0) for p in early_predictions])
        late_attack = np.mean([p.get("attack_timing", 0) for p in late_predictions])

        early_defend = np.mean([p.get("defend_timing", 0) for p in early_predictions])
        late_defend = np.mean([p.get("defend_timing", 0) for p in late_predictions])

        return {
            "attack_improvement": late_attack - early_attack,
            "defend_improvement": late_defend - early_defend,
            "early_phase_confidence": (early_attack + early_defend) / 2,
            "late_phase_confidence": (late_attack + late_defend) / 2,
        }

    def _assess_combo_learning_effectiveness(self, combo_predictions):
        """Assess how effectively combos were learned"""
        if not combo_predictions:
            return {"status": "no_data"}

        # Group by combo length
        by_length = defaultdict(list)
        for pred in combo_predictions:
            length = pred["combo_count"]
            by_length[length].append(pred["attack_timing"])

        # Check if longer combos have higher confidence
        avg_confidence_by_length = {
            length: np.mean(confidences) for length, confidences in by_length.items()
        }

        if len(avg_confidence_by_length) > 1:
            lengths = sorted(avg_confidence_by_length.keys())
            confidences = [avg_confidence_by_length[l] for l in lengths]

            # Check if confidence increases with combo length
            correlation = np.corrcoef(lengths, confidences)[0, 1]

            return {
                "length_confidence_correlation": correlation,
                "effectiveness_score": max(0, correlation),  # 0-1 scale
                "confidence_by_length": avg_confidence_by_length,
            }

        return {"status": "insufficient_data"}

    def _calculate_learning_speed(self, milestones):
        """Calculate learning speed from milestone data"""
        if len(milestones) < 2:
            return {"status": "insufficient_milestones"}

        # Calculate steps between milestones
        milestone_gaps = []
        for i in range(1, len(milestones)):
            steps_gap = milestones[i]["step"] - milestones[i - 1]["step"]
            milestone_gaps.append(steps_gap)

        return {
            "avg_steps_per_milestone": np.mean(milestone_gaps),
            "learning_acceleration": self._calculate_trend(milestone_gaps),
            "total_milestones": len(milestones),
        }


def main():
    """Main analysis execution"""
    parser = argparse.ArgumentParser(
        description="Street Fighter II Transformer Analytics"
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

    args = parser.parse_args()

    print("🎮 STREET FIGHTER II TRANSFORMER ANALYTICS")
    print("=" * 50)

    # Initialize analytics system
    analytics = StreetFighterAnalytics(
        analysis_dir=args.analysis_dir, logs_dir=args.logs_dir
    )
    analytics.output_dir = Path(args.output_dir)
    analytics.output_dir.mkdir(exist_ok=True)

    try:
        # Load all available data
        analytics.load_all_data()

        # Perform comprehensive analysis
        analytics.analyze_transformer_learning()
        analytics.analyze_performance_metrics()
        analytics.generate_improvement_recommendations()

        # Create visualizations (unless disabled)
        if not args.no_plots:
            analytics.create_visualizations()

        # Generate comprehensive report
        report = analytics.generate_comprehensive_report()

        # Print summary to console
        print("\n" + "=" * 60)
        print("🎯 ANALYSIS COMPLETE - KEY INSIGHTS")
        print("=" * 60)

        exec_summary = report["executive_summary"]
        print(
            f"📊 Overall Status: {exec_summary['overall_status'].replace('_', ' ').title()}"
        )

        if exec_summary["key_findings"]:
            print("\n🔍 Key Findings:")
            for finding in exec_summary["key_findings"]:
                print(f"  • {finding}")

        if exec_summary["critical_issues"]:
            print("\n❌ Critical Issues:")
            for issue in exec_summary["critical_issues"]:
                print(f"  • {issue}")

        if exec_summary["top_recommendations"]:
            print("\n💡 Top Recommendations:")
            for i, rec in enumerate(exec_summary["top_recommendations"][:3], 1):
                print(f"  {i}. {rec}")

        # Print file locations
        print(f"\n📁 Output Files:")
        print(
            f"  • Comprehensive Report: {analytics.output_dir}/comprehensive_analysis_report.json"
        )
        print(f"  • Summary Report: {analytics.output_dir}/analysis_summary.md")
        if not args.no_plots:
            print(f"  • Visualizations: {analytics.output_dir}/plots/")

        # Specific insights based on analysis
        print("\n" + "=" * 60)
        print("🧠 TRANSFORMER LEARNING INSIGHTS")
        print("=" * 60)

        feature_analysis = analytics.insights.get("feature_importance", {})
        if feature_analysis and feature_analysis.get("ranked_features"):
            print("\n🏆 Most Important Features Learned:")
            for i, (feature_name, metrics) in enumerate(
                feature_analysis["ranked_features"][:5], 1
            ):
                activation = metrics["mean_activation"]
                attack_corr = metrics["attack_correlation"]
                defend_corr = metrics["defend_correlation"]
                print(f"  {i}. {feature_name}")
                print(
                    f"     Activation: {activation:.3f} | Attack Corr: {attack_corr:.3f} | Defend Corr: {defend_corr:.3f}"
                )

        combo_analysis = analytics.insights.get("combo_learning", {})
        if combo_analysis and combo_analysis.get("combo_stats"):
            combo_stats = combo_analysis["combo_stats"]
            print(f"\n🔥 Combo Learning Performance:")
            print(
                f"  • Max Combo Detected: {combo_stats.get('max_combo_detected', 0)} hits"
            )
            print(
                f"  • Average Combo Length: {combo_stats.get('avg_combo_length', 0):.1f}"
            )
            print(
                f"  • Total Combo Predictions: {combo_stats.get('total_combo_predictions', 0):,}"
            )
            print(
                f"  • Combo Attack Confidence: {combo_stats.get('combo_attack_confidence', 0):.3f}"
            )

        temporal_analysis = analytics.insights.get("temporal_patterns", {})
        if temporal_analysis and temporal_analysis.get("confidence_evolution"):
            conf_evo = temporal_analysis["confidence_evolution"]
            print(f"\n📈 Learning Evolution:")
            print(
                f"  • Attack Confidence Improvement: {conf_evo.get('attack_improvement', 0):+.3f}"
            )
            print(
                f"  • Defend Confidence Improvement: {conf_evo.get('defend_improvement', 0):+.3f}"
            )
            print(
                f"  • Early Phase Confidence: {conf_evo.get('early_phase_confidence', 0):.3f}"
            )
            print(
                f"  • Late Phase Confidence: {conf_evo.get('late_phase_confidence', 0):.3f}"
            )

        action_analysis = analytics.insights.get("action_correlation", {})
        if action_analysis and action_analysis.get("most_effective_categories"):
            print(f"\n🎮 Most Effective Action Categories:")
            for i, (category, data) in enumerate(
                action_analysis["most_effective_categories"][:3], 1
            ):
                effectiveness = data["effectiveness_score"]
                count = data["count"]
                print(f"  {i}. {category.replace('_', ' ').title()}")
                print(
                    f"     Effectiveness: {effectiveness:.3f} | Usage: {count:,} predictions"
                )

        print("\n" + "=" * 60)
        print("🚀 IMPROVEMENT ROADMAP")
        print("=" * 60)

        recommendations = analytics.insights.get("recommendations", {})

        # High priority recommendations
        high_priority_recs = []
        for category in [
            "feature_improvements",
            "training_improvements",
            "architecture_improvements",
            "hyperparameter_improvements",
        ]:
            for rec in recommendations.get(category, []):
                if rec.get("priority") == "high":
                    high_priority_recs.append((category, rec))

        if high_priority_recs:
            print("\n🔴 HIGH PRIORITY ACTIONS:")
            for i, (category, rec) in enumerate(high_priority_recs[:5], 1):
                cat_name = category.replace("_", " ").title()
                suggestion = rec.get("suggestion", "Unknown")
                print(f"  {i}. [{cat_name}] {suggestion}")
                if "issue" in rec:
                    print(f"     Issue: {rec['issue']}")

        # Feature-specific recommendations
        feature_recs = recommendations.get("feature_improvements", [])
        new_feature_recs = [r for r in feature_recs if r.get("type") == "new_feature"]

        if new_feature_recs:
            print(f"\n💡 SUGGESTED NEW FEATURES:")
            for i, rec in enumerate(new_feature_recs[:3], 1):
                print(f"  {i}. {rec.get('suggestion', 'Unknown')}")
                print(f"     Rationale: {rec.get('rationale', 'N/A')}")
                if "implementation" in rec:
                    print(f"     Implementation: {rec['implementation']}")

        # Architecture recommendations
        arch_recs = recommendations.get("architecture_improvements", [])
        if arch_recs:
            print(f"\n🏗️ ARCHITECTURE IMPROVEMENTS:")
            for i, rec in enumerate(arch_recs[:3], 1):
                print(f"  {i}. {rec.get('suggestion', 'Unknown')}")
                if "rationale" in rec:
                    print(f"     Rationale: {rec['rationale']}")

        print(
            f"\n✅ Analysis complete! Check {analytics.output_dir}/ for detailed reports and visualizations."
        )

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
