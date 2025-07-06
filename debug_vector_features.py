#!/usr/bin/env python3
"""
debug_vector_features.py - Diagnose why vector features are all zeros
"""

import numpy as np
import retro
from wrapper import StreetFighterVisionWrapper


def debug_vector_features():
    """Debug vector feature generation step by step."""
    print("🔍 VECTOR FEATURE DEBUG")
    print("=" * 50)

    # Create environment
    try:
        env = retro.make(
            game="StreetFighterIISpecialChampionEdition-Genesis",
            state="ken_bison_12.state",
        )
        env = StreetFighterVisionWrapper(env, frame_stack=8)
        print("✅ Environment created")
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return

    # Reset and get initial observation
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    print(f"\n📊 Initial Observation Analysis:")
    print(f"   - Visual shape: {obs['visual_obs'].shape}")
    print(f"   - Vector shape: {obs['vector_obs'].shape}")
    print(f"   - Vector mean: {obs['vector_obs'].mean():.6f}")
    print(f"   - Vector std: {obs['vector_obs'].std():.6f}")
    print(
        f"   - Vector min/max: {obs['vector_obs'].min():.6f}/{obs['vector_obs'].max():.6f}"
    )

    if np.all(obs["vector_obs"] == 0):
        print("   🚨 PROBLEM: All vector features are zero at start!")

    # Take some actions and see if vector features change
    print(f"\n🎮 Testing Vector Feature Evolution:")

    vector_history = []
    info_history = []

    for step in range(20):
        # Take random action
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        vector_obs = obs["vector_obs"]
        vector_history.append(vector_obs.copy())
        info_history.append(info.copy())

        # Check current vector state
        vector_mean = vector_obs.mean()
        vector_std = vector_obs.std()
        vector_max = vector_obs.max()

        print(
            f"   Step {step+1:2d}: mean={vector_mean:.6f}, std={vector_std:.6f}, max={vector_max:.6f}"
        )

        # Show info that should influence vector features
        player_hp = info.get("agent_hp", "N/A")
        enemy_hp = info.get("enemy_hp", "N/A")
        player_x = info.get("agent_x", "N/A")
        enemy_x = info.get("enemy_x", "N/A")

        if step % 5 == 0:
            print(f"      Info: HP={player_hp}/{enemy_hp}, X={player_x}/{enemy_x}")

        if done or truncated:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            print("      🔄 Environment reset")

    # Analyze vector feature components
    print(f"\n🧪 Vector Feature Component Analysis:")

    if len(vector_history) > 0:
        # Stack all vector observations
        vector_stack = np.stack(vector_history)  # [steps, seq_len, features]

        print(f"   - Collected {len(vector_history)} vector observations")
        print(f"   - Shape: {vector_stack.shape}")

        # Check each feature dimension
        feature_stats = []
        for feat_idx in range(vector_stack.shape[2]):
            feat_data = vector_stack[:, :, feat_idx]
            feat_mean = feat_data.mean()
            feat_std = feat_data.std()
            feat_min = feat_data.min()
            feat_max = feat_data.max()

            feature_stats.append(
                {
                    "index": feat_idx,
                    "mean": feat_mean,
                    "std": feat_std,
                    "min": feat_min,
                    "max": feat_max,
                    "active": feat_std > 1e-6,
                }
            )

        # Show feature breakdown
        active_features = sum(1 for fs in feature_stats if fs["active"])
        print(f"   - Active features: {active_features}/{len(feature_stats)}")

        # Show first few active features
        print(f"   - Active feature details:")
        for fs in feature_stats[:10]:  # First 10 features
            status = "✅ ACTIVE" if fs["active"] else "❌ ZERO"
            print(
                f"      Feature {fs['index']:2d}: {status} | "
                f"mean={fs['mean']:.6f}, std={fs['std']:.6f}"
            )

        if active_features == 0:
            print("   🚨 CRITICAL: NO ACTIVE FEATURES!")
            print("   📋 Checking info extraction...")

            # Check if info contains the expected fields
            if info_history:
                sample_info = info_history[-1]
                print(f"   📊 Last info keys: {list(sample_info.keys())}")

                expected_keys = ["agent_hp", "enemy_hp", "agent_x", "enemy_x", "score"]
                missing_keys = [key for key in expected_keys if key not in sample_info]

                if missing_keys:
                    print(f"   ❌ Missing info keys: {missing_keys}")
                    print("   💡 This explains why vector features are zero!")
                else:
                    print(f"   ✅ All expected info keys present")
                    print("   💡 Issue is in feature extraction logic")

    # Test strategic tracker directly
    print(f"\n🎯 Testing Strategic Tracker Directly:")

    try:
        # Access the strategic tracker from the wrapper
        tracker = env.strategic_tracker

        # Create some dummy info
        test_info = {
            "agent_hp": 150,
            "enemy_hp": 140,
            "agent_x": 50,
            "enemy_x": 80,
            "agent_y": 64,
            "enemy_y": 64,
            "score": 1000,
            "agent_status": 0,
            "enemy_status": 0,
        }

        # Create dummy button features
        button_features = np.array(
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32
        )

        # Update tracker
        features = tracker.update(test_info, button_features)

        print(f"   ✅ Strategic tracker working")
        print(f"   📊 Features shape: {features.shape}")
        print(f"   📊 Features mean: {features.mean():.6f}")
        print(f"   📊 Features std: {features.std():.6f}")
        print(f"   📊 Non-zero features: {np.count_nonzero(features)}/{len(features)}")

        if np.all(features == 0):
            print("   🚨 Strategic tracker also producing zeros!")
            print("   💡 Bug in strategic feature calculation")
        else:
            print("   ✅ Strategic tracker producing valid features")
            print("   💡 Issue might be in wrapper integration")

    except Exception as e:
        print(f"   ❌ Strategic tracker test failed: {e}")
        import traceback

        traceback.print_exc()

    env.close()
    print(f"\n🏁 Vector Feature Debug Complete")


if __name__ == "__main__":
    debug_vector_features()
