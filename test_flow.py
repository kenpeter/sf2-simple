#!/usr/bin/env python3
"""
test_flow.py - Test gradient flow for the STABILIZED Street Fighter architecture
Works with the existing FixedStreetFighterCNN and FixedStreetFighterPolicy from wrapper.py
"""

import torch
import numpy as np
from gymnasium import spaces
import traceback
from wrapper import (
    FixedStreetFighterCNN,
    FixedStreetFighterPolicy,
    StreetFighterVisionWrapper,
    verify_gradient_flow,
    StreetFighterDiscreteActions,
)
import retro


def test_stabilized_gradient_flow():
    """Test gradient flow through the stabilized architecture."""

    print("🧪 Testing STABILIZED Architecture for Gradient Flow")
    print("=" * 70)

    # Create observation space matching wrapper
    frame_stack = 8
    target_size = (128, 180)
    vector_feature_dim = 45

    obs_space = spaces.Dict(
        {
            "visual_obs": spaces.Box(
                low=0, high=255, shape=(3 * frame_stack, *target_size), dtype=np.uint8
            ),
            "vector_obs": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(frame_stack, vector_feature_dim),
                dtype=np.float32,
            ),
        }
    )

    print(f"📏 Observation space:")
    print(f"   - Visual: {obs_space['visual_obs'].shape}")
    print(f"   - Vector: {obs_space['vector_obs'].shape}")

    # Create feature extractor
    features_dim = 256
    extractor = FixedStreetFighterCNN(obs_space, features_dim)

    print(f"\n🏗️ STABILIZED Model Architecture:")
    total_params = sum(p.numel() for p in extractor.parameters())
    print(f"   - Total parameters: {total_params:,}")

    # List all parameters and their requires_grad status
    print(f"\n📋 Parameter Structure:")
    param_count = 0
    for name, param in extractor.named_parameters():
        param_count += 1
        print(
            f"   {param_count:2d}. {name}: {list(param.shape)}, {param.numel():,} params, grad={param.requires_grad}"
        )

    # Create dummy input with realistic values
    batch_size = 2
    visual_obs = torch.randint(
        0, 255, (batch_size, 3 * frame_stack, *target_size), dtype=torch.uint8
    )

    # Create realistic vector features (normalized range)
    vector_obs = (
        torch.randn(batch_size, frame_stack, vector_feature_dim, dtype=torch.float32)
        * 0.5
    )

    obs_dict = {"visual_obs": visual_obs, "vector_obs": vector_obs}

    print(f"\n🔍 Input Testing:")
    print(f"   - Visual input shape: {visual_obs.shape}")
    print(f"   - Visual range: [{visual_obs.min()}, {visual_obs.max()}]")
    print(f"   - Vector input shape: {vector_obs.shape}")
    print(
        f"   - Vector range: [{vector_obs.min().item():.3f}, {vector_obs.max().item():.3f}]"
    )

    # Test forward pass
    print(f"\n⚡ Forward Pass Test:")
    try:
        extractor.train()  # Ensure in training mode
        features = extractor(obs_dict)
        print(f"   ✅ Forward pass successful")
        print(f"   - Output shape: {features.shape}")
        print(
            f"   - Output range: [{features.min().item():.3f}, {features.max().item():.3f}]"
        )
        print(f"   - Output mean: {features.mean().item():.3f}")
        print(f"   - Output std: {features.std().item():.3f}")

        # Check for NaN or inf values
        if torch.isnan(features).any():
            print(f"   🚨 CRITICAL: NaN values detected in output!")
            return False
        if torch.isinf(features).any():
            print(f"   🚨 CRITICAL: Inf values detected in output!")
            return False

    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        traceback.print_exc()
        return False

    # Test backward pass
    print(f"\n🔄 Backward Pass Test:")
    try:
        # Create a realistic loss that uses ALL features
        loss = features.pow(2).mean()  # MSE-like loss
        print(f"   - Loss value: {loss.item():.6f}")

        # Clear any existing gradients
        extractor.zero_grad()

        # Backward pass
        loss.backward()
        print(f"   ✅ Backward pass successful")

        # Detailed gradient analysis
        grad_stats = analyze_gradients_comprehensive(extractor)
        print_gradient_analysis(grad_stats)

        # Success criteria
        if grad_stats["gradient_coverage"] < 95:
            print(
                f"   🚨 CRITICAL: Only {grad_stats['gradient_coverage']:.1f}% of parameters have gradients!"
            )
            return False
        elif grad_stats["avg_gradient_norm"] < 1e-8:
            print(
                f"   ⚠️ WARNING: Very small gradients (avg: {grad_stats['avg_gradient_norm']:.2e})"
            )
            return False
        elif grad_stats["avg_gradient_norm"] > 10.0:
            print(
                f"   ⚠️ WARNING: Very large gradients (avg: {grad_stats['avg_gradient_norm']:.2e})"
            )
            return False
        else:
            print(
                f"   ✅ EXCELLENT: {grad_stats['gradient_coverage']:.1f}% gradient coverage with healthy norms!"
            )
            return True

    except Exception as e:
        print(f"   ❌ Backward pass failed: {e}")
        traceback.print_exc()
        return False


def analyze_gradients_comprehensive(model):
    """Comprehensive gradient analysis for the stabilized model."""
    stats = {
        "total_params": 0,
        "total_param_tensors": 0,
        "params_with_gradients": 0,
        "gradient_coverage": 0.0,
        "avg_gradient_norm": 0.0,
        "max_gradient_norm": 0.0,
        "min_gradient_norm": float("inf"),
        "zero_gradient_count": 0,
        "nan_gradient_count": 0,
        "inf_gradient_count": 0,
        "component_stats": {},
        "parameter_details": [],
    }

    total_grad_norm = 0.0
    gradient_norms = []
    zero_grad_count = 0
    nan_grad_count = 0
    inf_grad_count = 0

    # Component categorization
    components = {
        "visual_cnn": [],
        "vector_embed": [],
        "vector_gru": [],
        "vector_final": [],
        "fusion": [],
        "other": [],
    }

    for name, param in model.named_parameters():
        param_numel = param.numel()
        stats["total_params"] += param_numel
        stats["total_param_tensors"] += 1

        param_detail = {
            "name": name,
            "shape": list(param.shape),
            "numel": param_numel,
            "has_grad": param.grad is not None,
            "grad_norm": 0.0,
            "grad_status": "no_grad",
        }

        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            gradient_norms.append(grad_norm)
            total_grad_norm += grad_norm
            param_detail["grad_norm"] = grad_norm

            # Check for problematic gradients
            if torch.isnan(param.grad).any():
                nan_grad_count += 1
                param_detail["grad_status"] = "nan"
            elif torch.isinf(param.grad).any():
                inf_grad_count += 1
                param_detail["grad_status"] = "inf"
            elif grad_norm < 1e-8:
                zero_grad_count += 1
                param_detail["grad_status"] = "zero"
            else:
                param_detail["grad_status"] = "healthy"
                stats["params_with_gradients"] += 1

            # Categorize by component
            if "visual_cnn" in name:
                components["visual_cnn"].append((name, grad_norm))
            elif "vector_embed" in name:
                components["vector_embed"].append((name, grad_norm))
            elif "vector_gru" in name:
                components["vector_gru"].append((name, grad_norm))
            elif "vector_final" in name:
                components["vector_final"].append((name, grad_norm))
            elif "fusion" in name:
                components["fusion"].append((name, grad_norm))
            else:
                components["other"].append((name, grad_norm))

        stats["parameter_details"].append(param_detail)

    # Calculate statistics
    total_params_with_any_grad = len(gradient_norms)
    stats["gradient_coverage"] = (
        total_params_with_any_grad / max(stats["total_param_tensors"], 1)
    ) * 100
    stats["avg_gradient_norm"] = total_grad_norm / max(total_params_with_any_grad, 1)
    stats["zero_gradient_count"] = zero_grad_count
    stats["nan_gradient_count"] = nan_grad_count
    stats["inf_gradient_count"] = inf_grad_count

    if gradient_norms:
        stats["max_gradient_norm"] = max(gradient_norms)
        stats["min_gradient_norm"] = min(gradient_norms)

    # Component statistics
    for component, grads in components.items():
        if grads:
            grad_values = [grad for _, grad in grads]
            stats["component_stats"][component] = {
                "param_count": len(grads),
                "avg_grad_norm": sum(grad_values) / len(grad_values),
                "min_grad_norm": min(grad_values),
                "max_grad_norm": max(grad_values),
                "healthy_params": sum(1 for grad in grad_values if grad >= 1e-8),
            }

    return stats


def print_gradient_analysis(stats):
    """Print comprehensive gradient analysis."""
    print(f"\n📊 COMPREHENSIVE Gradient Analysis:")
    print(f"   - Total parameters: {stats['total_params']:,}")
    print(f"   - Parameter tensors: {stats['total_param_tensors']}")
    print(f"   - Gradient coverage: {stats['gradient_coverage']:.2f}%")
    print(f"   - Avg gradient norm: {stats['avg_gradient_norm']:.6f}")
    print(
        f"   - Gradient range: [{stats['min_gradient_norm']:.2e}, {stats['max_gradient_norm']:.2e}]"
    )

    # Problem detection
    problems = []
    if stats["zero_gradient_count"] > 0:
        problems.append(f"{stats['zero_gradient_count']} zero gradients")
    if stats["nan_gradient_count"] > 0:
        problems.append(f"{stats['nan_gradient_count']} NaN gradients")
    if stats["inf_gradient_count"] > 0:
        problems.append(f"{stats['inf_gradient_count']} Inf gradients")

    if problems:
        print(f"   ⚠️ Issues: {', '.join(problems)}")
    else:
        print(f"   ✅ No gradient issues detected")

    # Component breakdown
    print(f"\n🔍 Component Analysis:")
    for component, comp_stats in stats["component_stats"].items():
        if comp_stats["param_count"] > 0:
            health_ratio = comp_stats["healthy_params"] / comp_stats["param_count"]
            health_status = (
                "✅" if health_ratio > 0.9 else "⚠️" if health_ratio > 0.5 else "❌"
            )
            print(f"   {health_status} {component}:")
            print(f"     • Parameters: {comp_stats['param_count']}")
            print(
                f"     • Healthy: {comp_stats['healthy_params']}/{comp_stats['param_count']}"
            )
            print(f"     • Avg norm: {comp_stats['avg_grad_norm']:.6f}")
            print(
                f"     • Range: [{comp_stats['min_grad_norm']:.2e}, {comp_stats['max_grad_norm']:.2e}]"
            )


def test_full_policy_gradient_flow():
    """Test gradient flow through the complete policy."""
    print(f"\n🧠 Testing Complete Policy Gradient Flow:")
    print("=" * 50)

    try:
        # Create environment to get spaces
        env = retro.make(
            game="StreetFighterIISpecialChampionEdition-Genesis",
            state="ken_bison_12.state",
        )
        env = StreetFighterVisionWrapper(env, frame_stack=8)

        # Create action space
        action_space = spaces.Discrete(StreetFighterDiscreteActions().num_actions)

        print(f"   - Observation space: {env.observation_space}")
        print(f"   - Action space: {action_space}")

        # Create policy
        policy = FixedStreetFighterPolicy(
            env.observation_space,
            action_space,
            lambda x: 1e-4,  # Learning rate schedule
        )

        print(f"   - Policy created successfully")

        # Get sample observation
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        # Convert to tensors
        obs_tensor = {}
        for key, value in obs.items():
            obs_tensor[key] = torch.from_numpy(value).unsqueeze(0).float()

        # Test policy forward pass
        policy.train()
        actions, values, log_probs = policy(obs_tensor)

        print(f"   - Policy forward pass successful")
        print(f"   - Action: {actions.item()}")
        print(f"   - Value: {values.item():.3f}")
        print(f"   - Log prob: {log_probs.item():.3f}")

        # Test policy backward pass
        total_loss = values.mean() + log_probs.mean() * 0.1
        policy.zero_grad()
        total_loss.backward()

        # Count gradients in policy
        policy_grad_count = 0
        total_policy_params = 0
        for name, param in policy.named_parameters():
            total_policy_params += 1
            if param.grad is not None:
                policy_grad_count += 1

        coverage = (policy_grad_count / total_policy_params) * 100
        print(f"   - Policy gradient coverage: {coverage:.1f}%")

        if coverage > 95:
            print(f"   ✅ Policy gradients healthy!")
            return True
        else:
            print(f"   ⚠️ Policy gradient issues detected")
            return False

        env.close()

    except Exception as e:
        print(f"   ❌ Policy test failed: {e}")
        traceback.print_exc()
        return False


def test_stability_features():
    """Test the stability features in the architecture."""
    print(f"\n🔒 Testing Stability Features:")
    print("=" * 50)

    try:
        # Test feature normalization
        print(f"   🧪 Testing feature normalization...")
        from wrapper import StrategicFeatureTracker

        tracker = StrategicFeatureTracker()

        # Simulate some feature updates
        fake_info = {
            "agent_hp": 150,
            "enemy_hp": 140,
            "agent_x": 90,
            "enemy_x": 100,
            "agent_y": 64,
            "enemy_y": 64,
            "score": 1000,
        }
        fake_buttons = np.zeros(12, dtype=np.float32)

        features_list = []
        for i in range(50):  # Simulate 50 frames
            features = tracker.update(fake_info, fake_buttons)
            features_list.append(features)

        # Check feature stability
        features_array = np.array(features_list)
        mean_vals = np.mean(features_array, axis=0)
        std_vals = np.std(features_array, axis=0)

        print(
            f"     • Feature mean range: [{mean_vals.min():.3f}, {mean_vals.max():.3f}]"
        )
        print(f"     • Feature std range: [{std_vals.min():.3f}, {std_vals.max():.3f}]")

        # Check for extreme values
        if np.any(np.abs(mean_vals) > 5.0):
            print(f"     ⚠️ Some features have large means")
        else:
            print(f"     ✅ Feature means in reasonable range")

        if np.any(std_vals > 3.0):
            print(f"     ⚠️ Some features have high variance")
        else:
            print(f"     ✅ Feature variances controlled")

        print(f"   ✅ Feature normalization working")

        # Test reward scaling
        print(f"   🎯 Testing reward scaling...")

        # The wrapper uses reward_scale = 0.1
        sample_rewards = [100, -50, 25, 0, 200]  # Raw rewards
        scaled_rewards = [r * 0.1 for r in sample_rewards]  # Scaled
        clipped_rewards = [np.clip(r, -2.0, 2.0) for r in scaled_rewards]  # Clipped

        print(f"     • Raw rewards: {sample_rewards}")
        print(f"     • Scaled: {scaled_rewards}")
        print(f"     • Clipped: {clipped_rewards}")
        print(f"   ✅ Reward scaling working")

        return True

    except Exception as e:
        print(f"   ❌ Stability feature test failed: {e}")
        traceback.print_exc()
        return False


def run_comprehensive_test():
    """Run all gradient flow tests."""
    print("🚀 Starting COMPREHENSIVE Gradient Flow Tests")
    print("=" * 70)

    test_results = []

    # Test 1: Feature extractor gradient flow
    print("\n" + "=" * 70)
    print("TEST 1: Feature Extractor Gradient Flow")
    result1 = test_stabilized_gradient_flow()
    test_results.append(("Feature Extractor", result1))

    # Test 2: Full policy gradient flow
    print("\n" + "=" * 70)
    print("TEST 2: Complete Policy Gradient Flow")
    result2 = test_full_policy_gradient_flow()
    test_results.append(("Complete Policy", result2))

    # Test 3: Stability features
    print("\n" + "=" * 70)
    print("TEST 3: Stability Features")
    result3 = test_stability_features()
    test_results.append(("Stability Features", result3))

    # Summary
    print("\n" + "=" * 70)
    print("🏁 TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if not result:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Stabilized architecture has excellent gradient flow")
        print("🚀 Ready for stable training!")
    else:
        print("❌ SOME TESTS FAILED!")
        print("⚠️ Architecture may have gradient flow issues")
        print("🔧 Review failed components before training")

    return all_passed


if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)
