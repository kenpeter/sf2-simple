#!/usr/bin/env python3
"""
test_flow.py - COMPREHENSIVE gradient flow test for Street Fighter AI
Tests gradient propagation through the ENTIRE network architecture
"""

import torch
import numpy as np
from gymnasium import spaces
import traceback
from wrapper import (
    FixedStreetFighterCNN,
    FixedStreetFighterPolicy,
    StreetFighterVisionWrapper,
    StreetFighterDiscreteActions,
)
import retro


def test_single_network_gradient_flow():
    """Test gradient flow within the FixedStreetFighterCNN feature extractor."""

    print("🧪 TEST 1: Single Network Internal Gradient Flow")
    print("=" * 70)
    print("🎯 Testing: How gradients flow WITHIN the CNN feature extractor")
    print("   • Visual CNN path: Input → Conv1 → Conv2 → Conv3 → Conv4 → Flatten")
    print("   • Vector path: Input → Embed → GRU → Final")
    print("   • Fusion path: [Visual + Vector] → Fusion → Output")

    # Create observation space
    frame_stack = 8
    target_size = (200, 256)  # Full size frames
    vector_feature_dim = 52  # With bait-punish features

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

    print(f"\n📏 Network Input Specifications:")
    print(
        f"   - Visual frames: {obs_space['visual_obs'].shape} ({np.prod(obs_space['visual_obs'].shape):,} elements)"
    )
    print(
        f"   - Vector features: {obs_space['vector_obs'].shape} ({np.prod(obs_space['vector_obs'].shape):,} elements)"
    )

    # Create the feature extractor
    features_dim = 256
    extractor = FixedStreetFighterCNN(obs_space, features_dim)
    extractor.train()

    print(f"\n🏗️ Network Architecture Analysis:")
    total_params = sum(p.numel() for p in extractor.parameters())
    trainable_params = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Memory footprint: ~{total_params * 4 / 1024 / 1024:.1f} MB")

    # Create realistic test data
    batch_size = 2
    visual_obs = torch.randint(
        0, 255, (batch_size, 3 * frame_stack, *target_size), dtype=torch.uint8
    )
    vector_obs = (
        torch.randn(batch_size, frame_stack, vector_feature_dim, dtype=torch.float32)
        * 0.5
    )

    obs_dict = {"visual_obs": visual_obs, "vector_obs": vector_obs}

    print(f"\n⚡ Forward Pass Analysis:")
    print(
        f"   - Input visual range: [{visual_obs.min().item()}, {visual_obs.max().item()}]"
    )
    print(
        f"   - Input vector range: [{vector_obs.min().item():.3f}, {vector_obs.max().item():.3f}]"
    )

    try:
        # Forward pass with intermediate outputs
        features = extractor(obs_dict)

        print(f"   ✅ Forward pass successful")
        print(f"   - Output shape: {features.shape}")
        print(
            f"   - Output range: [{features.min().item():.3f}, {features.max().item():.3f}]"
        )
        print(
            f"   - Output statistics: mean={features.mean().item():.3f}, std={features.std().item():.3f}"
        )

        # Check for numerical issues
        if torch.isnan(features).any():
            print(f"   🚨 CRITICAL: NaN values in output!")
            return False
        if torch.isinf(features).any():
            print(f"   🚨 CRITICAL: Infinite values in output!")
            return False

    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        traceback.print_exc()
        return False

    print(f"\n🔄 Backward Pass Analysis:")
    print(f"   🎯 Testing: Gradient propagation from output back to input")

    try:
        # Create loss that depends on ALL output features
        loss = features.pow(2).mean() + features.abs().mean() * 0.1
        print(f"   - Loss value: {loss.item():.6f}")

        # Clear gradients
        extractor.zero_grad()

        # Backward pass
        loss.backward()
        print(f"   ✅ Backward pass successful")

        # Analyze gradients at each network component
        gradient_analysis = analyze_component_gradients(extractor)
        print_component_analysis(gradient_analysis)

        # Success criteria
        if gradient_analysis["overall_coverage"] < 95:
            print(
                f"   🚨 CRITICAL: Only {gradient_analysis['overall_coverage']:.1f}% gradient coverage!"
            )
            return False
        elif gradient_analysis["avg_gradient_norm"] < 1e-8:
            print(
                f"   ⚠️ WARNING: Vanishing gradients (avg: {gradient_analysis['avg_gradient_norm']:.2e})"
            )
            return False
        elif gradient_analysis["avg_gradient_norm"] > 10.0:
            print(
                f"   ⚠️ WARNING: Exploding gradients (avg: {gradient_analysis['avg_gradient_norm']:.2e})"
            )
            return False
        else:
            print(f"   ✅ EXCELLENT: Healthy gradient flow throughout network!")
            return True

    except Exception as e:
        print(f"   ❌ Backward pass failed: {e}")
        traceback.print_exc()
        return False


def test_multi_network_gradient_flow():
    """Test gradient flow BETWEEN multiple networks (Policy = CNN + Actor + Critic)."""

    print("\n🧠 TEST 2: Multi-Network Gradient Flow")
    print("=" * 70)
    print("🎯 Testing: How gradients flow BETWEEN networks in the policy")
    print("   • CNN Feature Extractor → MLP Extractor → Actor Network")
    print("   • CNN Feature Extractor → MLP Extractor → Value Network")
    print("   • Policy Loss ← Actor Loss + Value Loss")

    try:
        # Create environment to get proper spaces
        env = retro.make(
            "StreetFighterIISpecialChampionEdition-Genesis", state="ken_bison_12.state"
        )
        env = StreetFighterVisionWrapper(env, frame_stack=8)
        action_space = spaces.Discrete(StreetFighterDiscreteActions().num_actions)

        print(f"\n📏 Policy Network Specifications:")
        print(f"   - Observation space: {env.observation_space}")
        print(f"   - Action space: {action_space}")

        # Create complete policy (CNN + Actor + Critic)
        policy = FixedStreetFighterPolicy(
            env.observation_space,
            action_space,
            lambda x: 1e-4,  # Learning rate schedule
        )
        policy.train()

        print(f"\n🏗️ Complete Policy Architecture:")

        # Count parameters in each component
        feature_params = sum(p.numel() for p in policy.features_extractor.parameters())
        mlp_params = sum(p.numel() for p in policy.mlp_extractor.parameters())
        actor_params = sum(p.numel() for p in policy.action_net.parameters())
        critic_params = sum(p.numel() for p in policy.value_net.parameters())
        total_params = sum(p.numel() for p in policy.parameters())

        print(f"   - Feature Extractor: {feature_params:,} parameters")
        print(f"   - MLP Extractor: {mlp_params:,} parameters")
        print(f"   - Actor Network: {actor_params:,} parameters")
        print(f"   - Critic Network: {critic_params:,} parameters")
        print(f"   - Total Policy: {total_params:,} parameters")

        # Get sample observation
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        # Convert to tensors
        obs_tensor = {}
        for key, value in obs.items():
            obs_tensor[key] = torch.from_numpy(value).unsqueeze(0).float()

        print(f"\n⚡ Multi-Network Forward Pass:")

        # Test complete policy forward pass
        actions, values, log_probs = policy(obs_tensor)

        print(f"   ✅ Complete policy forward successful")
        print(f"   - Action output: {actions.item()}")
        print(f"   - Value output: {values.item():.3f}")
        print(f"   - Log probability: {log_probs.item():.3f}")

        # Check for numerical issues
        if torch.isnan(values).any() or torch.isnan(log_probs).any():
            print(f"   🚨 CRITICAL: NaN in policy outputs!")
            return False

        print(f"\n🔄 Multi-Network Backward Pass:")
        print(f"   🎯 Testing: Gradients from Actor & Critic losses back to CNN")

        # Create realistic PPO-style losses
        actor_loss = -log_probs.mean()  # Policy gradient loss
        critic_loss = values.pow(2).mean()  # Value function loss
        total_loss = actor_loss + 0.5 * critic_loss  # Combined loss

        print(f"   - Actor loss: {actor_loss.item():.6f}")
        print(f"   - Critic loss: {critic_loss.item():.6f}")
        print(f"   - Total loss: {total_loss.item():.6f}")

        # Clear gradients
        policy.zero_grad()

        # Backward pass
        total_loss.backward()
        print(f"   ✅ Multi-network backward pass successful")

        # Analyze gradients across all components
        multi_net_analysis = analyze_multi_network_gradients(policy)
        print_multi_network_analysis(multi_net_analysis)

        # Multi-network success criteria
        if multi_net_analysis["feature_extractor_coverage"] < 90:
            print(f"   🚨 CRITICAL: Feature extractor not receiving gradients!")
            return False
        elif multi_net_analysis["actor_critic_balance"] < 0.1:
            print(f"   ⚠️ WARNING: Actor/Critic gradient imbalance!")
            return False
        else:
            print(f"   ✅ EXCELLENT: Healthy gradients flowing between all networks!")
            return True

        env.close()

    except Exception as e:
        print(f"   ❌ Multi-network test failed: {e}")
        traceback.print_exc()
        return False


def test_end_to_end_gradient_flow():
    """Test gradient flow from final loss all the way back to input pixels."""

    print("\n🎯 TEST 3: End-to-End Gradient Flow")
    print("=" * 70)
    print("🎯 Testing: Complete gradient path from loss to input pixels")
    print("   • Loss Function → Policy → CNN → Raw Visual Input")
    print("   • Verifying gradients reach the very first layer")

    try:
        # Create environment
        env = retro.make(
            "StreetFighterIISpecialChampionEdition-Genesis", state="ken_bison_12.state"
        )
        env = StreetFighterVisionWrapper(env, frame_stack=8)
        action_space = spaces.Discrete(StreetFighterDiscreteActions().num_actions)

        # Create policy
        policy = FixedStreetFighterPolicy(
            env.observation_space, action_space, lambda x: 1e-4
        )
        policy.train()

        # Get observation and make inputs require gradients
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        # Convert to tensors with gradient tracking
        obs_tensor = {}
        for key, value in obs.items():
            tensor = torch.from_numpy(value).unsqueeze(0).float()
            tensor.requires_grad_(True)  # KEY: Enable gradient tracking for inputs
            obs_tensor[key] = tensor

        print(f"\n⚡ End-to-End Forward Pass:")
        print(
            f"   - Input tensors require gradients: {all(t.requires_grad for t in obs_tensor.values())}"
        )

        # Forward pass
        actions, values, log_probs = policy(obs_tensor)

        # Create loss
        loss = values.mean().pow(2) + log_probs.mean().abs()
        print(f"   - Final loss: {loss.item():.6f}")

        print(f"\n🔄 End-to-End Backward Pass:")
        print(f"   🎯 Testing: Can gradients reach the input pixels?")

        # Backward pass
        loss.backward()

        # Check if gradients reached the inputs
        input_gradients_exist = {}
        for key, tensor in obs_tensor.items():
            has_grad = tensor.grad is not None
            input_gradients_exist[key] = has_grad
            if has_grad:
                grad_norm = tensor.grad.norm().item()
                print(f"   ✅ {key} input gradients: norm = {grad_norm:.6f}")
            else:
                print(f"   ❌ {key} input gradients: MISSING")

        # Success criteria
        if all(input_gradients_exist.values()):
            print(f"   ✅ EXCELLENT: Gradients successfully reached ALL inputs!")
            print(f"   🎉 Complete end-to-end gradient flow verified!")
            return True
        else:
            missing = [k for k, v in input_gradients_exist.items() if not v]
            print(f"   🚨 CRITICAL: Missing gradients for: {missing}")
            return False

        env.close()

    except Exception as e:
        print(f"   ❌ End-to-end test failed: {e}")
        traceback.print_exc()
        return False


def analyze_component_gradients(model):
    """Analyze gradients within each component of the CNN."""

    components = {
        "visual_cnn": [],
        "vector_embed": [],
        "vector_gru": [],
        "vector_final": [],
        "fusion": [],
    }

    total_params = 0
    params_with_grad = 0
    total_grad_norm = 0.0

    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None:
            params_with_grad += 1
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm

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

    return {
        "overall_coverage": (params_with_grad / max(total_params, 1)) * 100,
        "avg_gradient_norm": total_grad_norm / max(params_with_grad, 1),
        "components": components,
        "total_params": total_params,
        "params_with_grad": params_with_grad,
    }


def analyze_multi_network_gradients(policy):
    """Analyze gradients across multiple networks in the policy."""

    networks = {
        "feature_extractor": policy.features_extractor,
        "mlp_extractor": policy.mlp_extractor,
        "actor_net": policy.action_net,
        "value_net": policy.value_net,
    }

    network_stats = {}

    for net_name, network in networks.items():
        params_with_grad = 0
        total_params = 0
        total_grad_norm = 0.0

        for param in network.parameters():
            total_params += 1
            if param.grad is not None:
                params_with_grad += 1
                total_grad_norm += param.grad.norm().item()

        coverage = (params_with_grad / max(total_params, 1)) * 100
        avg_norm = total_grad_norm / max(params_with_grad, 1)

        network_stats[net_name] = {
            "coverage": coverage,
            "avg_grad_norm": avg_norm,
            "total_params": total_params,
            "params_with_grad": params_with_grad,
        }

    # Calculate balance between actor and critic
    actor_norm = network_stats["actor_net"]["avg_grad_norm"]
    critic_norm = network_stats["value_net"]["avg_grad_norm"]
    balance = min(actor_norm, critic_norm) / max(actor_norm, critic_norm, 1e-8)

    return {
        "networks": network_stats,
        "feature_extractor_coverage": network_stats["feature_extractor"]["coverage"],
        "actor_critic_balance": balance,
    }


def print_component_analysis(analysis):
    """Print analysis of single network components."""

    print(f"\n📊 Single Network Component Analysis:")
    print(f"   - Overall coverage: {analysis['overall_coverage']:.1f}%")
    print(f"   - Average gradient norm: {analysis['avg_gradient_norm']:.6f}")

    for component, gradients in analysis["components"].items():
        if gradients:
            grad_norms = [grad for _, grad in gradients]
            avg_norm = sum(grad_norms) / len(grad_norms)
            print(
                f"   - {component}: {len(gradients)} params, avg norm = {avg_norm:.6f}"
            )


def print_multi_network_analysis(analysis):
    """Print analysis of multi-network gradients."""

    print(f"\n📊 Multi-Network Gradient Analysis:")

    for net_name, stats in analysis["networks"].items():
        status = (
            "✅" if stats["coverage"] > 90 else "⚠️" if stats["coverage"] > 50 else "❌"
        )
        print(f"   {status} {net_name}:")
        print(f"     • Coverage: {stats['coverage']:.1f}%")
        print(f"     • Avg gradient norm: {stats['avg_grad_norm']:.6f}")
        print(f"     • Parameters: {stats['params_with_grad']}/{stats['total_params']}")

    print(f"\n🎯 Cross-Network Analysis:")
    print(
        f"   - Feature extractor receiving gradients: {analysis['feature_extractor_coverage']:.1f}%"
    )
    print(f"   - Actor/Critic gradient balance: {analysis['actor_critic_balance']:.3f}")


def run_comprehensive_gradient_tests():
    """Run all gradient flow tests."""

    print("🚀 COMPREHENSIVE GRADIENT FLOW ANALYSIS")
    print("=" * 70)
    print("🔍 Testing THREE types of gradient flow:")
    print("   1. Single Network: Within CNN feature extractor")
    print("   2. Multi-Network: Between Policy components")
    print("   3. End-to-End: From loss to input pixels")

    results = []

    # Test 1: Single network
    result1 = test_single_network_gradient_flow()
    results.append(("Single Network Flow", result1))

    # Test 2: Multi-network
    result2 = test_multi_network_gradient_flow()
    results.append(("Multi-Network Flow", result2))

    # Test 3: End-to-end
    result3 = test_end_to_end_gradient_flow()
    results.append(("End-to-End Flow", result3))

    # Summary
    print("\n" + "=" * 70)
    print("🏁 GRADIENT FLOW TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if not result:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL GRADIENT FLOW TESTS PASSED!")
        print("✅ Neural network has excellent gradient propagation")
        print("🚀 Ready for stable, efficient training!")
        print("\n💡 What this means:")
        print("   • Gradients flow smoothly from loss to inputs")
        print("   • No vanishing or exploding gradient problems")
        print("   • All network components will learn effectively")
        print("   • Training should be stable and converge well")
    else:
        print("❌ GRADIENT FLOW ISSUES DETECTED!")
        print("⚠️ Neural network has gradient propagation problems")
        print("🔧 Must fix gradient flow before stable training")
        print("\n💡 Potential issues:")
        print("   • Some layers may not learn (vanishing gradients)")
        print("   • Training instability (exploding gradients)")
        print("   • Poor convergence or learning performance")

    return all_passed


if __name__ == "__main__":
    success = run_comprehensive_gradient_tests()
    exit(0 if success else 1)
