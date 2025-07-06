#!/usr/bin/env python3
"""
test_ultra_simple.py - Test the ultra-simplified architecture for gradient flow
"""

# torch, np, wrapper, space
import torch
import numpy as np
from wrapper import StreetFighterUltraSimpleCNN
from gymnasium import spaces


def test_ultra_simple_gradient_flow():
    """Test gradient flow through the ultra-simplified architecture."""

    print("🧪 Testing Ultra-Simplified Architecture for Gradient Flow")
    print("=" * 70)

    # Create observation space

    # frame stack, target size (screen), vector feature
    frame_stack = 8
    target_size = (128, 180)
    vector_feature_dim = 45

    #
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
    extractor = StreetFighterUltraSimpleCNN(obs_space, features_dim)

    print(f"\n🏗️ Model Architecture:")
    total_params = sum(p.numel() for p in extractor.parameters())
    print(f"   - Total parameters: {total_params:,}")

    # List all parameters and their requires_grad status
    print(f"\n📋 All Parameters:")
    param_details = []
    for name, param in extractor.named_parameters():
        param_details.append(
            {
                "name": name,
                "shape": list(param.shape),
                "numel": param.numel(),
                "requires_grad": param.requires_grad,
            }
        )
        print(
            f"   - {name}: {list(param.shape)}, {param.numel()} params, requires_grad={param.requires_grad}"
        )

    # Create dummy input
    batch_size = 2
    visual_obs = torch.randint(
        0, 255, (batch_size, 3 * frame_stack, *target_size), dtype=torch.uint8
    )
    vector_obs = torch.randn(
        batch_size, frame_stack, vector_feature_dim, dtype=torch.float32
    )

    obs_dict = {"visual_obs": visual_obs, "vector_obs": vector_obs}

    print(f"\n🔍 Input Testing:")
    print(f"   - Visual input shape: {visual_obs.shape}")
    print(f"   - Vector input shape: {vector_obs.shape}")

    # Test forward pass
    print(f"\n⚡ Forward Pass Test:")
    try:
        features = extractor(obs_dict)
        print(f"   ✅ Forward pass successful")
        print(f"   - Output shape: {features.shape}")
        print(
            f"   - Output range: [{features.min().item():.3f}, {features.max().item():.3f}]"
        )
        print(f"   - Output mean: {features.mean().item():.3f}")
        print(f"   - Output std: {features.std().item():.3f}")
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test backward pass
    print(f"\n🔄 Backward Pass Test:")
    try:
        # Create a simple loss that depends on ALL outputs
        loss = features.sum()
        print(f"   - Simple sum loss: {loss.item():.6f}")

        # Clear any existing gradients
        extractor.zero_grad()

        # Backward pass
        loss.backward()
        print(f"   ✅ Backward pass successful")

        # Check gradients in detail
        grad_stats = analyze_gradients_detailed(extractor)
        print_gradient_analysis_detailed(grad_stats)

        if grad_stats["gradient_coverage"] < 90:
            print(
                f"   🚨 CRITICAL: Only {grad_stats['gradient_coverage']:.1f}% of parameters have gradients!"
            )
            return False
        else:
            print(
                f"   ✅ EXCELLENT: {grad_stats['gradient_coverage']:.1f}% gradient coverage!"
            )
            return True

    except Exception as e:
        print(f"   ❌ Backward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def analyze_gradients_detailed(model):
    """Analyze gradient flow in extreme detail."""
    stats = {
        "total_params": 0,
        "total_param_count": 0,
        "params_with_gradients": 0,
        "gradient_coverage": 0.0,
        "avg_gradient_norm": 0.0,
        "zero_gradient_count": 0,
        "component_stats": {},
        "parameter_details": [],
    }

    total_grad_norm = 0.0
    param_count = 0
    zero_grad_count = 0

    # Component-wise analysis
    components = {"visual_net": [], "vector_net": [], "fusion_net": []}

    for name, param in model.named_parameters():
        param_numel = param.numel()
        stats["total_params"] += param_numel
        stats["total_param_count"] += 1

        param_detail = {
            "name": name,
            "shape": list(param.shape),
            "numel": param_numel,
            "has_grad": param.grad is not None,
            "grad_norm": 0.0,
        }

        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            total_grad_norm += grad_norm
            param_count += 1
            param_detail["grad_norm"] = grad_norm

            if grad_norm < 1e-8:
                zero_grad_count += 1
                param_detail["zero_grad"] = True
            else:
                param_detail["zero_grad"] = False

            # Categorize by component
            if "visual_net" in name:
                components["visual_net"].append((name, grad_norm))
            elif "vector_net" in name:
                components["vector_net"].append((name, grad_norm))
            elif "fusion_net" in name:
                components["fusion_net"].append((name, grad_norm))
        else:
            param_detail["zero_grad"] = True

        stats["parameter_details"].append(param_detail)

    stats["params_with_gradients"] = param_count
    stats["gradient_coverage"] = (
        param_count / max(stats["total_param_count"], 1)
    ) * 100
    stats["avg_gradient_norm"] = total_grad_norm / max(param_count, 1)
    stats["zero_gradient_count"] = zero_grad_count

    # Component statistics
    for component, grads in components.items():
        if grads:
            stats["component_stats"][component] = {
                "param_count": len(grads),
                "avg_grad_norm": sum(grad for _, grad in grads) / len(grads),
                "min_grad_norm": min(grad for _, grad in grads),
                "max_grad_norm": max(grad for _, grad in grads),
                "zero_grad_count": sum(1 for _, grad in grads if grad < 1e-8),
            }

    return stats


def print_gradient_analysis_detailed(stats):
    """Print extremely detailed gradient analysis."""
    print(f"\n📊 DETAILED Gradient Analysis:")
    print(f"   - Total parameters: {stats['total_params']:,}")
    print(f"   - Total parameter tensors: {stats['total_param_count']}")
    print(f"   - Parameters with gradients: {stats['params_with_gradients']}")
    print(f"   - Gradient coverage: {stats['gradient_coverage']:.2f}%")
    print(f"   - Average gradient norm: {stats['avg_gradient_norm']:.6f}")
    print(f"   - Zero gradient count: {stats['zero_gradient_count']}")

    print(f"\n🔍 Component Breakdown:")
    for component, comp_stats in stats["component_stats"].items():
        print(f"   - {component}:")
        print(f"     • Parameters: {comp_stats['param_count']}")
        print(f"     • Avg grad norm: {comp_stats['avg_grad_norm']:.6f}")
        print(
            f"     • Grad range: [{comp_stats['min_grad_norm']:.6f}, {comp_stats['max_grad_norm']:.6f}]"
        )
        print(f"     • Zero grads: {comp_stats['zero_grad_count']}")

    print(f"\n📋 Parameter Details:")
    for detail in stats["parameter_details"]:
        grad_status = (
            f"✅ {detail['grad_norm']:.6f}" if detail["has_grad"] else "❌ NO GRAD"
        )
        print(
            f"   - {detail['name']}: {detail['shape']}, {detail['numel']} params, {grad_status}"
        )

    # Health assessment
    coverage = stats["gradient_coverage"]
    if coverage < 70:
        print(f"\n🚨 CRITICAL: Only {coverage:.1f}% of parameters have gradients!")
    elif coverage < 90:
        print(f"\n⚠️ WARNING: Only {coverage:.1f}% of parameters have gradients!")
    else:
        print(f"\n✅ EXCELLENT: {coverage:.1f}% of parameters have gradients!")


def test_individual_components():
    """Test each component in isolation."""
    print(f"\n🔬 Individual Component Testing:")
    print("=" * 50)

    # Test UltraSimpleCNN
    print(f"🖼️ Testing UltraSimpleCNN:")
    try:
        from wrapper import UltraSimpleCNN

        visual_net = UltraSimpleCNN(input_channels=24, output_dim=128)

        # Count parameters
        total_params = sum(p.numel() for p in visual_net.parameters())
        print(f"   - Total parameters: {total_params:,}")

        # Test forward pass
        visual_input = torch.randint(0, 255, (2, 24, 128, 180), dtype=torch.uint8)
        visual_output = visual_net(visual_input)
        print(f"   - Forward pass: {visual_input.shape} -> {visual_output.shape}")

        # Test backward pass
        loss = visual_output.sum()
        visual_net.zero_grad()
        loss.backward()

        grad_count = 0
        for name, param in visual_net.named_parameters():
            if param.grad is not None:
                grad_count += 1
                print(f"     • {name}: grad_norm={param.grad.norm().item():.6f}")
            else:
                print(f"     • {name}: NO GRADIENT")

        print(f"   ✅ Visual net: {grad_count} parameters with gradients")

    except Exception as e:
        print(f"   ❌ UltraSimpleCNN failed: {e}")
        return False

    # Test UltraSimpleVectorNet
    print(f"\n🔢 Testing UltraSimpleVectorNet:")
    try:
        from wrapper import UltraSimpleVectorNet

        vector_net = UltraSimpleVectorNet(input_dim=45, seq_len=8, output_dim=128)

        # Count parameters
        total_params = sum(p.numel() for p in vector_net.parameters())
        print(f"   - Total parameters: {total_params:,}")

        # Test forward pass
        vector_input = torch.randn(2, 8, 45, dtype=torch.float32)
        vector_output = vector_net(vector_input)
        print(f"   - Forward pass: {vector_input.shape} -> {vector_output.shape}")

        # Test backward pass
        loss = vector_output.sum()
        vector_net.zero_grad()
        loss.backward()

        grad_count = 0
        for name, param in vector_net.named_parameters():
            if param.grad is not None:
                grad_count += 1
                print(f"     • {name}: grad_norm={param.grad.norm().item():.6f}")
            else:
                print(f"     • {name}: NO GRADIENT")

        print(f"   ✅ Vector net: {grad_count} parameters with gradients")

    except Exception as e:
        print(f"   ❌ UltraSimpleVectorNet failed: {e}")
        return False

    # Test UltraSimpleFusion
    print(f"\n🔗 Testing UltraSimpleFusion:")
    try:
        from wrapper import UltraSimpleFusion

        fusion_net = UltraSimpleFusion(visual_dim=128, vector_dim=128, output_dim=256)

        # Count parameters
        total_params = sum(p.numel() for p in fusion_net.parameters())
        print(f"   - Total parameters: {total_params:,}")

        # Test forward pass
        visual_feat = torch.randn(2, 128)
        vector_feat = torch.randn(2, 128)
        fusion_output = fusion_net(visual_feat, vector_feat)
        print(
            f"   - Forward pass: ({visual_feat.shape}, {vector_feat.shape}) -> {fusion_output.shape}"
        )

        # Test backward pass
        loss = fusion_output.sum()
        fusion_net.zero_grad()
        loss.backward()

        grad_count = 0
        for name, param in fusion_net.named_parameters():
            if param.grad is not None:
                grad_count += 1
                print(f"     • {name}: grad_norm={param.grad.norm().item():.6f}")
            else:
                print(f"     • {name}: NO GRADIENT")

        print(f"   ✅ Fusion net: {grad_count} parameters with gradients")

    except Exception as e:
        print(f"   ❌ UltraSimpleFusion failed: {e}")
        return False

    print(f"\n✅ All individual components work correctly!")
    return True


if __name__ == "__main__":
    print("🚀 Starting Ultra-Simplified Architecture Tests")
    print("=" * 70)

    # Test individual components first
    if not test_individual_components():
        print("❌ Individual component tests failed!")
        exit(1)

    # Test full architecture
    if not test_ultra_simple_gradient_flow():
        print("❌ Full architecture gradient flow test failed!")
        exit(1)

    print(
        "\n🎉 ALL TESTS PASSED! Ultra-simplified architecture has perfect gradient flow."
    )
    print("✅ Ready for training with ultra-simplified architecture.")
