#!/usr/bin/env python3
"""
🥊 Demo: Qwen agent playing Street Fighter 2
Shows how to integrate Qwen reasoning with the existing wrapper
"""

import time
from wrapper import make_env
from qwen_agent import QwenStreetFighterAgent


def demo_qwen_gameplay(model_path: str = "Qwen/Qwen3-4B-Instruct-2507", 
                       episodes: int = 3, 
                       render: bool = True,
                       verbose: bool = True):
    """
    Demo Qwen agent playing Street Fighter 2
    
    Args:
        model_path: Path to Qwen model
        episodes: Number of episodes to play
        render: Whether to render the game
        verbose: Whether to print detailed reasoning
    """
    print("🥊 Starting Qwen Street Fighter Demo")
    print(f"Model: {model_path}")
    print(f"Episodes: {episodes}")
    print("-" * 50)
    
    # Create environment and agent
    env = make_env()
    agent = QwenStreetFighterAgent(model_path)
    
    wins = 0
    total_rewards = []
    
    try:
        for episode in range(episodes):
            print(f"\n🎮 Episode {episode + 1}/{episodes}")
            print("=" * 30)
            
            # Reset environment and agent
            obs, info = env.reset()
            agent.reset()
            
            episode_reward = 0
            steps = 0
            max_steps = 1000  # Prevent infinite episodes
            
            while steps < max_steps:
                if render:
                    env.render()
                    time.sleep(0.05)  # Slow down for visibility
                
                # Get action from Qwen agent
                action, reasoning = agent.get_action(info, verbose=verbose)
                
                # Take step in environment
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
                
                # Check if episode ended
                if done or truncated:
                    agent_won = info.get("agent_won", False)
                    if agent_won:
                        wins += 1
                        print(f"🏆 Won episode {episode + 1}!")
                    else:
                        print(f"💀 Lost episode {episode + 1}")
                    
                    print(f"Steps: {steps}, Reward: {episode_reward:.2f}")
                    break
            
            total_rewards.append(episode_reward)
            
            if not render:
                print(f"Episode {episode + 1} completed - Reward: {episode_reward:.2f}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    
    finally:
        env.close()
    
    # Summary
    print("\n" + "="*50)
    print("🏁 DEMO SUMMARY")
    print("="*50)
    print(f"Episodes Played: {len(total_rewards)}")
    print(f"Wins: {wins}")
    print(f"Win Rate: {wins/len(total_rewards)*100:.1f}%" if total_rewards else "N/A")
    print(f"Average Reward: {sum(total_rewards)/len(total_rewards):.2f}" if total_rewards else "N/A")
    

def test_qwen_simple():
    """Simple test of Qwen agent without full gameplay"""
    print("🧪 Simple Qwen Agent Test")
    print("-" * 30)
    
    # Create environment
    env = make_env()
    obs, info = env.reset()
    
    # Create agent  
    agent = QwenStreetFighterAgent()
    
    # Test a few decisions
    for i in range(3):
        print(f"\nTest {i+1}:")
        action, reasoning = agent.get_action(info, verbose=True)
        
        # Take action
        obs, reward, done, truncated, info = env.step(action)
        print(f"Reward: {reward:.3f}")
        
        if done:
            break
    
    env.close()
    print("\n✅ Simple test completed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen Street Fighter Demo")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                       help="Qwen model path (try: Qwen3-4B-Instruct-2507, Qwen3-8B, or Qwen2.5-7B/14B-Instruct)")
    parser.add_argument("--episodes", type=int, default=3,
                       help="Number of episodes to play")
    parser.add_argument("--no-render", action="store_true",
                       help="Disable rendering for faster execution")
    parser.add_argument("--quiet", action="store_true", 
                       help="Disable verbose reasoning output")
    parser.add_argument("--test-only", action="store_true",
                       help="Run simple test instead of full demo")
    
    args = parser.parse_args()
    
    if args.test_only:
        test_qwen_simple()
    else:
        demo_qwen_gameplay(
            model_path=args.model,
            episodes=args.episodes,
            render=not args.no_render,
            verbose=not args.quiet
        )