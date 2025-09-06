#!/usr/bin/env python3  # Shebang line to run script with python3 directly
"""
🥊 Demo: Qwen agent playing Street Fighter 2
Shows how to integrate Qwen reasoning with the existing wrapper
"""

import time  # Import time module for sleep delays
from wrapper import make_env  # Import environment creation function from wrapper
from qwen_agent import QwenStreetFighterAgent  # Import the Qwen AI agent class


def demo_qwen_gameplay(model_path: str = "/home/kenpeter/.cache/huggingface/hub/Qwen2.5-VL-3B-Instruct",  # Function to demo Qwen agent gameplay
                       episodes: int = 3,  # Default number of episodes to play
                       render: bool = True,  # Default to render game visuals
                       verbose: bool = True):  # Default to verbose output
    """
    Demo Qwen agent playing Street Fighter 2
    
    Args:
        model_path: Path to Qwen model
        episodes: Number of episodes to play
        render: Whether to render the game
        verbose: Whether to print detailed reasoning
    """
    print("🥊 Starting Qwen Street Fighter Demo")  # Print demo start message
    print(f"Model: {model_path}")  # Print model path being used
    print(f"Episodes: {episodes}")  # Print number of episodes to play
    print("-" * 50)  # Print separator line
    
    # Create environment and agent
    env = make_env()  # Create Street Fighter 2 environment
    agent = QwenStreetFighterAgent(model_path)  # Create Qwen agent with specified model
    
    wins = 0  # Initialize win counter
    total_rewards = []  # Initialize list to store episode rewards
    
    try:  # Try to run episodes with error handling
        for episode in range(episodes):  # Loop through each episode
            print(f"\n🎮 Episode {episode + 1}/{episodes}")  # Print episode header
            print("=" * 30)  # Print episode separator
            
            # Reset environment and agent
            obs, info = env.reset()  # Reset environment and get initial observation
            agent.reset()  # Reset agent internal state
            
            episode_reward = 0  # Initialize episode reward accumulator
            steps = 0  # Initialize step counter
            max_steps = 5000  # Allow full match completion with step limit
            
            while steps < max_steps:  # Main episode loop
                if render:  # Check if should render visuals
                    env.render()  # Render game screen
                    time.sleep(0.05)  # Slow down for visibility (50ms delay)
                
                # Get action from Qwen vision agent
                action, reasoning = agent.get_action(obs, info, verbose=verbose)  # Get AI action decision
                
                # Take step in environment
                obs, reward, done, truncated, info = env.step(action)  # Execute action and get results
                episode_reward += reward  # Add reward to episode total
                steps += 1  # Increment step counter
                
                # Check if episode ended
                if done or truncated:  # Check if episode is finished
                    agent_won = info.get("agent_won", False)  # Check if agent won the match
                    if agent_won:  # If agent won
                        wins += 1  # Increment win counter
                        print(f"🏆 Won episode {episode + 1}!")  # Print victory message
                    else:  # If agent lost
                        print(f"💀 Lost episode {episode + 1}")  # Print loss message
                    
                    print(f"Steps: {steps}, Reward: {episode_reward:.2f}")  # Print episode statistics
                    break  # Exit episode loop
            
            total_rewards.append(episode_reward)  # Add episode reward to total list
            
            if not render:  # If not rendering visuals
                print(f"Episode {episode + 1} completed - Reward: {episode_reward:.2f}")  # Print completion message
    
    except KeyboardInterrupt:  # Handle user interruption
        print("\n⚠️  Demo interrupted by user")  # Print interruption message
    
    finally:  # Always execute cleanup
        env.close()  # Close environment properly
    
    # Summary
    print("\n" + "="*50)  # Print summary header separator
    print("🏁 DEMO SUMMARY")  # Print summary title
    print("="*50)  # Print summary separator
    print(f"Episodes Played: {len(total_rewards)}")  # Print total episodes played
    print(f"Wins: {wins}")  # Print total wins
    print(f"Win Rate: {wins/len(total_rewards)*100:.1f}%" if total_rewards else "N/A")  # Calculate and print win rate
    print(f"Average Reward: {sum(total_rewards)/len(total_rewards):.2f}" if total_rewards else "N/A")  # Calculate and print average reward
    

def test_qwen_simple():  # Function for simple agent testing
    """Simple test of Qwen agent without full gameplay"""
    print("🧪 Simple Qwen Agent Test")  # Print test header
    print("-" * 30)  # Print separator line
    
    # Create environment
    env = make_env()  # Create Street Fighter 2 environment
    obs, info = env.reset()  # Reset environment and get initial state
    
    # Create agent  
    agent = QwenStreetFighterAgent("/home/kenpeter/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/main")  # Create agent with model path
    
    # Test a few decisions
    for i in range(3):  # Loop through 3 test iterations
        print(f"\nTest {i+1}:")  # Print test iteration number
        action, reasoning = agent.get_action(obs, info, verbose=True)  # Get action from agent
        
        # Take action
        obs, reward, done, truncated, info = env.step(action)  # Execute action in environment
        print(f"Reward: {reward:.3f}")  # Print reward received
        
        if done:  # Check if episode ended
            break  # Exit test loop if done
    
    env.close()  # Close environment
    print("\n✅ Simple test completed")  # Print test completion message


if __name__ == "__main__":  # Check if script is run directly
    import argparse  # Import argument parser module
    
    parser = argparse.ArgumentParser(description="Qwen Street Fighter Demo")  # Create argument parser
    parser.add_argument("--model", type=str, default="/home/kenpeter/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/main",  # Model path argument
                       help="SmolVLM model local path")  # Help text for model argument
    parser.add_argument("--episodes", type=int, default=3,  # Episodes count argument
                       help="Number of episodes to play")  # Help text for episodes argument
    parser.add_argument("--no-render", action="store_true",  # No render flag argument
                       help="Disable rendering for faster execution")  # Help text for no-render argument
    parser.add_argument("--quiet", action="store_true",  # Quiet mode flag argument
                       help="Disable verbose reasoning output")  # Help text for quiet argument
    parser.add_argument("--test-only", action="store_true",  # Test only flag argument
                       help="Run simple test instead of full demo")  # Help text for test-only argument
    
    args = parser.parse_args()  # Parse command line arguments
    
    if args.test_only:  # Check if test-only mode requested
        test_qwen_simple()  # Run simple test function
    else:  # Otherwise run full demo
        demo_qwen_gameplay(  # Call main demo function
            model_path=args.model,  # Pass model path from arguments
            episodes=args.episodes,  # Pass episode count from arguments
            render=not args.no_render,  # Invert no-render flag for render parameter
            verbose=not args.quiet  # Invert quiet flag for verbose parameter
        )