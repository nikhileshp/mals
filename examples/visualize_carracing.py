"""
Visualize CarRacing environment and save frames
"""
import os
import numpy as np
from PIL import Image
import gym

# Register the environment
import carracing_gym

def visualize_random_episode(output_dir="visualization", num_steps=500):
    """
    Run a random episode and save frames as images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    env = gym.make("CarRacingPLS-v1", verbose=1, seed=42, render_mode="gray")
    
    obs = env.reset()
    frames = []
    
    print(f"Running {num_steps} steps...")
    for step in range(num_steps):
        # Random action
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, done, info = env.step(action)
        
        # Get current frame (it's already in the obs)
        # obs shape is (4, 48, 48) - 4 stacked frames
        current_frame = obs[-1]  # Get last frame
        
        # Convert to image
        # Denormalize from [-1, 1] to [0, 255]
        frame_img = ((current_frame + 1) * 127.5).astype(np.uint8)
        
        frames.append(frame_img)
        
        if step % 50 == 0:
            print(f"Step {step}/{num_steps}, Reward: {reward:.2f}")
        
        if done:
            print(f"Episode finished at step {step}")
            obs = env.reset()
    
    env.close()
    
    # Save frames
    print(f"\nSaving {len(frames)} frames to {output_dir}/")
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame, mode='L')  # Grayscale
        img.save(f"{output_dir}/frame_{i:04d}.png")
    
    # Create a montage of some frames
    print("Creating montage...")
    selected_frames = frames[::50][:20]  # Every 50th frame, max 20
    
    # Create grid
    grid_size = int(np.ceil(np.sqrt(len(selected_frames))))
    montage_h = grid_size * 48
    montage_w = grid_size * 48
    montage = np.zeros((montage_h, montage_w), dtype=np.uint8)
    
    for idx, frame in enumerate(selected_frames):
        row = idx // grid_size
        col = idx % grid_size
        montage[row*48:(row+1)*48, col*48:(col+1)*48] = frame
    
    montage_img = Image.fromarray(montage, mode='L')
    montage_img.save(f"{output_dir}/montage.png")
    print(f"Montage saved to {output_dir}/montage.png")
    
    # Try to create a GIF if PIL supports it
    try:
        print("Creating GIF animation...")
        gif_frames = [Image.fromarray(f, mode='L') for f in frames[::5]]  # Every 5th frame
        gif_frames[0].save(
            f"{output_dir}/animation.gif",
            save_all=True,
            append_images=gif_frames[1:],
            duration=100,
            loop=0
        )
        print(f"GIF saved to {output_dir}/animation.gif")
    except Exception as e:
        print(f"Could not create GIF: {e}")
    
    print(f"\nVisualization complete! Check {output_dir}/ for outputs")

if __name__ == "__main__":
    visualize_random_episode()
