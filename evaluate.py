# Import necessary libraries
import tensorflow as tf
import numpy as np
import argparse  # Command-line argument parsing
import os  # For interacting with the operating system
import core  # Custom core functions module, defining dynamics, CBFs, etc.
import config  # Configuration settings for evaluation

# The evaluate.py file contains the main function to evaluate a trained neural network model for controlling agents using Control Barrier Functions (CBF).
# The purpose of this file is to evaluate how well the trained model maintains safety and achieves desired control objectives for multi-agent systems.

# Overview of evaluation code:
# 1. **parse_args Function**: Parses command-line arguments used for configuring the evaluation, such as number of agents, number of evaluation steps, model path, and visualization options.
# 2. **build_evaluation_graph Function**: Constructs the evaluation graph, including placeholders for the current state, reference state, and computing safety metrics using CBF.
# 3. **evaluate Function**: Loads the pre-trained model, runs the evaluation steps, and calculates metrics such as safety ratio, distance error, and reward.
# 4. **render_init and show_obstacles Functions**: Used for visualization of the agents' trajectories and obstacles in a 3D plot. These functions help in understanding the agents' behavior during evaluation.

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()  # Create argument parser
    # Add arguments for evaluation configuration such as number of agents, maximum steps, model path, and other settings
    parser.add_argument('--num_agents', type=int, default=64)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--model_path', type=str, default='model.ckpt')
    parser.add_argument('--vis', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()  # Parse the arguments
    return args  # Return parsed arguments

# Function to build the evaluation graph for the model
def build_evaluation_graph(num_agents):
    """
    Builds the computation graph for evaluating the neural network model.
    Args:
        num_agents: The number of agents to evaluate.
    Returns:
        Placeholders, evaluation metrics, and control actions for the model.
    """
    # Placeholder for state tensor representing the current state of agents
    state_dim = 8  # Generalized state dimension
    s = tf.placeholder(tf.float32, [None, num_agents, state_dim])  # Shape [batch_size, num_agents, state_dim]
    # Placeholder for goal states tensor
    s_ref = tf.placeholder(tf.float32, [None, num_agents, state_dim])  # Shape [batch_size, num_agents, state_dim]

    # Compute the CBF values for agents
    h, mask, indices = core.network_cbf(x=s, r=config.DIST_MIN_THRES, indices=None)
    # Compute control actions based on the current state and reference state
    u = core.network_action(s=s, s_ref=s_ref, obs_radius=config.OBS_RADIUS, indices=indices)
    # Compute safety mask indicating whether agents are in a safe configuration
    safe_mask = core.compute_safe_mask(s, r=config.DIST_SAFE, indices=indices)
    is_safe = tf.reduce_mean(tf.cast(safe_mask, tf.float32))  # Safety metric as mean safety mask

    return s, s_ref, u, is_safe  # Return placeholders, control action, and safety metric

# Function to evaluate the model
def evaluate(args):
    """
    Function to evaluate the model for safe multi-agent control using Control Barrier Functions.
    Args:
        args: Parsed command-line arguments.
    """
    # Set GPU for evaluation if specified
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Build the evaluation graph
    s, s_ref, u, is_safe = build_evaluation_graph(args.num_agents)

    # Create a session for evaluation
    with tf.Session() as sess:
        # Load the pre-trained model
        saver = tf.train.Saver()
        saver.restore(sess, args.model_path)  # Restore model parameters

        # Loop for evaluation over the maximum number of steps
        for step in range(args.max_steps):
            # Generate random evaluation data for states and reference states (placeholders for actual data generation)
            s_batch = np.random.rand(1, args.num_agents, 8)  # Random state batch
            s_ref_batch = np.random.rand(1, args.num_agents, 8)  # Random reference state batch

            # Run the evaluation operation to get control inputs and compute safety metrics
            u_value, safety_value = sess.run([u, is_safe], feed_dict={s: s_batch, s_ref: s_ref_batch})

            # Print safety information every 100 steps
            if step % 100 == 0:
                print(f"Step {step}, Safety Metric: {safety_value}")

# Function to initialize the rendering environment
def render_init(num_agents):
    fig = plt.figure(figsize=(10, 7))  # Create a matplotlib figure
    return fig

# Function to visualize obstacles in a 3D plot
def show_obstacles(obs, ax, z=[0, 6], alpha=0.6, color='deepskyblue'):
    for x1, y1, x2, y2 in obs:
        xs, ys = np.meshgrid([x1, x2], [y1, y2])
        zs = np.ones_like(xs)
        # Plot the top and bottom surfaces of the obstacle
        ax.plot_surface(xs, ys, zs * z[0], alpha=alpha, color=color)
        ax.plot_surface(xs, ys, zs * z[1], alpha=alpha, color=color)
        # Plot the sides of the obstacle
        xs, zs = np.meshgrid([x1, x2], z)
        ys = np.ones_like(xs)
        ax.plot_surface(xs, ys * y1, zs, alpha=alpha, color=color)
        ax.plot_surface(xs, ys * y2, zs, alpha=alpha, color=color)
        ys, zs = np.meshgrid([y1, y2], z)
        xs = np.ones_like(ys)
        ax.plot_surface(xs * x1, ys, zs, alpha=alpha, color=color)
        ax.plot_surface(xs * x2, ys, zs, alpha=alpha, color=color)

# Main function to parse arguments and call the evaluation function
def main():
    args = parse_args()  # Parse command-line arguments
    evaluate(args)  # Evaluate the model using the parsed arguments

# Run the main function if the script is executed directly
if __name__ == '__main__':
    main()
