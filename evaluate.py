import sys
# Prevent Python from generating .pyc files in the __pycache__ directory.
sys.dont_write_bytecode = True

import os  # Operating system utilities.
import time  # Time-related functions.
import argparse  # Command-line argument parsing.
import numpy as np  # Numerical computing library.
import tensorflow as tf  # Machine learning framework.
import matplotlib.pyplot as plt  # Plotting library for visualization.
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting.
import pickle  # Object serialization.

# Placeholder imports for core functions and configuration settings.
# Replace these with appropriate implementations for different applications.
import core  # Custom module for core functions (generalized for various tasks).
import config  # Custom module for configuration settings (generalized).

# Function to parse command-line arguments.
def parse_args():
    parser = argparse.ArgumentParser()  # Create argument parser.
    # Add arguments for number of agents, max steps, model path, visualization, GPU, and reference file.
    parser.add_argument('--num_agents', type=int, default=64)
    parser.add_argument('--max_steps', type=int, default=12)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--vis', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--ref', type=str, default=None)
    args = parser.parse_args()  # Parse the arguments.
    return args  # Return the parsed arguments.

# Function to build the evaluation graph for multi-agent control.
def build_evaluation_graph(num_agents):
    # Placeholder for agents' states with shape [num_agents, state_dim].
    state_dim = 8  # Generalized state dimension.
    s = tf.placeholder(tf.float32, [num_agents, state_dim])
    # Placeholder for goal states with shape [num_agents, state_dim].
    s_ref = tf.placeholder(tf.float32, [num_agents, state_dim])
    # Compute the difference between the state of each agent and other agents.
    x = tf.expand_dims(s, 1) - tf.expand_dims(s, 0)
    # Compute the Control Barrier Function (CBF) value for agents.
    h, mask, indices = core.network_cbf(
        x=x, r=config.DIST_MIN_THRES, indices=None)
    # Compute the control action for each agent.
    u = core.network_action(
        s=s, s_ref=s_ref, obs_radius=config.OBS_RADIUS, indices=indices)
    # Compute the safety mask indicating whether agents are within a safe distance.
    safe_mask = core.compute_safe_mask(s, r=config.DIST_SAFE, indices=indices)
    # Check if all agents are in a safe state (mean safety mask is 1).
    is_safe = tf.equal(tf.reduce_mean(tf.cast(safe_mask, tf.float32)), 1)

    # Initialize a variable for the control adjustment to ensure safety.
    u_res = tf.Variable(tf.zeros_like(u), name='u_res')
    loop_count = tf.Variable(0, name='loop_count')
   
    # Function to update control adjustments to satisfy safety conditions.
    def opt_body(u_res, loop_count, is_safe):
        # Compute the next state based on the dynamics with control adjustments.
        dsdt = core.system_dynamics_tf(s, u + u_res)  # Generalized dynamics function.
        s_next = s + dsdt * config.TIME_STEP_EVAL
        # Compute the difference between the next state and other agents.
        x_next = tf.expand_dims(s_next, 1) - tf.expand_dims(s_next, 0)
        # Compute the CBF value for the next state.
        h_next, mask_next, _ = core.network_cbf(
            x=x_next, r=config.DIST_MIN_THRES, indices=indices)
        # Derivative of the CBF should be greater than or equal to zero.
        deriv = h_next - h + config.TIME_STEP_EVAL * config.ALPHA_CBF * h
        deriv = deriv * mask * mask_next
        # Compute the error for unsafe conditions.
        error = tf.reduce_sum(tf.math.maximum(-deriv, 0), axis=1)
        # Compute the gradient of the error with respect to u_res to adjust control.
        error_gradient = tf.gradients(error, u_res)[0]
        u_res = u_res - config.REFINE_LEARNING_RATE * error_gradient
        loop_count = loop_count + 1  # Increment loop count.
        return u_res, loop_count, is_safe

    # Condition to continue updating u_res until safety is achieved or maximum loops.
    def opt_cond(u_res, loop_count, is_safe):
        cond = tf.logical_and(
            tf.less(loop_count, config.REFINE_LOOPS), 
            tf.logical_not(is_safe))
        return cond
    
    # Use while loop to refine u_res until safety conditions are satisfied.
    with tf.control_dependencies([
        u_res.assign(tf.zeros_like(u)), loop_count.assign(0)]):
        u_res, _, _ = tf.while_loop(opt_cond, opt_body, [u_res, loop_count, is_safe])
        u_opt = u + u_res  # Compute optimal control.

    # Compute loss functions and accuracies for different safety conditions.
    loss_dang, loss_safe, acc_dang, acc_safe = core.loss_barrier(
        h=h, s=s, indices=indices)
    (loss_dang_deriv, loss_safe_deriv, loss_medium_deriv, acc_dang_deriv, 
    acc_safe_deriv, acc_medium_deriv) = core.loss_derivatives(
        s=s, u=u_opt, h=h, x=x, indices=indices)
    # Compute the loss between optimal and nominal control actions.
    loss_action = core.loss_actions(s=s, u=u_opt, s_ref=s_ref, indices=indices)

    # Compile lists of loss and accuracy metrics.
    loss_list = [loss_dang, loss_safe, loss_dang_deriv, 
                 loss_safe_deriv, loss_medium_deriv, loss_action]
    acc_list = [acc_dang, acc_safe, acc_dang_deriv, acc_safe_deriv, acc_medium_deriv]

    return s, s_ref, u_opt, loss_list, acc_list

# Function to print the average accuracy values.
def print_accuracy(accuracy_lists):
    acc = np.array(accuracy_lists)
    acc_list = []
    for i in range(acc.shape[1]):
        acc_i = acc[:, i]
        acc_list.append(np.mean(acc_i[acc_i > 0]))
    print('Accuracy: {}'.format(acc_list))

# Function to initialize the rendering environment.
def render_init(num_agents):
    fig = plt.figure(figsize=(10, 7))  # Create a matplotlib figure.
    return fig

# Function to visualize obstacles in a 3D plot.
def show_obstacles(obs, ax, z=[0, 6], alpha=0.6, color='deepskyblue'):
    for x1, y1, x2, y2 in obs:
        xs, ys = np.meshgrid([x1, x2], [y1, y2])
        zs = np.ones_like(xs)
        # Plot the top and bottom surfaces of the obstacle.
        ax.plot_surface(xs, ys, zs * z[0], alpha=alpha, color=color)
        ax.plot_surface(xs, ys, zs * z[1], alpha=alpha, color=color)
        # Plot the sides of the obstacle.
        xs, zs = np.meshgrid([x1, x2], z)
        ys = np.ones_like(xs)
        ax.plot_surface(xs, ys * y1, zs, alpha=alpha, color=color)
        ax.plot_surface(xs, ys * y2, zs, alpha=alpha, color=color)
        ys, zs = np.meshgrid([y1, y2], z)
        xs = np.ones_like(ys)
        ax.plot_surface(xs * x1, ys, zs, alpha=alpha, color=color)
        ax.plot_surface(xs * x2, ys, zs, alpha=alpha, color=color)

# Function to clip the norm of vectors to a threshold.
def clip_norm(x, thres):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    mask = (norm > thres).astype(np.float32)
    x = x * (1 - mask) + x * mask / (1e-6 + norm)
    return x

# Function to clip the state of agents within certain thresholds.
def clip_state(s, x_thres, v_thres=0.1, h_thres=6):
    x, v, r = s[:, :3], s[:, 3:6], s[:, 6:]
    x = np.concatenate([np.clip(x[:, :2], 0, x_thres),
                        np.clip(x[:, 2:], 0, h_thres)], axis=1)
    v = clip_norm(v, v_thres)
    s = np.concatenate([x, v, r], axis=1)
    return s

# Main function to execute the multi-agent control simulation.
def main():
    args = parse_args()  # Parse command-line arguments.
    # Build the evaluation graph for multi-agent control.
    s, s_ref, u, loss_list, acc_list = build_evaluation_graph(args.num_agents)
    # Load the pre-trained model weights.
    vars = tf.trainable_variables()
    vars_restore = []
    for v in vars:
        if 'action' in v.name or 'cbf' in v.name:
            vars_restore.append(v)
    # Initialize the TensorFlow session.
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_list=vars_restore)
    saver.restore(sess, args.model_path)  # Restore model parameters.

    # Lists to store evaluation metrics over epochs.
    safety_ratios_epoch = []
    safety_ratios_epoch_baseline = []
    dist_errors = []
    dist_errors_baseline = []
    accuracy_lists = []

    # Visualization setup if enabled.
    if args.vis > 0:
        plt.ion()  # Interactive plotting mode.
        plt.close()
        fig = render_init(args.num_agents)
    # Initialize the environment with agents.
    scene = core.Environment(args.num_agents, max_steps=args.max_steps)  # Generalized environment.
    if args.ref is not None:
        scene.read(args.ref)  # Load reference points for the environment.

    # Create directory to store trajectory results.
    if not os.path.exists('trajectory'):
        os.mkdir('trajectory')
    traj_dict = {'ours': [], 'baseline': [], 'obstacles': [np.array(scene.OBSTACLES)]}
    
    # Lists to store reward values.
    safety_reward = []
    dist_reward = []
 
    # Evaluate for a fixed number of steps defined in config.
    for istep in range(config.EVALUATE_STEPS):
        if args.vis > 0:
            plt.clf()  # Clear current figure.
            ax_1 = fig.add_subplot(121, projection='3d')
            ax_2 = fig.add_subplot(122, projection='3d')
        safety_ours = []
        safety_baseline = []

        scene.reset()  # Reset the environment to start new episode.
        start_time = time.time()  # Start time for measuring computation duration.
        # Initial state with starting positions and zero velocity and rotational states.
        s_np = np.concatenate(
            [scene.start_points, np.zeros((args.num_agents, 5))], axis=1)
        s_traj = []
        safety_info = np.zeros(args.num_agents, dtype=np.float32)
        # Loop through each goal state in the environment sequence.
        for t in range(scene.steps):
            # Define reference (goal) state.
            s_ref_np = np.concatenate(
                [scene.waypoints[t], np.zeros((args.num_agents, 5))], axis=1)
            # Run the control loop for reaching the goal state.
            for i in range(config.INNER_LOOPS_EVAL):
                # Run the TensorFlow session to get control inputs and accuracy metrics.
                u_np, acc_list_np = sess.run(
                    [u, acc_list], feed_dict={s: s_np, s_ref: s_ref_np})
                if args.vis == 1:
                    # Compute reference control using a baseline controller.
                    u_ref_np = core.baseline_controller_np(s_np, s_ref_np)  # Generalized baseline controller.
                    u_np = clip_norm(u_np - u_ref_np, 100.0) + u_ref_np
                # Update the state based on dynamics and control input.
                dsdt = core.system_dynamics_np(s_np, u_np)  # Generalized dynamics function.
                s_np = s_np + dsdt * config.TIME_STEP_EVAL
                # Compute safety ratio for each agent.
                safety_ratio = 1 - np.mean(
                    core.dangerous_mask_np(s_np, config.DIST_MIN_CHECK), axis=1)
                individual_safety = safety_ratio == 1
                safety_ours.append(individual_safety)
                safety_info = safety_info + individual_safety - 1
                safety_ratio = np.mean(individual_safety)
                safety_ratios_epoch.append(safety_ratio)
                accuracy_lists.append(acc_list_np)
                # Stop inner loop if agents are close enough to reference states.
                if np.mean(
                    np.linalg.norm(s_np[:, :3] - s_ref_np[:, :3], axis=1)
                    ) < config.DIST_TOLERATE:
                    break
                # Store trajectory information.
                s_traj.append(np.expand_dims(s_np[:, [0, 1, 2, 6, 7]], axis=0))
        # Compute and store reward metrics for the current episode.
        safety_reward.append(np.mean(safety_info))
        dist_reward.append(np.mean((np.linalg.norm(
            s_np[:, :3] - s_ref_np[:, :3], axis=1) < 1.5).astype(np.float32) * 10))
        dist_errors.append(
            np.mean(np.linalg.norm(s_np[:, :3] - s_ref_np[:, :3], axis=1)))
        traj_dict['ours'].append(np.concatenate(s_traj, axis=0))
        end_time = time.time()  # End time for current episode.

        # Perform the same task using a baseline controller.
        s_np = np.concatenate(
            [scene.start_points, np.zeros((args.num_agents, 5))], axis=1)
        s_traj = []
        for t in range(scene.steps):
            s_ref_np = np.concatenate(
                [scene.waypoints[t], np.zeros((args.num_agents, 5))], axis=1)
            for i in range(config.INNER_LOOPS_EVAL):
                # Compute control using the baseline controller.
                u_np = core.baseline_controller_np(s_np, s_ref_np)  # Generalized baseline controller.
                dsdt = core.system_dynamics_np(s_np, u_np)  # Generalized dynamics function.
                s_np = s_np + dsdt * config.TIME_STEP_EVAL
                # Compute safety metrics for baseline controller.
                safety_ratio = 1 - np.mean(
                    core.dangerous_mask_np(s_np, config.DIST_MIN_CHECK), axis=1)
                individual_safety = safety_ratio == 1
                safety_baseline.append(individual_safety)
                safety_ratio = np.mean(individual_safety)
                safety_ratios_epoch_baseline.append(safety_ratio)
                s_traj.append(np.expand_dims(s_np[:, [0, 1, 2, 6, 7]], axis=0))
        dist_errors_baseline.append(np.mean(np.linalg.norm(s_np[:, :3] - s_ref_np[:, :3], axis=1)))
        traj_dict['baseline'].append(np.concatenate(s_traj, axis=0))

        if args.vis > 0:
            # Visualize the trajectories of agents using both controllers.
            s_traj_ours = traj_dict['ours'][-1]
            s_traj_baseline = traj_dict['baseline'][-1]
    
            for j in range(0, max(s_traj_ours.shape[0], s_traj_baseline.shape[0]), 10):
                ax_1.clear()
                ax_1.view_init(elev=80, azim=-45)
                ax_1.axis('off')
                show_obstacles(scene.OBSTACLES, ax_1)
                j_ours = min(j, s_traj_ours.shape[0]-1)
                s_np = s_traj_ours[j_ours]
                safety = safety_ours[j_ours]

                ax_1.set_xlim(0, 20)
                ax_1.set_ylim(0, 20)
                ax_1.set_zlim(0, 10)
                ax_1.scatter(s_np[:, 0], s_np[:, 1], s_np[0, 2], 
                             color='darkorange', label='Agent')
                ax_1.scatter(s_np[safety<1, 0], s_np[safety<1, 1], s_np[safety<1, 2], 
                             color='red', label='Collision')
                ax_1.set_title('Ours: Safety Rate = {:.4f}'.format(
                    np.mean(safety_ratios_epoch)), fontsize=16)

                ax_2.clear()
                ax_2.view_init(elev=80, azim=-45)
                ax_2.axis('off')
                show_obstacles(scene.OBSTACLES, ax_2)
                j_baseline = min(j, s_traj_baseline.shape[0]-1)
                s_np = s_traj_baseline[j_baseline]
                safety = safety_baseline[j_baseline]

                ax_2.set_xlim(0, 20)
                ax_2.set_ylim(0, 20)
                ax_2.set_zlim(0, 10)
                ax_2.scatter(s_np[:, 0], s_np[:, 1], s_np[1, 2], 
                             color='darkorange', label='Agent')
                ax_2.scatter(s_np[safety<1, 0], s_np[safety<1, 1], s_np[safety<1, 2], 
                             color='red', label='Collision')
                ax_2.set_title('Baseline: Safety Rate = {:.4f}'.format(
                    np.mean(safety_ratios_epoch_baseline)), fontsize=16)
                plt.legend(loc='lower right')

                fig.canvas.draw()
                plt.pause(0.001)

        # Print evaluation progress.
        print('Evaluation Step: {} | {}, Time: {:.4f}'.format(
            istep + 1, config.EVALUATE_STEPS, end_time - start_time))

    # Print accuracy statistics.
    print_accuracy(accuracy_lists)
    # Print distance errors for both learning and baseline.
    print('Distance Error (Learning | Baseline): {:.4f} | {:.4f}'.format(
          np.mean(dist_errors), np.mean(dist_errors_baseline)))
    # Print mean safety ratio for both learning and baseline.
    print('Mean Safety Ratio (Learning | Baseline): {:.4f} | {:.4f}'.format(
          np.mean(safety_ratios_epoch), np.mean(safety_ratios_epoch_baseline)))

    # Compute and print final reward metrics.
    safety_reward = np.mean(safety_reward)
    dist_reward = np.mean(dist_reward)
    print('Safety Reward: {:.4f}, Dist Reward: {:.4f}, Reward: {:.4f}'.format(
        safety_reward, dist_reward, 9 + 0.1 * (safety_reward + dist_reward)))

    # Save trajectory data for further analysis.
    pickle.dump(traj_dict, open('trajectory/traj_eval.pkl', 'wb'))
    scene.write_trajectory('trajectory/env_traj_eval.pkl', traj_dict['ours'])

# Run the main function when the script is executed.
if __name__ == '__main__':
    main()
