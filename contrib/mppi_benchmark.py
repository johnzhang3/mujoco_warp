"""Run benchmarks for sample-based mpc"""

import inspect
from typing import Sequence

import mujoco
import numpy as np
import warp as wp
from absl import app
from absl import flags
from etils import epath

import mujoco_warp as mjwarp
import matplotlib.pyplot as plt

def get_total_sim_time(mjcf_path, batch_size, nstep):
    """Runs testpeed function."""
    wp.init()

    path = epath.Path(mjcf_path)
    if not path.exists():
        path = epath.resource_path("mujoco_warp") / mjcf_path
    if not path.exists():
        raise FileNotFoundError(f"file not found: {mjcf_path}\nalso tried: {path}")
    if path.suffix == ".mjb":
        mjm = mujoco.MjModel.from_binary_path(path.as_posix())
    else:
        mjm = mujoco.MjModel.from_xml_path(path.as_posix())

    mjm.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL # force pyramidal cone for benchmarking
    mjm.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON

    # default parameters for benchmarking
    mjm.opt.iterations = 10
    mjm.opt.ls_iterations = 20
    mjm.opt.timestep = 0.01 

    mjd = mujoco.MjData(mjm)

    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    # populate some constraints
    mujoco.mj_forward(mjm, mjd)

    m = mjwarp.put_model(mjm)
    m.opt.ls_parallel = False
    d = mjwarp.put_data(
        mjm, mjd, nworld=batch_size, nconmax=None, njmax=None
    )

    wp.clear_kernel_cache()    

    jit_time, run_time, trace, ncon, nefc = mjwarp.benchmark(
    mjwarp.__dict__["step"],
        m,
        d,
        nstep,
        False,
        False,
    )
    steps = batch_size * nstep

    print(f"""
    Summary for {batch_size} parallel rollouts and {nstep} steps at dt = {m.opt.timestep:.3f}:

    Total JIT time: {jit_time:.4f} s
    Total simulation time: {run_time:.4f} s
    Total steps per second: {steps / run_time:,.0f}
    Total realtime factor: {steps * m.opt.timestep / run_time:,.2f} x
    Total time per step: {1e9 * run_time / steps:.2f} ns""")

    return run_time

def plot_heatmap(batch_sizes, nsteps, total_times, experiment_name, save_path='benchmark_heatmap.png'):
    """Creates and saves a log-scale heatmap visualization of the benchmark results.
    
    Args:
        batch_sizes: List of batch sizes used in the benchmark.
        nsteps: List of step counts used in the benchmark.
        total_times: 2D array of total simulation times.
        save_path: Path to save the resulting figure.
    """
    # Create figure for log scale only
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Log scale heatmap (add small epsilon to avoid log(0))
    epsilon = 1e-10
    log_times = np.log10(total_times + epsilon)
    im = ax.imshow(log_times, cmap='viridis')
    ax.set_title("Total Simulation Time", fontsize=14)
    
    # Show all ticks and label them
    ax.set_xticks(np.arange(len(nsteps)))
    ax.set_yticks(np.arange(len(batch_sizes)))
    ax.set_xticklabels(nsteps)
    ax.set_yticklabels(batch_sizes)
    
    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add axis labels
    ax.set_xlabel("Number of Steps", fontsize=12)
    ax.set_ylabel("Batch Size", fontsize=12)
    
    # Add annotations showing the actual time values (not log)
    for i in range(len(batch_sizes)):
        for j in range(len(nsteps)):
            time_value = total_times[i, j]
            text_color = "white" if log_times[i, j] > log_times.max()/2 else "black"
            ax.text(j, i, f"{time_value:.4f}s", ha="center", va="center", 
                    color=text_color, fontsize=10)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('total sim time (s)', rotation=-90, va="bottom", fontsize=12)
    
    # Add overall title
    fig.suptitle(f"{experiment_name} benchmark results", fontsize=16)
    
    # Adjust layout and save
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"heatmap saved to {save_path}")

    return fig

def main(experiment_name):
    if experiment_name == "humanoid":
        mjcf_path = "test_data/humanoid/humanoid.xml"
    elif experiment_name == "quadruped":
        mjcf_path = "test_data/quadruped/scene.xml"
    elif experiment_name == "quadruped_box":
        mjcf_path = "test_data/quadruped/scene_box.xml"
    else:
        raise ValueError(f"Unknown experiment name: {experiment_name}")

    run_benchmark = True
    if run_benchmark:
        batch_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
        nsteps = [1, 10, 50, 100]

        total_times = np.zeros((len(batch_sizes), len(nsteps)))
        for (i, nstep) in enumerate(nsteps):
            for (j, batch_size) in enumerate(batch_sizes):
                total_time = get_total_sim_time(mjcf_path, batch_size, nstep)
                total_times[j, i] = total_time

        # Save the results to a numpy file for later analysis
        np.save(f'contrib/{experiment_name}_benchmark_results.npy', {
            'batch_sizes': batch_sizes,
            'nsteps': nsteps,
            'total_times': total_times
        })
    else:
        # Load the results from the numpy file
        data = np.load('benchmark_results.npy', allow_pickle=True).item()
        batch_sizes = data['batch_sizes']
        nsteps = data['nsteps']
        total_times = data['total_times']
    
    # Plot and save the heatmap
    plot_heatmap(batch_sizes, nsteps, total_times, experiment_name, save_path=f"contrib/{experiment_name}_benchmark_heatmap.png")

    pass 

if __name__ == "__main__":
    # main("humanoid")
    # main("quadruped")
    main("quadruped_box")