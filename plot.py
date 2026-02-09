import matplotlib.pyplot as plt
import numpy as np
import torch

#########################################################
# Functions primarily used in basics.ipynb
#########################################################

def compare_to_target(function_to_learn, type):
    xmin, xmax = -3, 3
    ymin, ymax = -8, 8

    x = np.linspace(xmin, xmax, 100)

    if type == 'function':
        target_function = lambda x: 2*x + 1
        target_x = x
        target_y = target_function(target_x)
    elif type == 'data':
        target_function = lambda x: -2*x - 1
        target_x = np.array([-2, 0, 2])
        noise = np.array([1.3, -0.6, 1.0])
        target_y = target_function(target_x) + noise

    # Plot code
    fig, ax = plt.subplots(figsize=(8, 6))

    def add_arrow(ax, x1, y1, x2, y2, i):
        # Draw arrow from (x1, y1) to (x2, y2)
        ax.annotate(
            '',  # No text
            xy=(x2, y2),  # End of arrow
            xytext=(x1, y1),  # Start of arrow
            arrowprops=dict(
                arrowstyle='<->',
                color='black',
                linewidth=2,
            )
        )

        mid_y = (y1 + y2) / 2
        err = np.abs(y2 - y1)
        ax.text(x1 + 0.1, mid_y, f'$Δ_{i} = {err:.2f}$', va='center', ha='left', fontsize=12)

    ax.plot(x, function_to_learn(x), label='Our function to learn', color='red', zorder=11, linewidth=3)

    if type == 'function':
        ax.plot(x, target_function(x), label='Unknown target function $f$', color='blue', zorder=10, linewidth=2)

    elif type == 'data':
        my_y = function_to_learn(target_x)
        ax.scatter(target_x, target_y, label='Data from our unknown target function $f$', color='blue', zorder=10, linewidth=2)
        ax.scatter(target_x, my_y, color='red', zorder=11, linewidth=3)

        sum = np.sum((my_y - target_y) ** 2)
        ax.text(xmin + 0.2, ymin + 1.0, f'$\\Delta_1^2 + \\Delta_2^2 + \\Delta_3^2 = {sum:.2f}$', va='center', ha='left',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.5), fontsize=18)

        add_arrow(ax, target_x[0], target_y[0], target_x[0], my_y[0], 1)
        add_arrow(ax, target_x[1], target_y[1], target_x[1], my_y[1], 2)
        add_arrow(ax, target_x[2], target_y[2], target_x[2], my_y[2], 3)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.arange(xmin, xmax + 1, 1))
    ax.set_yticks(np.arange(ymin, ymax + 1, 1))
    ax.grid()
    ax.legend()
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_title(f'Comparison of our function to learn with the target {type}')
    plt.show()


def plot_loss_landscape(func, sequence, x_range, y_range):
    xmin, xmax = x_range
    ymin, ymax = y_range

    x = np.linspace(xmin, xmax, 400)
    y = np.linspace(ymin, ymax, 400)

    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    opt_a, opt_b = -2.075, -0.43333333
    f_opt = func(opt_a, opt_b)

    # Plot code

    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(111, projection='3d')

    def plot_line_on_surface(ax, p0, p1, func, n=20):
        x0, y0 = p0
        x1, y1 = p1

        if not isinstance(p0, np.ndarray):
            p0 = np.array([p0])
        if not isinstance(p1, np.ndarray):
            p1 = np.array([p1])

        x = p0 + (p1 - p0) * np.linspace(0, 1, n).reshape(-1, 1)
        z = func(x[:, 0], x[:, 1])

        ax.plot(x[:, 0], x[:, 1], z, color='red', alpha=1.0, zorder=5)
        ax.plot([x0], [y0], [func(x0, y0)], color='red', alpha=1.0, zorder=5, marker='o', markersize=6)
        ax.plot([x1], [y1], [func(x1, y1)], color='red', alpha=1.0, zorder=5, marker='o', markersize=6)

    surface = ax.plot_surface(X, Y, Z, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
                              alpha=1.0)  # Use default matplotlib blue with transparency

    for p0, p1 in zip(sequence[:-1], sequence[1:]):
        plot_line_on_surface(ax, p0, p1, func)

    ax.plot([opt_a], [opt_b], [f_opt], color='orange', alpha=1.0, marker='*', markersize=10, zorder=5)
    ax.plot([opt_a, opt_a], [opt_b, opt_b], [0.0, f_opt], color='orange', alpha=1.0)
    ax.plot([opt_a, x_range[1] + 0.01], [opt_b, opt_b], [0.0, 0.0], color='orange', alpha=1.0)
    ax.plot([opt_a, opt_a], [opt_b, y_range[1] + 0.03], [0.0, 0.0], color='orange', alpha=1.0)

    Z = func(opt_a, opt_b) * np.ones_like(Z)
    surface = ax.plot_surface(X, Y, Z, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], alpha=0.2)

    ax.set_xlabel("a", fontsize=14)
    ax.set_ylabel("b", fontsize=14)
    ax.view_init(elev=20, azim=75, roll=0)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(0.0, 4.0)

    ax.text(opt_a + 0.1, opt_b, f_opt + 0.3, "global minimum", fontsize=12, color='orange', zorder=100)

    ax.set_zlabel("$\\text{loss}(a,b)$", fontsize=14)
    plt.title("3D plot of our loss in dependence of the two parameters $a,b$", fontsize=14)
    plt.show()

def plot_loss_point(func, x0, xrange=(-3.0, -0.5), plot_loss=True):
    xmin, xmax = xrange
    x = np.linspace(xmin, xmax, 400)
    
    y = func(x)

    # Differentiate func symbolically to create tangent line
    from sympy import symbols, diff, lambdify
    
    xs = symbols('xs')
    f = func(xs)
    df = diff(f, xs)
    df = lambdify(xs, df, modules=['numpy'])
    tangent = lambda x: df(x0) * (x - x0) + func(x0)

    y0 = func(x0)
    slope = df(x0)

    fig, ax = plt.subplots(figsize=(10,6))

    if plot_loss:

        ax.plot(x, y, label='loss')
        
        ax.axhline(
            y=func(-2.075),
            color='orange',
            linestyle='--',
            linewidth=1,
            label='Optimal loss value'
        )

    # Draw tangent line
    ax.scatter([x0], [y0], color='red', alpha=1.0, marker='o', s=40, zorder=20)
    
    ax.plot([xmin, xmax], [tangent(xmin), tangent(xmax)], 'red', linestyle='--', linewidth=1.0, label='Tangent line')
    ax.text(x0 + 0.03, y0 - .3, r'$a_0$', va='center')
    
    # Labels and title
    ax.set_xlabel('a')
    ax.set_ylabel('loss(a)')
    ax.set_ylim(0.0, 15.0)
    ax.set_xlim(xrange)
    ax.legend()
    ax.grid()
    ax.set_title(f'Step 1: start at $a_0={x0}$')
    
    plt.show()

def plot_tangents(func, x0, step, xrange=(-3.0, -0.5), no_steps = 10, plot_loss=True):

    if step > 0.1:
            print("Warning: step size too large to plot!")
            step = 0.1
    
    xmin, xmax = xrange
    x = np.linspace(xmin, xmax, 400)

    x_opt = -2.075
    
    y = func(x)

    from sympy import symbols, diff, lambdify

    xs = symbols('xs')
    f = func(xs)
    df = diff(f, xs)
    df = lambdify(xs, df, modules=['numpy'])
    
    fig, ax = plt.subplots(figsize=(10,6))
    
    if plot_loss:

        ax.plot(x, y, label='loss')
        
        ax.axhline(
            y=func(x_opt),
            color='orange',
            linestyle='--',
            linewidth=1,
            label='Optimal loss value'
        )

    def add_tangent_decision(ax, x0, tangent, xmin, xmax, direction, step = 1.0, no = 0):
        
        y0 = tangent(x0)
        ax.scatter([x0], [y0], color='red', alpha=1.0, marker='o', s=40, zorder=20)
        ax.plot([xmin, xmax], [tangent(xmin), tangent(xmax)], 'red', linestyle='--', linewidth=1.0)
        ax.text(x0-0.07, y0+0.5, f'$a_{no}$', va='center')

        right = dict(
            arrowstyle='<-', 
            color=('black' if direction == 1 else 'lightgrey'), linewidth=2.0)
        left = dict(
            arrowstyle='->', 
            color=('lightgrey' if direction == 1 else 'black'), linewidth=2.0)

        if direction != 0:
            ax.annotate(
                '', 
                xy=(x0-step, y0), 
                xytext=(x0, y0),
                arrowprops=left
            )
            ax.annotate(
                '', 
                xy=(x0, y0), 
                xytext=(x0+step, y0),
                arrowprops=right
            )

    x_i = x0
    for i in range(no_steps):
        error = np.abs(x_i - x_opt)
        if error < 1e-3:
            print(f'You are already close enough to the optimal value with step {i}!')
            break

        d = df(x_i)
        d_i = -np.sign(d)
        tangent = lambda x: d * (x - x_i) + func(x_i)
        
        add_tangent_decision(ax, x_i, tangent, xmin, xmax, d_i, step=step * d, no=i)
        x_i -=step * d

    add_tangent_decision(ax, x_i, tangent, xmin, xmax, 0, step=step * d, no=i)
    
    # Labels and title
    ax.set_xlabel('a')
    ax.set_ylabel('loss(a)')
    ax.set_ylim(0.0, 15.0)
    ax.set_xlim(xrange)
    ax.grid()
    ax.set_title('Step 2: finding our way to the optimal value')
    
    plt.show()

def plot_activations(x_range, functions, labels, colors, linewidths, title="Function Comparison"):
    x = np.linspace(x_range[0], x_range[1], 500)
    plt.figure(figsize=(8, 6))

    for func, label, color, linewidth in zip(functions, labels, colors, linewidths):
        y = func(x)
        plt.plot(x, y, label=label, linewidth=linewidth, color=color)

    plt.title(title)
    plt.ylim(0, 2)
    plt.xlim(x_range)
    plt.xlabel('$x$')
    plt.ylabel('$y=f(x)$')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_2d_function(func, x_range, y_range):
    x = np.linspace(x_range[0], x_range[1], 400)
    y = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(X, Y, Z, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0], alpha=1)  # Use default matplotlib blue with transparency

    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    #ax.set_zlabel("z", fontsize=14)
    #ax.set_title("Surface Plot of the Function", fontsize=16)
    ax.invert_xaxis()  # Flip the x-axis
    plt.show()

#########################################################
# Functions primarily used in cats_dogs.ipynb
#########################################################

def show_selected_images_labels(images, labels, descriptions, title_prefix="Label", vmin=-1, vmax=1, rows=3, cols=3):
    """
    Display selected images and their corresponding labels.

    Args:
        images (Tensor): shape (B, C, H, W)
        labels (Tensor or np.ndarray): shape (B, ...)
        indices (list or array): indices to show
    """
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    n = images.shape[0]

    fig, axs = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axs = axs.flatten() if n > 1 else [axs]

    for i, (img, label, desc) in enumerate(zip(images, labels, descriptions)):
        img_np = img.permute(1, 2, 0).numpy()
        label_str = np.array2string(np.asarray(label), precision=3, separator=', ')
        axs[i].imshow(img_np, vmin=vmin, vmax=vmax, cmap='gray' if img_np.shape[-1] == 1 else None)
        axs[i].set_title(f"{title_prefix}: {desc}, {label_str}", fontsize=9)
        axs[i].axis("off")

    for j in range(n, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_predictions(x_test, y_test, y_pred, vmin=-1, vmax=1, rows=3, cols=3):
    true_labels = y_test
    images = x_test
    pred_labels = y_pred

    num_images = images.shape[0]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i in range(rows * cols):
        if i < num_images:
            ax = axes[i]
            img = images[i, :].permute(1, 2, 0).numpy()
            true_label = true_labels[i].item()
            pred_label = pred_labels[i].item()

            # Show the image
            ax.imshow(img, vmin=vmin, vmax=vmax, cmap='gray' if img.shape[-1] == 1 else None)
            ax.axis("off")

            # Set title with correct or incorrect prediction
            title_color = "green" if true_labels[i] == pred_labels[i] else "red"
            ax.set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=10, color=title_color)

    plt.tight_layout()
    plt.show()


def plot_loss(ob_val, acc):
    def moving_average(x, window=50):
        return np.convolve(x, np.ones(window) / window, mode='valid')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    # --- Plot Loss ---
    ax1.plot(ob_val, linewidth=1, alpha=0.5, color='blue')
    smoothed_loss = moving_average(ob_val)
    ax1.plot(range(len(smoothed_loss)), smoothed_loss, linewidth=2, color='darkblue')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss over Training Steps')
    ax1.set_ylim([0, 1.0])
    ax1.set_xlim([0, len(ob_val)])

    # --- Plot Accuracy ---
    ax2.plot(acc, linewidth=1, alpha=0.5, color='red', label='Raw Accuracy')
    smoothed_acc = moving_average(acc)
    ax2.plot(range(len(smoothed_acc)), smoothed_acc, linewidth=2, color='darkred')
    ax2.set_title('Accuracy over Training Steps')

    # Add horizontal milestone lines
    milestones = [0.5, 0.75]
    for milestone in milestones:
        ax2.axhline(y=milestone, color='gray', linestyle='--', linewidth=1)
        ax2.text(len(acc) * 0.01, milestone + 0.01, f"{milestone:.1%}", color='gray', fontsize=8)

    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0.25, 1.0])
    ax2.set_xlim([0, len(acc)])

    plt.tight_layout()
    # plt.pause(0.001)

#########################################################
# Functions primarily used in diffusion.ipynb
#########################################################

def visualize_diffusion_manual(pipeline, k_steps=50, l_snapshots=5):
    """
    Manually runs the DDPM reverse process and plots l results.
    """
    device = pipeline.device
    unet = pipeline.unet
    scheduler = pipeline.scheduler

    # 1. Initialize with pure Gaussian noise: x_T ~ N(0, I)
    # MNIST size is 1 channel, 28x28
    image = torch.randn((1, 1, 28, 28)).to(device)

    # 2. Set the number of inference steps
    scheduler.set_timesteps(k_steps)

    # Calculate which indices to capture
    snapshot_indices = torch.linspace(0, k_steps - 1, l_snapshots, dtype=torch.long)

    intermediates = []
    noise_preds = []
    noise_scales = []

    # 3. The Reverse Process Loop
    for i, t in enumerate(scheduler.timesteps):
        with torch.no_grad():
            # Predict noise residual: epsilon_theta(x_t, t)
            noise_pred = unet(image, t).sample

            # Compute previous sample: x_{t-1}
            # This handles the alpha/beta scaling and adding noise back
            image = scheduler.step(noise_pred, t, image).prev_sample

        # Capture snapshot
        if i in snapshot_indices:
            intermediates.append(image.detach().cpu().clone())
            noise_preds.append(noise_pred.detach().cpu().clone())

            alpha_bar_t = scheduler.alphas_cumprod[t]
            noise_scale = alpha_bar_t
            noise_scales.append(noise_scale.detach().cpu().clone())

    # 4. Final Result
    intermediates.append(image.detach().cpu())
    noise_preds.append(noise_pred.detach().cpu().clone())

    alpha_bar_t = scheduler.alphas_cumprod[t]
    noise_scale = alpha_bar_t
    noise_scales.append(noise_scale.detach().cpu().clone())

    # 5. Plotting
    num_plots = len(intermediates)
    fig, axs = plt.subplots(1, num_plots, figsize=(3 * num_plots, 3))

    for idx, tensor in enumerate(intermediates):
        img_data = tensor.squeeze().numpy()
        # Rescale for visualization
        img_data = (img_data - img_data.min()) / (
            img_data.max() - img_data.min() + 1e-8
        )

        axs[idx].imshow(img_data, cmap="gray")
        axs[idx].set_title(
            f"Step {snapshot_indices[idx] if idx < l_snapshots else k_steps}"
        )
        axs[idx].axis("off")

    plt.show()
    return intermediates, noise_preds, noise_scales


def plot_pictures(
    images,
    titles,
    super_title="diffusion",
    idx_frac=[0.0, 0.5, 1.0],
    same_scale=True,
    save_individual=False,
):
    # 1. Scaling Logic
    if same_scale:
        # Using quantiles as you suggested to avoid outlier stretching
        v_min, v_max = torch.quantile(images, 0.05), torch.quantile(images, 0.95)
    else:
        v_min, v_max = None, None

    n = images.shape[0] - 1
    indices = [int(n * f) for f in idx_frac]
    imlist = [images[idx] for idx in indices]

    # 2. Plotting Grid
    fig, axes = plt.subplots(1, len(idx_frac), figsize=(15, 5))
    fig.suptitle(super_title, fontsize=16)  # Add the Super Title

    for i, (img, tit) in enumerate(zip(imlist, titles)):
        axes[i].imshow(img.squeeze().cpu(), cmap="gray", vmin=v_min, vmax=v_max)
        axes[i].set_title(tit)
        axes[i].axis("off")

        # 3. Save Individual Images (Optional)
        if save_individual:
            filename = f"{super_title}_{tit.replace(' ', '_')}.png"
            # We normalize to [0, 1] for saving if v_min/v_max are provided
            save_image(
                img,
                filename,
                normalize=True,
                value_range=(v_min, v_max) if same_scale else None,
            )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
    plt.show()