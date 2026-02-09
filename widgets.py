import ipywidgets as widgets
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt

class ReLU_NNWidget:
    def __init__(self):
        self.x = np.linspace(-5, 5, 200)
        self.sliders = []
        self.lines = [] # Store individual neuron lines

        # 1. SETUP PLOT ONCE (Fixes flickering)
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.sum_line, = self.ax.plot([], [], 'k-', linewidth=3, label='Sum NN(x) (Piecewise Linear)')

        # Static Axis Setup
        self.ax.set_title("ReLU Network")
        self.ax.set_ylim(-5, 5)
        self.ax.set_xlim(-5, 5)
        self.ax.set_xticks(np.arange(-5, 6, 1))
        self.ax.set_yticks(np.arange(-5, 6, 1))
        self.ax.grid(True, which='both', linestyle='--', alpha=0.6)
        self.ax.axhline(0, color='grey', linewidth=1.5, zorder=0)
        self.ax.axvline(0, color='grey', linewidth=1.5, zorder=0)
        self.ax.legend(loc='upper left')

        # Prevent double display in notebook output
        plt.close(self.fig)

        # 2. Main Control
        self.num_neurons_slider = widgets.IntSlider(
            value=1, min=1, max=10,
            description='N Neurons:',
            layout=widgets.Layout(width='50%'),
            style={'description_width': 'initial'}
        )
        self.num_neurons_slider.observe(self.on_neuron_count_change, names='value')

        self.save_button = widgets.Button(
            description="Export to PDF",
            icon="download",
            button_style='info'
        )
        self.save_button.on_click(self.save_plot_to_file)

        self.controls_container = widgets.VBox([])
        self.plot_output = widgets.Output()

        self.update_slider_list(1)
        self.update_plot()

    def activation(self, x):
        return np.maximum(0, x)

    def on_neuron_count_change(self, change):
        n = change['new']
        self.update_slider_list(n)
        self.update_plot()

    def create_neuron_widgets(self, index):
        slider_layout = widgets.Layout(width='40%')
        style = {'description_width': '50px'}

        w = widgets.FloatSlider(value=1.0, min=-5, max=5, step=1.0,
                                description='', layout=slider_layout, style=style)
        b = widgets.FloatSlider(value=0.0, min=-5, max=5, step=1.0,
                                description='', layout=slider_layout, style=style)
        v = widgets.FloatSlider(value=1.0, min=-5, max=5, step=1.0,
                                description='', layout=slider_layout, style=style)

        w.observe(self.on_param_change, names='value')
        b.observe(self.on_param_change, names='value')
        v.observe(self.on_param_change, names='value')

        return {'w': w, 'b': b, 'v': v}

    def update_slider_list(self, n):
        current_count = len(self.sliders)

        # SYNC SLIDERS AND PLOT LINES
        if n > current_count:
            for i in range(current_count, n):
                # Add Slider
                self.sliders.append(self.create_neuron_widgets(i))
                # Add Line to Plot
                line, = self.ax.plot([], [], '--', alpha=0.4, linewidth=1)
                self.lines.append(line)

        elif n < current_count:
            # Remove Sliders
            self.sliders = self.sliders[:n]
            # Remove Lines from Plot
            for i in range(n, current_count):
                self.lines[i].remove() # Clear from matplotlib
            self.lines = self.lines[:n]

        ui_rows = []
        for i, s in enumerate(self.sliders):
            # Layout Construction (Your existing logic)
            label_style = widgets.Layout(width='40px', display='flex', justify_content='flex-end', margin='0px 10px 0px 0px')

            label_k = widgets.HTMLMath(value=fr"$k_{{{i+1}}}$", layout=label_style)
            label_d = widgets.HTMLMath(value=fr"$d_{{{i+1}}}$", layout=label_style)
            label_a = widgets.HTMLMath(value=fr"$a_{{{i+1}}}$", layout=label_style)

            row_k = widgets.HBox([label_k, s['w']])
            row_d = widgets.HBox([label_d, s['b']])
            row_a = widgets.HBox([label_a, s['v']])

            row = widgets.VBox([
                widgets.HTMLMath(value=fr"<b>Neuron {i+1}:</b> $a_{{{i+1}}} \cdot \text{{ReLU}}(k_{{{i+1}}} x + d_{{{i+1}}})$"),
                row_k, row_d, row_a,
                widgets.HTML("<hr style='opacity:0.2'>")
            ])
            ui_rows.append(row)
        self.controls_container.children = ui_rows

    def on_param_change(self, change):
        self.update_plot()

    def save_plot_to_file(self, b):
        self.fig.savefig("relu_network_plot.pdf", bbox_inches='tight', dpi=300)
        print(f"Saved current plot")

    def update_plot(self):
        # OPTIMIZED UPDATE: Only change data, do not recreate figure
        total_y = np.zeros_like(self.x)

        for i, s in enumerate(self.sliders):
            w_val = s['w'].value
            b_val = s['b'].value
            v_val = s['v'].value

            neuron_y = v_val * self.activation(w_val * self.x + b_val)
            total_y += neuron_y

            # Update the existing line object
            self.lines[i].set_data(self.x, neuron_y)

        # Update Sum Line
        self.sum_line.set_data(self.x, total_y)
        self.ax.set_title(f"ReLU Network with {len(self.sliders)} Neurons")

        # Refresh Output
        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            display(self.fig)

    def display(self):
        header = widgets.HBox([self.num_neurons_slider, self.save_button])
        display(header)
        display(self.plot_output)
        display(self.controls_container)