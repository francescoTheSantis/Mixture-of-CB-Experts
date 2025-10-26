import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patheffects as path_effects

# --- Create some sample data ---
np.random.seed(0)
n = 50

x = np.random.rand(n) * 100        # e.g. GDP
y = np.random.rand(n) * 80         # e.g. Life expectancy
z = np.random.rand(n) * 1000       # e.g. Population (third variable)

# --- Create a DataFrame (optional, for clarity) ---
data = pd.DataFrame({
    'GDP': x,
    'Life Expectancy': y,
    'Population': z
})

# --- Plot ---
plt.figure(figsize=(8,6))
scatter = plt.scatter(
    data['GDP'],
    data['Life Expectancy'],
    s=data['Population'] / 10,   # divide to scale bubble size
    c=data['Population'],        # optional: color by same or another variable
    cmap='viridis',
    alpha=0.6,
    edgecolors='w',
    linewidth=0.5
)

# Annotate each bubble with its population (rounded). You can change the label to use index: str(i)
for gx, ly, pop in zip(data['GDP'], data['Life Expectancy'], data['Population']):
    txt = plt.text(gx, ly, f"{int(round(pop))}", fontsize=7, ha='center', va='center', color='black', weight='bold')
    # add a white stroke to improve contrast against the bubble color
    txt.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='white'), path_effects.Normal()])

plt.title("Bubble Plot Example: 3 Variables in 2D (labels shown)")
plt.xlabel("GDP")
plt.ylabel("Life Expectancy")
plt.colorbar(scatter, label="Population (color scale)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("bubble_plot_example.pdf")
