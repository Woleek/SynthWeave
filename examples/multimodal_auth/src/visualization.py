from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.manifold import TSNE


def plot_tsne_figure(embeddings, labels, preds, pids):
    tsne = TSNE(n_components=2, perplexity=30, max_iter=500, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    # Helper to generate color maps for any class set
    def get_color_map(classes):
        unique_classes = np.unique(classes)
        cmap = plt.get_cmap('tab10' if len(unique_classes) <= 10 else 'tab20')
        return {cls: cmap(i % cmap.N) for i, cls in enumerate(unique_classes)}

    def plot_panel(ax, values, title, color_map, show_legend=False):
        for cls in np.unique(values):
            idx = (values == cls)
            ax.scatter(emb_2d[idx, 0], emb_2d[idx, 1], s=30, color=color_map[cls], label=str(cls), alpha=0.8)
        ax.set_title(title)
        ax.axis('off')
        if show_legend:
            legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[cls], label=str(cls), markersize=8)
                               for cls in np.unique(values)]
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Set up color maps
    label_colors = get_color_map(labels)
    pred_colors = get_color_map(preds)
    pid_colors = get_color_map(pids)

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    plot_panel(axs[0], labels, "Labels", label_colors, show_legend=True)
    plot_panel(axs[1], preds, "Predictions", pred_colors)
    plot_panel(axs[2], pids, "Person Identity (PID)", pid_colors)

    fig.tight_layout()
    return fig