"""Worm graph visualization for FPL backtesting."""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def plot_worm_graph(actual_points, model_paths, output_path, current_gw=31):
    """
    Generate a cricket-style worm graph comparing actual vs model paths.
    
    actual_points: dict {gw: points}
    model_paths: dict {switch_gw: {gw: points}}
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    gws = list(range(1, current_gw + 1))
    
    # Actual cumulative line
    actual_cum = []
    total = 0
    for gw in gws:
        total += actual_points.get(gw, 0)
        actual_cum.append(total)
    
    ax.plot(gws, actual_cum, 'k-', linewidth=2.5, label='Actual', 
            marker='o', markersize=4, zorder=10)
    
    # Model paths
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#27ae60', '#3498db', '#8e44ad']
    
    for i, switch_gw in enumerate(sorted(model_paths.keys())):
        gw_scores = model_paths[switch_gw]
        model_cum = []
        total = 0
        for gw in gws:
            total += gw_scores.get(gw, 0)
            model_cum.append(total)
        
        ax.plot(gws, model_cum, '--', color=colors[i % len(colors)], linewidth=1.8,
                label=f'Model from GW{switch_gw}', alpha=0.85, marker='s', markersize=3)
        
        # Mark the switch point
        switch_idx = switch_gw - 1
        if switch_idx < len(model_cum):
            ax.axvline(x=switch_gw, color=colors[i % len(colors)], 
                       alpha=0.2, linestyle=':', linewidth=1)
    
    ax.set_xlabel('Gameweek', fontsize=12)
    ax.set_ylabel('Cumulative Points', fontsize=12)
    ax.set_title('FPL Worm Graph: Actual vs Model-Recommended Transfers', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(gws)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📈 Worm graph saved to {output_path}")
    plt.close()
