import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import time

def analyze_models(model_results):
    """Analyze and compare model performance with improved visualizations."""
    
    # Extract model data with a more efficient approach
    models_data = {}
    for model_key, model_data in model_results.items():
        if not all(k in model_data for k in ['test_acc', 'test_loss', 'history', 'class_acc', 'model']):
            continue
            
        # Convert model key to a display name
        display_name = model_key.replace('_', ' ')
        models_data[display_name] = {
            'test_acc': model_data['test_acc'],
            'test_loss': model_data['test_loss'],
            'history': model_data['history'],
            'class_acc': model_data['class_acc'],
            'params': sum(p.numel() for p in model_data['model'].parameters()),
            'epoch_time': np.mean(model_data['history']['epoch_time'])
        }
    
    # Create summary DataFrame with sorted results
    summary_df = pd.DataFrame({
        'Model': list(models_data.keys()),
        'Test Accuracy': [data['test_acc'] for data in models_data.values()],
        'Test Loss': [data['test_loss'] for data in models_data.values()],
        'Avg Epoch Time (s)': [data['epoch_time'] for data in models_data.values()],
        'Parameters': [data['params'] for data in models_data.values()]
    })
    
    # Calculate efficiency metrics
    summary_df['Accuracy/Param Ratio'] = summary_df['Test Accuracy'] / summary_df['Parameters'] * 10**6  # Scale for readability
    summary_df['Accuracy/Time Ratio'] = summary_df['Test Accuracy'] / summary_df['Avg Epoch Time (s)']
    
    # Sort by accuracy for better visualization
    summary_df = summary_df.sort_values('Test Accuracy', ascending=False).reset_index(drop=True)
    
    # Set display formatting
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    # Set a consistent color palette with distinct colors
    model_names = summary_df['Model'].tolist()
    palette = sns.color_palette("husl", len(model_names))
    colors_dict = dict(zip(model_names, palette))
    
    # Display summary statistics
    print("=" * 80)
    print("MODEL PERFORMANCE SUMMARY:")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("\n")
    
    # Create visualizations
    create_visualizations(models_data, summary_df, colors_dict)
    
    # Print key insights
    print_insights(summary_df, models_data)

def create_visualizations(models_data, summary_df, colors_dict):
    """Create enhanced visualizations for model comparison."""
    
    # 1. Test metrics comparison with error bars
    fig, ax = plt.subplots(1, 2, figsize=(18, 7))
    
    # Sort by accuracy for consistent ordering
    model_names = summary_df['Model'].tolist()
    test_accs = summary_df['Test Accuracy'].tolist()
    test_losses = summary_df['Test Loss'].tolist()
    
    # Accuracy plot with percentage formatting
    bars1 = ax[0].bar(model_names, test_accs, color=[colors_dict[m] for m in model_names])
    ax[0].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax[0].set_ylabel('Accuracy', fontsize=12)
    ax[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    ax[0].set_ylim(min(0.95, min(test_accs) - 0.01), 1.0)
    ax[0].grid(axis='y', alpha=0.3)
    ax[0].set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add value annotations to bars
    for bar in bars1:
        height = bar.get_height()
        ax[0].text(bar.get_x() + bar.get_width()/2, height + 0.001,
                f'{height:.2%}', ha='center', va='bottom', fontweight='bold')
    
    # Loss plot
    bars2 = ax[1].bar(model_names, test_losses, color=[colors_dict[m] for m in model_names])
    ax[1].set_title('Test Loss Comparison', fontsize=14, fontweight='bold')
    ax[1].set_ylabel('Loss', fontsize=12)
    ax[1].grid(axis='y', alpha=0.3)
    ax[1].set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add value annotations to bars
    for bar in bars2:
        height = bar.get_height()
        ax[1].text(bar.get_x() + bar.get_width()/2, height + 0.001,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Enhanced per-class accuracy heatmap
    class_acc_matrix = np.vstack([data['class_acc'] for data in models_data.values()])
    model_order = summary_df['Model'].tolist()  # Keep consistent order
    
    # Map the data to the correct order
    ordered_indices = [list(models_data.keys()).index(model) for model in model_order]
    ordered_class_acc = class_acc_matrix[ordered_indices]
    
    # Create DataFrame with proper ordering
    class_acc_df = pd.DataFrame(
        ordered_class_acc,
        index=model_order,
        columns=[f'Digit {i}' for i in range(10)]
    )
    
    plt.figure(figsize=(14, len(models_data) * 0.7 + 3))
    
    # Use a more intuitive colormap and center it at the mean accuracy
    mean_acc = np.mean(class_acc_matrix)
    vmin = max(0, mean_acc - 0.1)
    vmax = min(1, mean_acc + 0.1)
    
    # Create heatmap with better contrast
    ax = sns.heatmap(class_acc_df, annot=True, cmap="RdYlGn", fmt=".2%", 
                   linewidths=0.5, vmin=vmin, vmax=vmax, annot_kws={"fontsize":9})
    
    plt.title('Per-Class Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Model', fontsize=12)
    plt.xlabel('Digit Class', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # 3. Enhanced model complexity vs performance analysis
    plt.figure(figsize=(14, 8))
    
    # Get data for plotting
    names = summary_df['Model'].tolist()
    accuracies = summary_df['Test Accuracy'].tolist()
    params = summary_df['Parameters'].tolist()
    times = summary_df['Avg Epoch Time (s)'].tolist()
    
    # Scale for size of bubbles - normalize to reasonable range
    size_scale = np.array(times) / max(times) * 1000 + 100
    
    # Create scatter plot with size representing training time
    scatter = plt.scatter(params, accuracies, s=size_scale, 
                         c=[colors_dict[m] for m in names], alpha=0.7)
    
    # Add annotations
    for i, name in enumerate(names):
        plt.annotate(name, (params[i], accuracies[i]),
                    textcoords="offset points", xytext=(5, 5),
                    ha='left', fontsize=10, fontweight='bold')
    
    plt.title('Model Complexity vs Performance Trade-off', fontsize=16, fontweight='bold')
    plt.xscale('log')  # Log scale for better visualization of parameter counts
    plt.xlabel('Number of Parameters (log scale)', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add a legend for the size of points
    handles, labels = [], []
    sizes = [min(times), np.median(times), max(times)]
    for size in sizes:
        handles.append(plt.scatter([], [], s=size/max(times)*1000+100, color='gray', alpha=0.7))
        labels.append(f'{size:.2f}s/epoch')
    plt.legend(handles, labels, title="Training Time", loc="lower right", title_fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # 4. Learning curves - organized in a more readable grid
    plot_learning_curves(models_data, summary_df, colors_dict)
    
    # 5. Efficiency metrics visualization
    plt.figure(figsize=(14, 8))
    
    x = summary_df['Model']
    y1 = summary_df['Accuracy/Param Ratio']
    y2 = summary_df['Accuracy/Time Ratio']
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot the first metric
    color = 'tab:blue'
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy per Million Parameters', color=color)
    bars1 = ax1.bar(x, y1, color=color, alpha=0.7, label='Parameter Efficiency')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='x', rotation=45)
    
    # Create the second axis and plot the second metric
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Accuracy per Second', color=color)
    bars2 = ax2.bar(x, y2, color=color, alpha=0.3, label='Time Efficiency')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add title and legend
    plt.title('Model Efficiency Metrics', fontsize=16, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.show()

def plot_learning_curves(models_data, summary_df, colors_dict):
    """Plot enhanced learning curves with better layout and design."""
    
    # Get the model order from summary DataFrame
    model_order = summary_df['Model'].tolist()
    
    # Calculate layout
    n_models = len(model_order)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    fig.suptitle('Training and Validation Curves', fontsize=18, fontweight='bold', y=0.98)
    
    # Flatten axes array for easier indexing if multiple rows
    if n_rows > 1:
        axes = axes.flatten()
    elif n_cols == 1:
        axes = [axes]  # Make it iterable if only one subplot
    
    # Plot each model's learning curves
    for i, model_name in enumerate(model_order):
        ax = axes[i] if n_models > 1 else axes
        history = models_data[model_name]['history']
        
        # Plot training and validation accuracy
        ax.plot(history['train_acc'], label='Training', 
               color=colors_dict[model_name], linewidth=2)
        ax.plot(history['val_acc'], label='Validation', 
               color=colors_dict[model_name], linestyle='--', linewidth=2, alpha=0.7)
        
        # Add loss curves if available (on secondary y-axis)
        if 'train_loss' in history and 'val_loss' in history:
            ax2 = ax.twinx()
            ax2.plot(history['train_loss'], label='Train Loss', 
                    color='gray', linewidth=1, alpha=0.5)
            ax2.plot(history['val_loss'], label='Val Loss', 
                    color='black', linewidth=1, alpha=0.5, linestyle=':')
            ax2.set_ylabel('Loss', color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')
        
        ax.set_title(model_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        
        # Set y-axis to show percentages
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        if n_models > 1:
            axes[j].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def print_insights(summary_df, models_data):
    """Print insightful analysis of model performance."""
    
    print("=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    
    # 1. Best performing models
    best_acc_model = summary_df.iloc[0]['Model']
    best_acc = summary_df.iloc[0]['Test Accuracy']
    print(f"1. Best performing model: {best_acc_model} with {best_acc:.2%} accuracy")
    
    # 2. Most parameter-efficient model
    param_efficient_model = summary_df['Accuracy/Param Ratio'].idxmax()
    param_efficiency = summary_df.loc[param_efficient_model, 'Accuracy/Param Ratio']
    param_efficient_name = summary_df.loc[param_efficient_model, 'Model']
    print(f"2. Most parameter-efficient model: {param_efficient_name} with {param_efficiency:.4f} accuracy per million parameters")
    
    # 3. Most time-efficient model
    time_efficient_model = summary_df['Accuracy/Time Ratio'].idxmax()
    time_efficiency = summary_df.loc[time_efficient_model, 'Accuracy/Time Ratio']
    time_efficient_name = summary_df.loc[time_efficient_model, 'Model']
    print(f"3. Most time-efficient model: {time_efficient_name} with {time_efficiency:.4f} accuracy per second")
    
    # 4. Challenging digits analysis
    print("\n4. Per-digit performance analysis:")
    
    # Get most challenging digit across all models
    all_class_acc = np.vstack([data['class_acc'] for data in models_data.values()])
    avg_per_digit = np.mean(all_class_acc, axis=0)
    hardest_digit = np.argmin(avg_per_digit)
    easiest_digit = np.argmax(avg_per_digit)
    
    print(f"   - Most challenging digit across all models: Digit {hardest_digit} (avg accuracy: {avg_per_digit[hardest_digit]:.2%})")
    print(f"   - Easiest digit across all models: Digit {easiest_digit} (avg accuracy: {avg_per_digit[easiest_digit]:.2%})")
    
    # Analyze per-model challenges
    print("\n   Per-model digit challenges:")
    for model_name, data in models_data.items():
        hardest_digit = np.argmin(data['class_acc'])
        print(f"   - {model_name}: Digit {hardest_digit} ({data['class_acc'][hardest_digit]:.2%} accuracy)")
    
    # 5. Model architecture insights
    cnn_models = [m for m in summary_df['Model'] if 'CNN' in m]
    mlp_models = [m for m in summary_df['Model'] if 'CNN' not in m]
    
    if cnn_models and mlp_models:
        cnn_avg_acc = summary_df[summary_df['Model'].isin(cnn_models)]['Test Accuracy'].mean()
        mlp_avg_acc = summary_df[summary_df['Model'].isin(mlp_models)]['Test Accuracy'].mean()
        
        print(f"\n5. Architecture comparison:")
        print(f"   - CNN models average accuracy: {cnn_avg_acc:.2%}")
        print(f"   - MLP models average accuracy: {mlp_avg_acc:.2%}")
        print(f"   - Advantage: {'CNN' if cnn_avg_acc > mlp_avg_acc else 'MLP'} by {abs(cnn_avg_acc - mlp_avg_acc):.2%}")
    
    # 6. Training observations
    epochs_to_convergence = {}
    for model_name, data in models_data.items():
        history = data['history']
        # Find where validation accuracy plateaus (within 0.5% of max for 3 epochs)
        val_acc = np.array(history['val_acc'])
        max_acc = np.max(val_acc)
        converged_at = np.where(val_acc >= max_acc - 0.005)[0]
        if len(converged_at) > 0:
            epochs_to_convergence[model_name] = converged_at[0] + 1  # +1 because epochs are 1-indexed
    
    if epochs_to_convergence:
        fastest_convergence = min(epochs_to_convergence.items(), key=lambda x: x[1])
        slowest_convergence = max(epochs_to_convergence.items(), key=lambda x: x[1])
        
        print(f"\n6. Training convergence:")
        print(f"   - Fastest convergence: {fastest_convergence[0]} ({fastest_convergence[1]} epochs)")
        print(f"   - Slowest convergence: {slowest_convergence[0]} ({slowest_convergence[1]} epochs)")
    
    # 7. Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    
    if cnn_models and mlp_models and cnn_avg_acc > mlp_avg_acc:
        print("CNN architectures outperform MLP models for this image classification task, demonstrating")
        print("the effectiveness of convolutional layers for capturing spatial patterns in images.")
    
    # Find the best parameter-to-performance ratio model
    best_efficiency_model = summary_df.iloc[summary_df['Accuracy/Param Ratio'].idxmax()]['Model']
    print(f"\nThe {best_efficiency_model} offers the best balance of accuracy and model complexity,")
    print("making it an excellent choice when computational resources are limited.")
    
    print("\nFor deployment decisions, consider the trade-offs between:")
    print("1. Accuracy requirements")
    print("2. Inference time constraints")
    print("3. Memory footprint limitations")
    print("4. Training cost considerations")

