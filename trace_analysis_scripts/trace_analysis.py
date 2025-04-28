import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
import glob
from collections import defaultdict
import networkx as nx

def process_csv_file(file_path):
    """
    Process a single CSV file to extract layer ID, token, and activated expert IDs.
    """
    try:
        # Try different encodings if necessary
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin-1')
        
        # Check if the CSV has the expected columns
        required_columns = ['Layer ID', 'Batch Number', 'Token', 'Activated Expert IDs']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Columns {missing_columns} not found in {file_path}")
            return None, None
        
        # Convert the string representation of lists to actual lists
        df['Activated Expert IDs'] = df['Activated Expert IDs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
       
        if df['Layer ID'].min() == 1:
            df['Layer ID'] = df['Layer ID'] - 1

        # Extract the filename without extension for reference
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        return df, filename
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def create_expert_activation_matrix(dataframes, max_experts=None):
    """
    Create a matrix showing the frequency of expert activations across layers.
    """
    all_data = pd.concat([df for df, _ in dataframes], ignore_index=True)
    
    # Find the maximum expert ID if not provided
    if max_experts is None:
        max_expert_id = 0
        for experts in all_data['Activated Expert IDs']:
            if experts and len(experts) > 0:
                max_expert_id = max(max_expert_id, max(experts))
        max_experts = max_expert_id + 1
    
    # Find the maximum layer ID
    max_layer_id = all_data['Layer ID'].max()
    
    # Create matrices to hold activation counts
    overall_matrix = np.zeros((max_layer_id + 1, max_experts))
    per_prompt_matrices = {}
    
    # Process each prompt's dataframe
    for df, prompt_name in dataframes:
        prompt_matrix = np.zeros((max_layer_id + 1, max_experts))
        
        # Count activations of each expert in each layer
        for layer_id, experts in zip(df['Layer ID'], df['Activated Expert IDs']):
            if experts and len(experts) > 0:
                for expert_id in experts:
                    if expert_id < max_experts:  # Ensure expert ID is within bounds
                        prompt_matrix[layer_id, expert_id] += 1
                        overall_matrix[layer_id, expert_id] += 1
        
        per_prompt_matrices[prompt_name] = prompt_matrix
    
    return overall_matrix, per_prompt_matrices, max_layer_id, max_experts

def plot_expert_heatmap(matrix, title, output_path=None, layer_ids=None, expert_ids=None):
    """
    Create and display a heatmap of expert activations.
    """
    plt.figure(figsize=(14, 10))
    
    # Create the heatmap with appropriate tick labels
    ax = sns.heatmap(matrix, cmap="YlGnBu", 
                xticklabels=expert_ids if expert_ids is not None else 10, 
                yticklabels=layer_ids if layer_ids is not None else 1,
                cbar_kws={'label': 'Number of activations'})
    
    # Set tick labels with appropriate font size
    if expert_ids is not None:
        ax.set_xticks(np.arange(len(expert_ids)) + 0.5)
        ax.set_xticklabels(expert_ids, rotation=90, fontsize=8)
    
    if layer_ids is not None:
        ax.set_yticks(np.arange(len(layer_ids)) + 0.5)
        ax.set_yticklabels(layer_ids, fontsize=8)
    
    plt.xlabel('Expert ID')
    plt.ylabel('Layer ID')
    plt.title(title)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_aggregated_per_layer_expert_bars(overall_matrix, layer_ids, expert_ids, output_dir=None):
    """
    Generate per-layer bar plots showing aggregated expert activations across all prompts.
    """
    for layer_id in layer_ids:
        expert_activations = overall_matrix[layer_id, :]
        if np.sum(expert_activations) == 0:
            continue  # Skip empty layers

        plt.figure(figsize=(10, 4))
        plt.bar(expert_ids, expert_activations)
        plt.xlabel('Expert ID')
        plt.ylabel('Total Activations')
        plt.title(f'Aggregated Expert Activations - Layer {layer_id}')
        plt.tight_layout()

        if output_dir:
            filename = f"aggregated_layer_{layer_id}_expert_activations.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def analyze_token_patterns(dataframes, output_dir=None):
    """
    Analyze patterns in token-expert relationships.
    """
    # Combine all dataframes
    all_data = pd.concat([df for df, _ in dataframes], ignore_index=True)
    
    # Create a dictionary to track which experts are activated for each token
    token_to_experts = defaultdict(list)
    
    for token, experts in zip(all_data['Token'], all_data['Activated Expert IDs']):
        if experts and len(experts) > 0:
            token_to_experts[token].extend(experts)
    
    # Calculate the most common experts for each token
    token_expert_counts = {}
    for token, experts in token_to_experts.items():
        expert_counts = pd.Series(experts).value_counts().to_dict()
        token_expert_counts[token] = expert_counts
    
    # Create a detailed report
    report = []
    for token, expert_counts in token_expert_counts.items():
        total_activations = sum(expert_counts.values())
        top_experts = sorted(expert_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        report.append({
            'Token': token,
            'Total Activations': total_activations,
            'Top Experts': repr(top_experts)  # Convert to string for CSV compatibility
        })
    
    # Convert to DataFrame for easier analysis
    report_df = pd.DataFrame(report)
    
    # Sort by total activations
    report_df = report_df.sort_values('Total Activations', ascending=False)
    
    if output_dir:
        report_df.to_csv(os.path.join(output_dir, 'token_expert_report.csv'), index=False)
    
    return report_df

def analyze_expert_coactivation(dataframes, output_dir=None, max_experts=None):
    """
    Analyze which experts tend to be activated together.
    """
    all_data = pd.concat([df for df, _ in dataframes], ignore_index=True)
    
    # Find the maximum expert ID if not provided
    if max_experts is None:
        max_expert_id = 0
        for experts in all_data['Activated Expert IDs']:
            if experts and len(experts) > 0:
                max_expert_id = max(max_expert_id, max(experts))
        max_experts = max_expert_id + 1
    
    # Create a co-activation matrix
    coactivation_matrix = np.zeros((max_experts, max_experts))
    
    # Count co-activations
    for experts in all_data['Activated Expert IDs']:
        if experts and len(experts) > 1:  # At least 2 experts activated
            for i in range(len(experts)):
                for j in range(i+1, len(experts)):
                    expert1 = experts[i]
                    expert2 = experts[j]
                    if expert1 < max_experts and expert2 < max_experts:
                        coactivation_matrix[expert1, expert2] += 1
                        coactivation_matrix[expert2, expert1] += 1  # Symmetric matrix
    
    # Create a heatmap of co-activations
    plt.figure(figsize=(14, 12))
    mask = np.zeros_like(coactivation_matrix, dtype=bool)
    mask[np.diag_indices_from(mask)] = True  # Mask the diagonal
    
    # Use a logarithmic color scale for better visualization
    with np.errstate(divide='ignore'):  # Suppress divide by zero warning
        log_matrix = np.log1p(coactivation_matrix)  # log(1+x) to handle zeros
    
    ax = sns.heatmap(log_matrix, mask=mask, cmap="YlGnBu", 
                    xticklabels=20, yticklabels=20,
                    cbar_kws={'label': 'Log(Number of co-activations + 1)'})
    
    plt.xlabel('Expert ID')
    plt.ylabel('Expert ID')
    plt.title('Expert Co-activation Patterns (Log Scale)')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'expert_coactivation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # Create a network graph for top co-activations
    G = nx.Graph()
    
    # Add nodes for all experts
    for expert_id in range(max_experts):
        G.add_node(expert_id)
    
    # Find the threshold for top 10% of co-activations
    nonzero_coactivations = coactivation_matrix[coactivation_matrix > 0]
    if len(nonzero_coactivations) > 0:
        threshold = np.percentile(nonzero_coactivations, 90)
        
        # Add edges for strong co-activations
        for i in range(max_experts):
            for j in range(i+1, max_experts):
                weight = coactivation_matrix[i, j]
                if weight > threshold:
                    G.add_edge(i, j, weight=weight)
        
        # Plot the network
        plt.figure(figsize=(14, 14))
        pos = nx.spring_layout(G, seed=42)  # Positions for all nodes
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=50, alpha=0.8)
        
        # Draw edges with varying thickness based on weight
        edges = G.edges(data=True)
        weights = [data['weight'] for _, _, data in edges]
        nx.draw_networkx_edges(G, pos, width=[w/max(weights)*5 for w in weights], alpha=0.5)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title('Expert Co-activation Network (Strong Connections)')
        plt.axis('off')  # Turn off axis
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'expert_coactivation_network.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    return coactivation_matrix

def plot_per_layer_expert_bars(per_prompt_matrices, layer_ids, expert_ids, output_dir=None):
    """
    Generate per-layer bar plots showing expert activations for each prompt.
    """
    for prompt_name, matrix in per_prompt_matrices.items():
        for layer_id in layer_ids:
            expert_activations = matrix[layer_id, :]
            if np.sum(expert_activations) == 0:
                continue  # Skip empty layers

            plt.figure(figsize=(10, 4))
            plt.bar(expert_ids, expert_activations)
            plt.xlabel('Expert ID')
            plt.ylabel('Number of Activations')
            plt.title(f'Expert Activations - {prompt_name} - Layer {layer_id}')
            plt.tight_layout()

            if output_dir:
                filename = f"{prompt_name}_layer_{layer_id}_expert_activations.png"
                plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

def analyze_layer_expert_distribution(overall_matrix, output_dir=None):
    """
    Analyze the distribution of expert utilization in each layer.
    """
    # Calculate the number of unique experts activated in each layer
    unique_experts_per_layer = np.count_nonzero(overall_matrix > 0, axis=1)
    
    # Calculate the Gini coefficient for each layer to measure inequality in expert usage
    gini_coefficients = []
    for layer_id in range(overall_matrix.shape[0]):
        layer_activations = overall_matrix[layer_id, :]
        # Skip layers with no activations
        if np.sum(layer_activations) == 0:
            gini_coefficients.append(0)
            continue
        
        # Sort the activations
        sorted_activations = np.sort(layer_activations)
        n = len(sorted_activations)
        
        # Calculate Gini coefficient
        cumsum = np.cumsum(sorted_activations)
        gini = 1 - 2 * np.sum((cumsum - sorted_activations/2) / cumsum[-1]) / n
        gini_coefficients.append(gini)
    
    # Plot the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot number of unique experts per layer
    ax1.bar(range(len(unique_experts_per_layer)), unique_experts_per_layer)
    ax1.set_ylabel('Number of Unique Experts')
    ax1.set_title('Number of Unique Experts Activated per Layer')
    ax1.grid(True, alpha=0.3)
    
    # Plot Gini coefficients per layer
    ax2.bar(range(len(gini_coefficients)), gini_coefficients)
    ax2.set_xlabel('Layer ID')
    ax2.set_ylabel('Gini Coefficient')
    ax2.set_title('Inequality in Expert Usage per Layer (Gini Coefficient)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'layer_expert_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return unique_experts_per_layer, gini_coefficients

def plot_layer_activations(matrix, layer_ids, title, output_path=None):
    """
    Create a bar graph of total expert activations per layer
    """
    layer_activations = np.sum(matrix, axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(layer_ids)), layer_activations)
    plt.xlabel('Layer ID')
    plt.ylabel('Total Expert Activations')
    plt.title(title)
    plt.xticks(range(len(layer_ids)), layer_ids, rotation=90)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def analyze_moe_activations(directory_path, output_dir=None, max_experts=None):
    """
    Analyze expert activations from CSV files in the provided directory.
    """
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return
    
    print(f"Found {len(csv_files)} CSV files to analyze")
    
    # Process each CSV file
    dataframes = []
    for file_path in csv_files:
        df, filename = process_csv_file(file_path)
        if df is not None:
            dataframes.append((df, filename))
    
    if not dataframes:
        print("No valid data found in the CSV files")
        return
    
    print(f"Successfully processed {len(dataframes)} CSV files")
    
    # Create the activation matrices
    overall_matrix, per_prompt_matrices, max_layer_id, max_experts = create_expert_activation_matrix(dataframes, max_experts)
    
    print(f"Found {max_experts} experts across {max_layer_id + 1} layers")
    
    # Create a list of layer IDs and expert IDs for better labeling
    layer_ids = list(range(max_layer_id + 1))
    expert_ids = list(range(max_experts))
    
    # Plot the overall heatmap
    if output_dir:
        output_path = os.path.join(output_dir, 'overall_heatmap.png')
    else:
        output_path = None
    plot_expert_heatmap(overall_matrix, 'Overall Expert Activations Across Layers', 
                      output_path, layer_ids, expert_ids)
    
    print("Created overall activation heatmap")
    
    # Plot individual prompt heatmaps
    #for prompt_name, matrix in per_prompt_matrices.items():
    #    if output_dir:
    #        output_path = os.path.join(output_dir, f'{prompt_name}_heatmap.png')
    #    else:
    #        output_path = None
    #    plot_expert_heatmap(matrix, f'Expert Activations for {prompt_name}', 
    #                     output_path, layer_ids, expert_ids)
    
    # Plot individual prompt layer activations
    #for prompt_name, matrix in per_prompt_matrices.items():
    #    output_path = os.path.join(output_dir, f'{prompt_name}_layer_activations.png') if output_dir else None
    #    plot_layer_activations(matrix, layer_ids, 
    #                     f'Expert Activations per Layer ({prompt_name})', 
    #                     output_path)

    #    # Plot aggregated layer activations
    #    output_path = os.path.join(output_dir, 'aggregated_layer_activations.png') if output_dir else None
    #    plot_layer_activations(overall_matrix, layer_ids,
    #                 'Aggregated Expert Activations Across All Prompts',
    #                 output_path)

    #print(f"Created individual heatmaps for {len(per_prompt_matrices)} prompts")
    
    #plot_per_layer_expert_bars(per_prompt_matrices, layer_ids, expert_ids, output_dir)
    #print("Created per prompt expert activation bar plots per layer")

    plot_aggregated_per_layer_expert_bars(overall_matrix, layer_ids, expert_ids, output_dir)
    print("Created aggregated expert activation bar plots per layer")

    # Analyze token patterns
    #token_report = analyze_token_patterns(dataframes, output_dir)
    #print(f"Analyzed token-expert patterns for {len(token_report)} unique tokens")
    
    # Analyze expert co-activation
    #coactivation_matrix = analyze_expert_coactivation(dataframes, output_dir, max_experts)
    #print("Analyzed expert co-activation patterns")
    
    # Analyze layer-wise expert distribution
    #unique_experts, gini_coefficients = analyze_layer_expert_distribution(overall_matrix, output_dir)
    #print("Analyzed layer-wise expert distribution")
    
    return {
        'overall_matrix': overall_matrix,
        'per_prompt_matrices': per_prompt_matrices,
        'token_report': token_report,
        'coactivation_matrix': coactivation_matrix,
        'unique_experts_per_layer': unique_experts,
        'gini_coefficients': gini_coefficients
    }

def main():
    """
    Main function to parse command line arguments and run the analysis.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze MoE expert activations from CSV files.')
    parser.add_argument('directory', help='Directory containing CSV files to analyze')
    parser.add_argument('--output', '-o', help='Directory to save output files', default=None)
    parser.add_argument('--max-experts', '-m', type=int, help='Maximum number of experts to consider', default=None)
    
    args = parser.parse_args()
    
    analyze_moe_activations(args.directory, args.output, args.max_experts)
    print("Analysis complete!")

if __name__ == "__main__":
    main()
