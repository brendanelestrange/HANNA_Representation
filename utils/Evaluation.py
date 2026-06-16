import matplotlib.pyplot as plt
import numpy as np
import torch
import csv
import os
class Evaluation:

    @staticmethod
    def find_highest_mae(system_wise_mae_all_dict, test_systems_ID, int_to_smiles, component_1_ID, component_2_ID, x=10):
        """
        Prints the top x systems with the highest MAE, including their SMILES strings.

        Parameters:
        - system_wise_mae_all_dict: Dictionary mapping system IDs to MAE values.
        - int_to_smiles: Dictionary mapping component IDs to their SMILES strings.
        - component_1_ID: Series or list mapping system IDs to the first component ID.
        - component_2_ID: Series or list mapping system IDs to the second component ID.
        - x: Number of top systems to display (default: 10).
        """
        # Sort the systems by MAE in descending order and get the top x
        top_x_systems = sorted(system_wise_mae_all_dict.items(), key=lambda item: item[1], reverse=True)[:x]

        print(f"Top {x} systems by MAE:")
        for system_id, mae in top_x_systems:
            system_indices = np.where(test_systems_ID == system_id)[0]
            smiles_1 = int_to_smiles[component_1_ID.iloc[system_indices[0]]]
            smiles_2 = int_to_smiles[component_2_ID.iloc[system_indices[0]]]
            print(f"System ID: {system_id}, MAE: {mae}, Component 1 SMILES: {smiles_1}, Component 2 SMILES: {smiles_2}")

    @staticmethod
    def write_predictions_to_csv(filename, predictions, targets, test_systems_ID, test_classification, mod_unifac_preds, int_to_smiles, component_1_ID, component_2_ID, classification_descriptions):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            header = ['System ID', 'System Classification', 'Component 1 SMILES', 'Component 2 SMILES', 
                    'Model ln(gamma 1)', 'Model ln(gamma 2)', 
                    'Modified UNIFAC ln(gamma 1)', 'Modified UNIFAC ln(gamma 2)',
                    'Target ln(gamma 1)', 'Target ln(gamma 2)']
            writer.writerow(header)

            for system_id in predictions.keys():
                system_model_predictions = torch.cat(predictions[system_id], dim=0).cpu()
                system_targets = torch.cat(targets[system_id], dim=0).cpu()

                if mod_unifac_preds and system_id in mod_unifac_preds:
                    system_mod_unifac_predictions = torch.cat(mod_unifac_preds[system_id], dim=0).cpu()
                else:
                    system_mod_unifac_predictions = torch.full_like(system_model_predictions, float('nan'))

                # Determine system classification
                if test_classification is not None:
                    system_classification = classification_descriptions.get(test_classification.get(system_id, -1), "Unknown Classification")
                else:
                    system_classification = "Classification Not Provided"
                system_indices = (test_systems_ID == system_id).nonzero(as_tuple=True)[0]
                smiles_1 = int_to_smiles[component_1_ID[system_indices[0]].item()]
                smiles_2 = int_to_smiles[component_2_ID[system_indices[0]].item()]

                for i in range(0, len(system_model_predictions), 2):
                    row = [system_id, system_classification, smiles_1, smiles_2]
                    row.extend(system_model_predictions[i:i+2].tolist())
                    row.extend(system_mod_unifac_predictions[i:i+2].tolist() if i + 1 < len(system_mod_unifac_predictions) else [None, None])
                    row.extend(system_targets[i:i+2].tolist() if i + 1 < len(system_targets) else [system_targets[i].item(), None])
                    writer.writerow(row)

    @staticmethod                   
    def write_avg_metrics_to_csv(filename, avg_metrics):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            header = ['Metric', 'Value']
            writer.writerow(header)
            
            for metric, value in avg_metrics.items():
                writer.writerow([metric, value])
                
    @staticmethod
    def evaluate(T_Test, x_Test, FP_Test, y_test, test_systems_ID, trained_model, round_id=None, mod_unifac_predictions=None, test_classification=None, results_dir=None, suffix=''):
        if results_dir is None:
            round_id = round_id or "central"
            results_dir = os.path.join("Results", "global", f"round_{round_id}")
        os.makedirs(results_dir, exist_ok=True)

        trained_model.eval()  # Set the model to evaluation mode
        
        if test_classification is not None:
            classification_descriptions = {
                0: "Both components unknown",
                1: "One component unknown",
                2: "System unknown"
                }
            
            # Initialize dictionaries to hold MAE and MSE for each classification
            classification_mae = {desc: [] for desc in classification_descriptions.values()}
            classification_mse = {desc: [] for desc in classification_descriptions.values()}
            # Initialize dictionaries to hold MAE and MSE for each classification in sync with UNIFAC and modified UNIFAC
            classification_mae_sync = {desc: [] for desc in classification_descriptions.values()}
            classification_mse_sync = {desc: [] for desc in classification_descriptions.values()}

        # Initialize dictionaries to hold predictions and targets        
        predictions = {}
        targets = {}
        if mod_unifac_predictions is not None:
            mod_unifac_preds = {}
            system_wise_mae, system_wise_mse = [], []  # Trained model metrics in sync with UNIFAC
            system_wise_mae_unifac, system_wise_mse_unifac = [], []  # UNIFAC metrics
            system_wise_mae_mod_unifac, system_wise_mse_mod_unifac = [], []  # Modified UNIFAC metrics

        outputs, _ = trained_model(T_Test, x_Test, FP_Test)  # shape: [N, 2]
        outputs = outputs.view(-1, 2)
        targets_tensor = y_test.view(-1, 2)

        for i in range(T_Test.shape[0]):
            system_id = test_systems_ID[i].item()

            if system_id not in predictions:
                predictions[system_id] = []
                targets[system_id] = []
                if mod_unifac_predictions is not None:
                    mod_unifac_preds[system_id] = []

            predictions[system_id].append(outputs[i])
            targets[system_id].append(targets_tensor[i])

            if mod_unifac_predictions is not None:
                mod_unifac_preds[system_id].append(mod_unifac_predictions[i])

        # Compute metrics for the trained model on the full dataset
        all_targets = torch.cat([t for t_list in targets.values() for t in t_list])
        all_predictions = torch.cat([p for p_list in predictions.values() for p in p_list])
        # If UNIFAC, also concatenate the predictions
        if mod_unifac_predictions is not None:
            all_mod_unifac_predictions = np.concatenate([np.array(p).reshape(-1) for p in mod_unifac_preds.values()])
        
        # Function to compute MAE and MSE
        def compute_metrics(targets, predictions):
            mae = torch.nn.functional.l1_loss(predictions, targets, reduction='mean').item()
            mse = torch.nn.functional.mse_loss(predictions, targets, reduction='mean').item()
            return mae, mse
        
        overall_mae, overall_mse = compute_metrics(all_targets, all_predictions)
        print(f'Overall MAE (Trained Model): ', overall_mae)
        print(f'Overall MSE (Trained Model): ', overall_mse)

        # Add overall MAE/MSE for different models to the avg_metrics dictionary
        avg_metrics = {}
        avg_metrics['Overall MAE - Trained Model'] = overall_mae
        avg_metrics['Overall MSE - Trained Model'] = overall_mse

        # Lists to hold MAE and MSE for all three scenarios
        system_wise_mae_all, system_wise_mse_all = [], []  # Trained model metrics for all systems

        #Initiliaze dictionary for system mae
        #system_wise_mae_all_dict = {}  # Initialize the dictionary
        # Process each system for the trained model
        for system_id in predictions.keys():
            system_targets = torch.cat(targets[system_id], dim=0)
            system_predictions = torch.cat(predictions[system_id], dim=0)

            mae, mse = compute_metrics(system_targets, system_predictions)
            system_wise_mae_all.append(mae)
            system_wise_mse_all.append(mse)
            #system_wise_mae_all_dict[system_id] = mae

            # Determine the classification of the system and compute metrics
            if test_classification is not None and system_id in test_classification:
                classification_num = test_classification[system_id]
                if classification_num in classification_descriptions:
                    desc = classification_descriptions[classification_num]
                    classification_mae[desc].append(mae)
                    classification_mse[desc].append(mse)

            #if unifac_preds is not None and unifac_preds[system_id] is not None and not np.isnan(unifac_preds[system_id]).any() and mod_unifac_preds[system_id] is not None and not np.isnan(mod_unifac_preds[system_id]).any():
            if mod_unifac_predictions is not None:
                if mod_unifac_preds is not None and mod_unifac_preds[system_id] is not None and not np.isnan(mod_unifac_preds[system_id]).any():
                    #unifac_data = np.array(unifac_preds[system_id]).reshape(-1)
                    mod_unifac_data = np.array(mod_unifac_preds[system_id]).reshape(-1)

                    #mae_unifac, mse_unifac = compute_metrics(system_targets, unifac_data)
                    mae_mod_unifac, mse_mod_unifac = compute_metrics(system_targets, mod_unifac_data)
                    mae_sync, mse_sync = compute_metrics(system_targets, system_predictions)

                    #system_wise_mae_unifac.append(mae_unifac)
                    #system_wise_mse_unifac.append(mse_unifac)
                    system_wise_mae.append(mae_sync)
                    system_wise_mse.append(mse_sync)
                    system_wise_mae_mod_unifac.append(mae_mod_unifac)
                    system_wise_mse_mod_unifac.append(mse_mod_unifac)

                    # Determine the classification of the system and compute metrics for synced model
                    if test_classification is not None and system_id in test_classification:
                        classification_num = test_classification[system_id]
                        if classification_num in classification_descriptions:
                            desc = classification_descriptions[classification_num]
                            classification_mae_sync[desc].append(mae)
                            classification_mse_sync[desc].append(mse)

        # Print MAE/MSE for each classification
        # Storing and printing MAE/MSE for each classification
        if test_classification is not None:
            for desc in classification_descriptions.values():
                if classification_mae[desc]:
                    avg_mae = np.mean(classification_mae[desc])
                    avg_mse = np.mean(classification_mse[desc])
                    print(f'Average System-wise MAE (Trained Model - {desc}):', avg_mae)
                    print(f'Average System-wise MSE (Trained Model - {desc}):', avg_mse)
                    avg_metrics[f'MAE - {desc}'] = avg_mae
                    avg_metrics[f'MSE - {desc}'] = avg_mse
                    #Print the same for synced case
                    avg_mae_sync = np.mean(classification_mae_sync[desc])
                    avg_mse_sync = np.mean(classification_mse_sync[desc])
                    print(f'Average System-wise MAE (Trained Model - {desc} - Synced with UNIFAC):', avg_mae_sync)
                    print(f'Average System-wise MSE (Trained Model - {desc} - Synced with UNIFAC):', avg_mse_sync)
                    avg_metrics[f'MAE - {desc} - Synced with UNIFAC'] = avg_mae_sync
                    avg_metrics[f'MSE - {desc} - Synced with UNIFAC'] = avg_mse_sync

        # Calculate and print average system-wise MAE/MSE for all data
        avg_mae_all = torch.tensor(system_wise_mae_all).mean().item()
        avg_mse_all = torch.tensor(system_wise_mse_all).mean().item()
        print(f'Average System-wise MAE (Trained Model - All): ', avg_mae_all)
        print(f'Average System-wise MSE (Trained Model - All): ', avg_mse_all)
        avg_metrics['MAE - All Data'] = avg_mae_all
        avg_metrics['MSE - All Data'] = avg_mse_all
        valid_mask = np.ones(all_targets.shape, dtype=bool)

        if mod_unifac_predictions is not None:
            if system_wise_mae_mod_unifac:  # Check if there's any UNIFAC data

                avg_metrics['MAE - Modified UNIFAC'] = np.mean(system_wise_mae_mod_unifac)
                avg_metrics['MSE - Modified UNIFAC'] = np.mean(system_wise_mse_mod_unifac)
                avg_metrics['MAE - Trained Model (Synced with UNIFAC)'] = np.mean(system_wise_mae)
                avg_metrics['MSE - Trained Model (Synced with UNIFAC)'] = np.mean(system_wise_mse)
                print(f'Average System-wise MAE (Trained Model - Synced with UNIFAC): ', np.mean(system_wise_mae))
                print(f'Average System-wise MSE (Trained Model - Synced with UNIFAC): ', np.mean(system_wise_mse))

                print(f'Average System-wise MAE (Modified UNIFAC): ', np.mean(system_wise_mae_mod_unifac))
                print(f'Average System-wise MSE (Modified UNIFAC): ', np.mean(system_wise_mse_mod_unifac))
                # Find all entries where UNIFAC and mod. UNIFAC is not NAN
                valid_mask =  ~np.isnan(all_mod_unifac_predictions)
                # Overall MAE and MSE for trained model in sync with UNIFAC
                overall_mae_sync, overall_mse_sync = compute_metrics(all_targets[valid_mask], all_predictions[valid_mask])
                print(f'Overall MAE (Trained Model - Synced with UNIFAC): ', overall_mae_sync)
                print(f'Overall MSE (Trained Model - Synced with UNIFAC): ', overall_mse_sync)
                # Overall MAE and MSE for modified UNIFAC
                overall_mae_mod_unifac, overall_mse_mod_unifac = compute_metrics(all_targets[valid_mask], all_mod_unifac_predictions[valid_mask])
                print(f'Overall MAE (Modified UNIFAC): ', overall_mae_mod_unifac)
                print(f'Overall MSE (Modified UNIFAC): ', overall_mse_mod_unifac)
                avg_metrics['Overall MAE - Modified UNIFAC'] = overall_mae_mod_unifac
                avg_metrics['Overall MSE - Modified UNIFAC'] = overall_mse_mod_unifac

        # For the UNIFAC predictions, just use the flattened arrays directly:
        plt.figure(figsize=(8, 8))
        plt.scatter(all_targets[valid_mask].detach().cpu().numpy(), all_predictions[valid_mask].detach().cpu().numpy(), alpha=0.2, label='ln (gamma) - Model', marker='s', c='blue')
        if mod_unifac_predictions is not None:
            plt.scatter(all_targets[valid_mask], all_mod_unifac_predictions[valid_mask], c='green', alpha=0.2, label='ln (gamma) - mod UNIFAC', marker='o')

        combined = torch.cat([all_targets[valid_mask], all_predictions[valid_mask]])
        min_val, max_val = combined.min().item(), combined.max().item()

        plt.plot([min_val, max_val], [min_val, max_val], 'r')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.grid(True)
        plt.title('Overall parity plot')
        plt.legend()
        plt.savefig(os.path.join(results_dir, f"parity_plot_{suffix}.png"))
        plt.close('all')

        # Create parity plot for data not used in UNIFAC, so ~valid_mask
        if mod_unifac_predictions is not None:
            plt.figure(figsize=(8, 8))
            plt.scatter(all_targets[~valid_mask], all_predictions[~valid_mask], alpha=0.2, label='ln (gamma) - Model', marker='s', c='blue')

            min_val = min(all_targets[~valid_mask].min(), all_predictions[~valid_mask].min())
            max_val = max(all_targets[~valid_mask].max(), all_predictions[~valid_mask].max())

            plt.plot([min_val, max_val], [min_val, max_val], 'r')
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.axis('equal')
            plt.xlim(min_val, max_val)
            plt.ylim(min_val, max_val)
            plt.grid(True)
            plt.title('Overall parity plot (only system that UNIFAC can not predict)')
            plt.legend()
            plt.savefig(os.path.join(results_dir, "parity_plot_no_unifac.png"))
            plt.close('all')
            
        # Residuals Histogram
        residuals_model_flat = (all_targets[valid_mask] - all_predictions[valid_mask]).detach().cpu().numpy().flatten()

        plt.figure(figsize=(8, 8))
        plt.hist(residuals_model_flat, bins=25, alpha=0.5,range=(-0.5, 0.5), label="Model Residuals", color="black")

        # If UNIFAC predictions are provided, calculate residuals and add to histogram
        if mod_unifac_predictions is not None:
            residuals_modified_unifac = all_targets[valid_mask].ravel() - all_mod_unifac_predictions[valid_mask].ravel()
            plt.hist(residuals_modified_unifac, bins=25, alpha=0.5,range=(-0.5, 0.5), label="Modified UNIFAC Residuals", color="yellow")

        plt.xlabel('Residuals')
        plt.xlim(-0.5, 0.5)
        plt.ylabel('Number of Data Points')
        plt.legend()
        plt.grid(True)
        plt.title('Residuals Distribution')
        plt.savefig(os.path.join(results_dir, f"residuals_histogram_{suffix}.png"))
        plt.close('all')

        # System-wise MAE Histogram
        plt.figure(figsize=(8, 8))
        plt.hist(system_wise_mae_all, bins=25, range=(0, 0.5), alpha=0.5, label="Model MAE", color="black")

        if mod_unifac_predictions is not None:  # If there's UNIFAC data
            #plt.hist(system_wise_mae_unifac, bins=25,range=(0, 0.5), alpha=0.5, label="UNIFAC MAE", color="red")
            plt.hist(system_wise_mae_mod_unifac, bins=25,range=(0, 0.5), alpha=0.5, label="Modified UNIFAC MAE", color="yellow")

        plt.xlabel('System-wise MAE')
        plt.ylabel('Number of Systems')
        plt.xlim(0, 0.5)
        plt.legend()
        plt.grid(True)
        plt.title('System-wise MAE Distribution')
        plt.savefig(os.path.join(results_dir, f"system_mae_histogram_{suffix}.png"))
        plt.close('all')

        # System-wise MSE Histogram
        plt.figure(figsize=(8, 8))          
        plt.hist(system_wise_mse_all, bins=30, range=(0, 0.2), alpha=0.5, label="Model MSE", color="black")

        if mod_unifac_predictions is not None:  # If there's UNIFAC data, use the same bins
            plt.hist(system_wise_mse_mod_unifac, bins=30, range=(0, 0.2), alpha=0.5, label="Modified UNIFAC MSE", color="yellow")

        plt.xlabel('System-wise MSE')
        plt.ylabel('Number of Systems')
        # Automatically adjust xlim based on the actual range of data
        plt.xlim(0, 0.2)
        plt.ylim(auto=True)
        plt.legend()
        plt.grid(True)
        plt.title('System-wise MSE Distribution')
        plt.savefig(os.path.join(results_dir, f"system_mse_histogram_{suffix}.png"))
        plt.close('all')

        #filename = os.path.join(results_dir, "system_predictions.csv")
        #Evaluation.write_predictions_to_csv(filename, predictions, targets, test_systems_ID, None, mod_unifac_preds, int_to_smiles, component_1_ID, component_2_ID, classification_descriptions=classification_descriptions)
        if suffix:
            filename = os.path.join(results_dir, f"avg_metrics_{suffix}.csv")
        else:
            filename = os.path.join(results_dir, "avg_metrics.csv")
        Evaluation.write_avg_metrics_to_csv(filename, avg_metrics)
        #Evaluation.find_highest_mae(system_wise_mae_all_dict, test_systems_ID, int_to_smiles, component_1_ID, component_2_ID)

        print(f"Evaluationsergebnisse gespeichert in: {results_dir}\n")

        return avg_mae_all, avg_mse_all, system_wise_mae_all, system_wise_mse_all