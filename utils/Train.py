import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import csv  

class HANNA(nn.Module):
    def __init__(self, Embedding_ChemBERT=384, nodes=96):
        super(HANNA, self).__init__()

        self.Embedding_ChemBERT = Embedding_ChemBERT # Pre-trained embeddings (E_i) from ChemBERTa-2
        self.nodes = nodes # Number of Nodes in HANNA

        # Component Embedding Network f_theta Input: E_i Output: f_theta(E_i)
        self.theta = nn.Sequential(
            nn.Linear(Embedding_ChemBERT, nodes),
            nn.SiLU(),
        )

        # Mixture Embedding Network f_alpha Input: C_i, Output: f_alpha(C_i)
        # nodes+2 is needed for concatenating T and x_i to the embedding f_theta(E_i)
        self.alpha = nn.Sequential(
            nn.Linear(nodes+2, nodes),
            nn.SiLU(),
            nn.Linear(nodes, nodes),
            nn.SiLU(),
        )

        # Property Network Input f_phi Input: C_mix, Output: g^E_NN 
        self.phi = nn.Sequential(
            nn.Linear(nodes, nodes),
            nn.SiLU(),
            nn.Linear(nodes, 1)
        )

    def forward(self, temperature, mole_fractions, E_i):
        # Determine batch_size (B) and number of components (N)
        batch_size, num_components, _ = E_i.shape # [B,N,E] E=384, ChemBERTa-2 embedding

        # Enable gradient tracking to use autograd
        E_i.requires_grad_(True)
        temperature.requires_grad_(True) # Standardized temperature
        mole_fractions.requires_grad_(True) # x_1

        # Calculate remaining mole fraction for the Nth component (here: N=2)
        mole_fraction_N = 1 - mole_fractions.sum(dim=1, keepdim=True) # x_2=1-x_1 [B,1]
        mole_fractions_complete = torch.cat([mole_fractions, mole_fraction_N], dim=1) # [x_1,1-x_1], [B,2]

        # Reshape mole fraction and temperature
        mole_fractions_complete_reshaped = mole_fractions_complete.unsqueeze(-1) # [B,N,1]
        T_reshaped = temperature.view(batch_size, 1, 1).expand(-1, num_components, 1) # [B,N,1]

        # Fine-tuning of the component embeddings
        theta_E_i = self.theta(E_i) # [B,N,nodes]

        # Calculate cosine similarity between the two components
        cosine_sim = F.cosine_similarity(theta_E_i[:, 0, :], theta_E_i[:, 1, :], dim=1) #[B]
        # Calculate cosine distance between the two components
        cosine_distance = 1 - cosine_sim # [B]

        # Concatenate embedding with T and x_i
        c_i = torch.cat([T_reshaped, mole_fractions_complete_reshaped, theta_E_i], dim=-1) #[B,N,nodes+2]
        alpha_c_i = self.alpha(c_i) # [B,N,nodes]
        c_mix = alpha_c_i.sum(dim=1) # [B,nodes]
        gE_NN = self.phi(c_mix).squeeze(-1) # [B]

        # Apply cosine similarity adjustment
        correction_factor_mole_fraction = torch.prod(mole_fractions_complete, dim=1) # [B] x1*(1-x1) term
        gE = gE_NN * correction_factor_mole_fraction * cosine_distance  # [B] Adjust gE_NN with the physical constraints and calculate gE/RT

        # Compute (dgE/dx1)/RT
        dgE_dx1 = torch.autograd.grad(gE.sum(), mole_fractions, create_graph=True)[0] # [B,1]

        # ln gamma_i equation (binary mixture). Unsqueeze to adjust dimension to [B,1] for gE/RT
        ln_gamma_1 = gE.unsqueeze(1) + (1 - mole_fractions) * dgE_dx1 # [B,1]
        ln_gamma_2 = gE.unsqueeze(1) - mole_fractions * dgE_dx1 # [B,1]
        # Concatenate the ln_gammas
        ln_gammas = torch.cat([ln_gamma_1, ln_gamma_2], dim=1) # [B,2]

        return ln_gammas, gE

#---------------------------------------------------------------------------------------------------------------------------------------------------------

def train_model(model, T_train, x_train, FP_train, ln_gamma_true_train, T_val, x_val, FP_val, ln_gamma_true_val, loss_fn, optimizer, scheduler, device, train_systems_ID_tensor, val_systems_ID_tensor, n_epochs=200, batch_size=64, patience=50, suffix=''):
    # Training and validation data
    train_data = TensorDataset(T_train, x_train, FP_train, ln_gamma_true_train,train_systems_ID_tensor)
    val_data = TensorDataset(T_val, x_val, FP_val, ln_gamma_true_val,val_systems_ID_tensor)
    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    # Other training parameters
    train_losses = []
    val_losses = []
    min_val_loss = np.inf
    early_stop_counter = 0

    # Training loop
    for epoch in range(n_epochs):
        batch_train_losses = []
        # Set the model to training mode
        model.train()
        for T, x, FP, ln_gamma_true, batch_systems in train_loader:
            T, x, FP, ln_gamma_true = T.to(device), x.to(device), FP.to(device), ln_gamma_true.to(device)
            # Forward pass
            ln_gammas_pred,_= model(T, x, FP)
            # Compute Loss
            loss = loss_fn(ln_gammas_pred, ln_gamma_true, batch_systems)
            # Set gradients to zero, perform a backward pass, and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())

        train_loss = torch.tensor(batch_train_losses).mean().item()
        train_losses.append(train_loss)

        # Validation
        model.eval()
        batch_val_losses = []
        for T, x, FP, ln_gamma_true, batch_systems in val_loader:
            T, x, FP, ln_gamma_true = T.to(device), x.to(device), FP.to(device), ln_gamma_true.to(device)
            # Forward pass
            ln_gammas_pred,_= model(T, x, FP)
            # Compute Loss
            loss = loss_fn(ln_gammas_pred, ln_gamma_true, batch_systems)
            batch_val_losses.append(loss.item())

        val_loss = torch.tensor(batch_val_losses).mean().item()
        val_losses.append(val_loss)
        # Set the learning rate based on the validation loss with the scheduler
        scheduler.step(val_loss)

        # Print epoch info
        print('Epoch [{}/{}], Train Loss: {:.4f}, Validation Loss: {:.4f}'
              .format(epoch+1, n_epochs, train_loss, val_loss))

        # Early stopping and model saving
        if val_loss < min_val_loss:
            best_model_state = model.state_dict()
            min_val_loss = val_loss
            early_stop_counter = 0
            print(f"New best model found in epoch {epoch+1}!")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"\nTraining stopped. No improvement for {patience} epochs.")
                break

    model_save_path = os.path.join("models", f"best_model_{suffix}.pth")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Save the best model at the end of training
    if best_model_state:
        torch.save(best_model_state, model_save_path)
        print(f"Best model saved after training: {model_save_path}")

    # Plot the losses over the epochs
    result_dir = os.path.join("Results", "losses")
    os.makedirs(result_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    epoch_numbers = range(1, len(train_losses) + 1)
    plt.plot(epoch_numbers, train_losses, label='Training Total Loss')
    plt.plot(epoch_numbers, val_losses, label='Validation Total Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(result_dir, f'losses_{suffix}.png'))

    # Save the losses to a csv file
    csv_path = os.path.join(result_dir, f"losses_{suffix}.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Training Total Loss', 'Validation Total Loss'])
        for epoch in range(len(train_losses)):
            writer.writerow([epoch + 1, train_losses[epoch], val_losses[epoch]])

#---------------------------------------------------------------------------------------------------------------------------------------------------------

class Smooth_L1_Loss(nn.Module):
    def __init__(self, delta=0.5, use_simple_loss=False):
        super().__init__()
        self.Smooth_L1_Loss = nn.SmoothL1Loss(reduction='none', beta=delta)
        self.use_simple_loss = use_simple_loss
        if use_simple_loss:
            # Use default reduction (mean) for Smooth_L1_Loss
            self.Smooth_L1_Loss = nn.SmoothL1Loss(beta=delta)
    
    def forward(self, ln_gamma_pred, ln_gamma_true, systems_ID):

        if self.use_simple_loss:
            # Compute the simple Smooth_L1_Loss
            # Average losses for gamma_1 and gamma_2 for each sample
            return self.Smooth_L1_Loss(ln_gamma_pred, ln_gamma_true)
        else:
            # Calculate Smooth_L1_Loss  for all samples and for both gamma_1 and gamma_2
            ln_gamma_losses = self.Smooth_L1_Loss(ln_gamma_pred, ln_gamma_true)

            # Average losses for gamma_1 and gamma_2 for each sample
            ln_gamma_losses_avg = ln_gamma_losses.mean(dim=1)

            # Initialize a tensor to store losses per system
            system_loss = torch.zeros(len(torch.unique(systems_ID)), device=ln_gamma_pred.device)

            # Loop over unique systems to compute the mean loss for each system
            for idx, system in enumerate(torch.unique(systems_ID)):
                system_indices = torch.where(systems_ID == system)[0]
                
                # Calculate the average loss for this system
                loss = ln_gamma_losses_avg[system_indices].mean()
                system_loss[idx] = loss
            
            return system_loss.mean()  # return the average loss across systems