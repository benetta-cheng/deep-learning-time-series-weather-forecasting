import matplotlib.pyplot as plt
import numpy as np
import torch

def model_evaluation_and_visualizations(model, dataloader, device, stats=None, indices=[2486, 2986]):
    feature_names = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Custom colors per feature

    plt.style.use('ggplot')  # Consistent, readable plotting style
    model.eval()

    num_features = len(feature_names)
    num_samples = len(indices)

    fig, axs = plt.subplots(num_samples, num_features, figsize=(5 * num_features, 4 * num_samples))

    for row, seed in enumerate(indices):
        g = torch.Generator()
        g.manual_seed(seed)

        # Rebuild a deterministic DataLoader with shuffling
        data_iter = iter(torch.utils.data.DataLoader(
            dataloader.dataset, batch_size=32, shuffle=True, generator=g
        ))

        inputs, outputs = next(data_iter)
        inputs, outputs = inputs.to(device), outputs.to(device)

        # Permute for (seq_len, batch, feature)
        inputs_seq = inputs.permute(1, 0, 2)      # (T_in, B, F)
        outputs_seq = outputs.permute(1, 0, 2)    # (T_out, B, F)
        preds = model(inputs_seq, outputs_seq)    # (T_out, B, F)

        b = 0  # Visualize the first sample in the batch

        # Get the data for sample b
        x = inputs_seq[:, b, :].detach().cpu().numpy()      # (T_in, F)
        y_true = outputs_seq[:, b, :].detach().cpu().numpy()  # (T_out, F)
        y_pred = preds[:, b, :].detach().cpu().numpy()        # (T_out, F)

        # Denormalize if stats are provided
        if stats is not None:
            for i, name in enumerate(feature_names):
                mean = stats.loc[name, 'mean']
                std = stats.loc[name, 'std']
                x[:, i] = x[:, i] * std + mean
                y_true[:, i] = y_true[:, i] * std + mean
                y_pred[:, i] = y_pred[:, i] * std + mean

        # Time steps
        t_input = list(range(x.shape[0]))
        t_output = [x.shape[0] + i + 1 for i in range(y_true.shape[0])]
        mse = np.mean((y_true - y_pred) ** 2)

        # Plot each feature separately
        for col in range(num_features):
            ax = axs[row, col] if num_samples > 1 else axs[col]
            feature = feature_names[col]
            color = colors[col % len(colors)]

            # Input sequence
            ax.plot(t_input, x[:, col], linestyle='--', linewidth=2, label="Input", color=color)
            # Ground truth output
            ax.scatter(t_output, y_true[:, col], label="True", color=color, marker='o', s=50, edgecolors='black', linewidth=0.5)
            # Predicted output
            ax.scatter(t_output, y_pred[:, col], label="Pred", color='black', marker='x', s=60)

            ax.set_title(f"{feature} | seed={seed} | MSE={mse:.4f}", fontsize=11)
            ax.grid(True)

            if row == num_samples - 1:
                ax.set_xlabel("Timestep")
            if col == 0:
                ax.set_ylabel("Value")
            if row == 0:
                ax.legend(loc='upper center', fontsize=8)

    plt.tight_layout()
    plt.show()
