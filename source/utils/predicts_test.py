import torch
import numpy as np
import matplotlib.pyplot as plt


def get_predictions_on_test(model, test_loader):

    predictions, label_list, data_list = [], [], []

    for name, loader in [('val', test_loader)]:

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            for data, labels in loader:
                inputs = data['encoder_cont'].permute(0, 2, 1) # B x C x T
                inputs = inputs.to(device,  dtype=torch.float)
                labels = labels[0].to(device,  dtype=torch.float)
                labels = labels.squeeze(2)
                
                y = model(inputs)
                predictions.append(y.detach().cpu().numpy())
                label_list.append(labels.detach().cpu().numpy())
                data_list.append(inputs.detach().cpu().numpy())

        all_preds = np.vstack(predictions)
        all_labels = np.vstack(label_list)
        all_data = np.vstack(data_list)

    return all_preds, all_labels, all_data

def plot_predictions_1(model, test_loader, plot_idx, target_name=None):

    # get predictions first.    
    all_preds, all_labels, all_data = get_predictions_on_test(model, test_loader)
    
    xticks_history = all_data.shape[2]
    xticks_pred = [xticks_history - 1 + i for i in range(all_labels.shape[1])]

    fig = plt.figure(figsize=(10, 3))

    ax = fig.subplots()
    ax.plot(all_data[plot_idx][0], linewidth=1)
    print(all_data[plot_idx].shape)

    ax.plot(xticks_pred, all_labels[plot_idx][0], linewidth=1, alpha=0.7, label="real")
    ax.plot(xticks_pred, all_preds[plot_idx], linewidth=1, label="pred")

    ax.legend(fontsize=7)
    ax.set_title(f"{target_name}", fontsize=9)

    plt.tight_layout()
    plt.show()

    return