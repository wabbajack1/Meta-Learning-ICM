import matplotlib.pyplot as plt
import pandas as pd
import os

def plot(log, setting_name, model_name):
    # Generally you should pull your plotting code out of your training
    # script but we are doing it here for brevity.
    df = pd.DataFrame(log)

    fig, ax = plt.subplots(figsize=(6, 4))
    train_df = df[df['mode'] == 'train']
    test_df = df[df['mode'] == 'test']
    ax.plot(train_df['epoch'], train_df['acc'], label='Train')
    ax.plot(test_df['epoch'], test_df['acc'], label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(70, 100)
    fig.legend(ncol=2, loc='lower right')
    fig.tight_layout()

    os.makedirs('plots', exist_ok=True)
    fname = f'plots/{model_name}-{setting_name}-accs.png'
    print(f'--- Plotting accuracy to {fname}')
    fig.savefig(fname)
    plt.close(fig)

def compute_total_loss(task_loss, s_t, s_t1, a_t, icm_model):
    beta = 0.7
    icm_weight = 0.5

    # icm module pass
    forward_error, backward_error = icm_model(s_t, a_t, s_t1)

    # Total loss
    total_loss = task_loss + icm_weight * ( (beta * forward_error.mean() + (1-beta) * backward_error.mean()) - forward_error.detach().mean() )
    return total_loss