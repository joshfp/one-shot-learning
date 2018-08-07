import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import matplotlib.cm as cm


def plot_tensor(t, titles=None):
    """
    Plot N images vectorized in a tensor.
    
    Args:
        t (LongTensor): Tensor of shape: N x 1 x img_sz x img_sz.
        titles (list, optional): Title for each image (lenght: N).
    """
    n = t.size(0)
    rows = n // 5 + 1
    fig = plt.figure(figsize=(16, rows * 3))

    for i in range(n):
        fig.add_subplot(n // 5 + 1, 5 , i + 1)
        plt.imshow(t[i].squeeze(), cmap='binary_r')
        if titles is not None: plt.title(titles[i])
    plt.show()

    
def plot_images(images, titles=None):
    """
    Plot N images.
    
    Args:
        images (List of Image): Images to plot (lenght: N).
        titles (list, optional): Title for each image (lenght: N).
    """
    t = []
    tfms = transforms.ToTensor()
    
    for img in images:
        t.append(tfms(Image.open(img)))
    t = torch.stack(t)
    plot_tensor(t, titles)

    
def plot_episode(support_samples, query_samples, samples_n=3):
    """
    Plot images of a Prototypical Networks's episode.
    
    Args:
        support_samples (tuple of (FloatTensor, LongTensor)): Support samples.
        query_samples (tuple of (FloatTensor, LongTensor)): Query samples.
        samples_n (int, optional): Number of samples to plot.
    """
    s_X, s_y = support_samples; q_X, q_y = query_samples
    classes = torch.unique(s_y)[:samples_n]
    
    for c in range(len(classes)):
        print('=== Class:', c, '===\n')

        s_idxs = (s_y == c).nonzero()
        print('Support samples')
        plot_tensor(s_X[s_idxs])
        
        q_idxs = (q_y == c).nonzero()
        print('Query samples')
        plot_tensor(q_X[q_idxs])

        
def plot_predictions(support_samples, query_samples, probs, correct, descending=True, samples_n=3):
    """
    Plot the samples, targets and predicted images.

    Args:
        support_samples (tuple of (FloatTensor, LongTensor)): Support samples.
        query_samples (tuple of (FloatTensor, LongTensor)): Query samples.
        probs (FloatTensor): Predicted probabilities for query samples.
        correct (bool): If ´´True´´ plot correct classified samples; otherwise plot misclassified.
        descending (bool, optional): If ´´True´´ sort samples by descending probs; otherwise sort ascending.
        samples (int, optional): Number of samples to plot.
    """
    s_X, s_y = support_samples; q_X, q_y = query_samples
    
    probs, preds = torch.max(probs, dim=1)
    
    # filter correct or incorrect
    filter_match = (preds == q_y) == correct
    idxs = filter_match.nonzero().view(-1)
    
    # sort by prob
    _, sorted_idxs = probs[idxs].sort(descending=descending)
    idxs = idxs[sorted_idxs]
     
    for i in idxs[:samples_n]:
        sample_img = q_X[i]
        
        target_idx = (s_y == q_y[i]).nonzero().view(-1)[0]
        target_img = s_X[target_idx]
        
        if correct:
            t = torch.stack([sample_img, target_img])
            title = ['Sample', f'Target/Predicted (prob: {probs[i]:.2f})']
        else:    
            pred_idx = (s_y == preds[i]).nonzero().view(-1)[0]
            pred_img = s_X[pred_idx]
            t = torch.stack([sample_img, target_img, pred_img])
            title = ['Sample', 'Target', f'Predicted (prob: {probs[i]:.2f})']
            
        plot_tensor(t, title)


def plot_tsne(X, tsne, samples_n=100):
    """
    Plot t-SNE with images.
    
    Args:
        X (FloatTensor): Features (images).
        tsne (np.array): 2-d t-SNE of features.
        samples_n (int, optional): Number of samples to plot.
    """
    plt.figure(figsize=(8, 8))
    plt.xlim(-20, 20); plt.ylim(-20, 20)
    cmap = cm.binary_r; cmap.set_over('k', alpha=0)

    sz = 2.5
    idxs = torch.randperm(len(tsne))[:samples_n]

    for img, (x, y) in zip(X[idxs], tsne[idxs]):
        location = (x-sz/2, x+sz/2, y-sz/2, y+sz/2)
        plt.imshow(img.squeeze(), extent=location, cmap=cmap, clim=[0, 0.1], interpolation='nearest')        