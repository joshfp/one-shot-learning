import torch
from torch.utils.data import DataLoader
from omniglot import OmniglotDataset
from functools import partial
from sklearn.preprocessing import LabelEncoder


class FewShotData():
    """Provides a data model for Few-shot Learning.
    
    Attributes:
        trn_ds (Dataset): Training dataset.
        val_ds (Dataset): Validation dataset.
        test_ds (Dataset): Test dataset.
        trn_dl (DataLoader): Training DataLoader.
        val_dl (DataLoader): Validation DataLoader.
        test_dl (DataLoader): Test DataLoader.
    """
    
    def __init__(self,
                 trn_ds, val_ds, test_ds,
                 trn_classes_n=60, trn_support_n=1, trn_query_n=5, 
                 eval_classes_n=20, eval_support_n=1, eval_query_n=15,
                 episodes_n=100, img_sz=28, labels_fld='labels'):
        """
        Args:
            trn_ds (Dataset): Training dataset.
            val_ds (Dataset): Validation dataset.
            test_ds (Dataset): Test dataset.         
            trn_classes_n (int, optional): Number of sampled classes per episode (on training).
            trn_support_n (int, optional): Number of support samples per class per episode (on training).
            trn_query_n (int, optional): Number of query samples per class per episode (on training).
            eval_classes_n (int, optional): Number of sampled classes per episode (on evaluation).
            eval_support_n (int, optional): Number of support samples per class per episode (on evaluation).
            eval_query_n (int, optional): Number of query samples per class per episode (on evaluation).
            episodes_n (int, optional): Number of episodes per epoch.
            img_sz (int, optional): Image size for resizing.
            labels_fld (str, optional): Name of the labels fields in the datasets class. 
        """
        self.trn_ds = trn_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        
        trn_labels = getattr(self.trn_ds, labels_fld)
        val_labels = getattr(self.val_ds, labels_fld)
        test_labels = getattr(self.test_ds, labels_fld)

        trn_bs = FewShotSampler(trn_labels, trn_classes_n, episodes_n, trn_support_n + trn_query_n)
        val_bs = FewShotSampler(val_labels, eval_classes_n, episodes_n, eval_support_n + eval_query_n)
        test_bs = FewShotSampler(test_labels, eval_classes_n, episodes_n, eval_support_n + eval_query_n)
        
        trn_collate = partial(few_shot_collate, classes_n=trn_classes_n, support_n=trn_support_n, query_n=trn_query_n)
        eval_collate = partial(few_shot_collate, classes_n=eval_classes_n, support_n=eval_support_n, query_n=eval_query_n)
        
        self.trn_dl = DataLoader(self.trn_ds, batch_sampler=trn_bs, collate_fn=trn_collate)
        self.val_dl = DataLoader(self.val_ds, batch_sampler=val_bs, collate_fn=eval_collate)
        self.test_dl = DataLoader(self.test_ds, batch_sampler=test_bs, collate_fn=eval_collate)
    
    
class FewShotSampler(object):
    """Yields samples indices for each episode."""
    
    def __init__(self, labels, classes_n, episodes_n, samples_n):
        """
        Args:
            labels (LongTensor): Labels in the dataset.
            classes_n (int): Number of sampled classes per episode.
            episodes_n (int): Number of episodes.
            samples_n (int): Number of samples (support + query) per class per episode.
        """
        super().__init__()
        self.labels = labels
        self.classes = torch.unique(labels) #: tensor: Classes in the dataset.
        self.episodes_n = episodes_n
        self.classes_n = classes_n
        self.samples_n = samples_n
        
        #: list of list: List of classes, each of them containing the list of indexes of that class.
        self.idxs_by_class = [[] for _ in range(len(self.classes))]
        for i, c in enumerate(labels):
            self.idxs_by_class[c].append(i)
                
    def __iter__(self):
        """
        Yields samples indexes for an episode. For each episode, 'samples_n' random samples are generated for 
        each of the 'classes_n' classes. samples_n = number of support samples + number of query samples.
        """
        for i in range(self.episodes_n): 
            classes_idxs = torch.randperm(len(self.classes))[:self.classes_n]
            episode = torch.LongTensor(self.samples_n * self.classes_n)
            
            for i, c in enumerate(classes_idxs): 
                class_idxs = torch.LongTensor(self.idxs_by_class[c])
                sample_idxs = torch.randperm(len(class_idxs))[:self.samples_n]
                class_samples = class_idxs[sample_idxs]
                slc = slice(i * self.samples_n , (i + 1) * self.samples_n)
                episode[slc] = class_samples
            yield episode                      
      
    def __len__(self):
        """Return number of episodes per epoch."""
        return self.episodes_n

    
def few_shot_collate(batch, classes_n, support_n, query_n):
    """
    Split a batch (episode) in support and query sets.
    Additionally, encodes labels for that particular episode in the range [0, classes_n].
    
    Args:
        batch (list of tuple (FloatTensor, FloatLong)): Tuples with features and labels.
        classes_n (int): Number of classes per episode.
        support_n (int): Number of support samples per class per episode.
        query_n (int): Number of query samples per class per episode.
        
    Return:
        tuple (FloatTensor, LongTensor): Support samples (features tensor, labels tensor).
        tuple (FloatTensor, LongTensor): Query samples (features tensor, labels tens)
    """ 
    X, y = tuple(zip(*batch))
    X = torch.stack(X)
    y = LabelEncoder().fit_transform(y)
    y = torch.from_numpy(y)    
    
    s_idxs = torch.LongTensor(support_n * classes_n)
    q_idxs = torch.LongTensor(query_n * classes_n)
    
    for c in range(classes_n):
        class_idxs = (y == c).nonzero().view(-1)
        s = slice(c * support_n, (c + 1) * support_n)
        q = slice(c * query_n, (c + 1) * query_n)
        s_idxs[s] = class_idxs[:support_n]
        q_idxs[q] = class_idxs[support_n : support_n + query_n]
        
    support_samples = (X[s_idxs], y[s_idxs])
    query_samples = (X[q_idxs], y[q_idxs])
    
    return support_samples, query_samples