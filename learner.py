import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm_notebook


class FewShotLearner():
    """
    Provides handy functions for Few-shot models.
    
    Attributes:
        data (object): Object containing Datasets & DataLoaders for training, validation and test.
        model (torch.nn.Module): Few-shot Neural Network.
        device (torch.device): Device where the model is executed (GPU or CPU).
        loss_fn (torch.nn.Module, optional): Loss function to evaluate the model.
    """
    
    def __init__(self, model, data, loss_fn=F.nll_loss, use_cuda=True):
        """
        Args:
            model (torch.nn.Module): Neural network.
            data: Object containing Datasets & Dataloaders for training, validation and test.
            loss_fn (torch.nn.Module, optional): Loss function to evaluate the model.
            use_cuda (bool, optional): Specify whether GPU or CPU is used.
        """
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')            
        self.model = model.to(self.device)
        self.data = data
        self.loss_fn = loss_fn
            
    def fit(self, epochs, opt=None, lr_sched=None):
        """
        Trains the model using train and validation sets.
        
        Args:
            epochs (int): Number of epochs for training.
            opt (torch.optim.Optimizer, optional): Model's optimizer. Default: Adam.
            lr_sched (torch.optim._LRScheduler, optional): Model's learning rate scheaduler. Default: StepLR.
        """
        episodes_n = len(self.data.trn_dl)
        trn_loss = 0; trn_acc = 0; val_loss = 0; val_acc = 0; best_val_acc = 0
        
        opt = opt or optim.Adam(self.model.parameters(), 1e-3)
        lr_sched = lr_sched or optim.lr_scheduler.StepLR(opt, 20, 0.5)
        
        for i in tqdm_notebook(range(epochs)):
            self.model.train()
            for episode in iter(self.data.trn_dl):
                support, query = episode
                support_X, support_y = support; query_X, query_y = query
                support_X = support_X.to(self.device); query_X = query_X.to(self.device)
                
                log_probs = self.model(support_X, support_y, query_X).cpu()
                loss = self.loss_fn(log_probs, query_y)
                preds = torch.argmax(log_probs, dim=1)
                acc = (preds == query_y).float().mean()

                opt.zero_grad()
                loss.backward()
                opt.step()               

                trn_loss += loss.item(); trn_acc += acc.item()
               
            lr_sched.step()
            
            val_loss, val_acc = self.evaluate(1, test=False)
            trn_loss /= episodes_n; trn_acc /= episodes_n
            
            if val_acc > best_val_acc:
                best_lbl = "(Best)" 
                best_state_dict = self.model.state_dict()
                best_val_acc = val_acc
            else:
                best_lbl = ""
            if i == 0: print(f'{"Epoch":>5} {"trn_loss":>8} {"trn_acc":>8} {"val_loss":>8} {"val_acc":>8}')
            print(f'{i+1:4}: {trn_loss:8.6f} {trn_acc:8.6f} {val_loss:8.6f} {val_acc:8.6f} {best_lbl}')
            trn_loss = 0; trn_acc = 0
            
        self.model.load_state_dict(best_state_dict)
        
    def evaluate(self, epochs=10, test=True):
        """
        Computes model's loss and accuracy (using either validation or test set).
        
        Args:
            epochs (int, optional): Number of epochs for evaluation.
            test (bool, optional): If ´True´ evaluate on test set; otherwise on validation set.
        
        Returns:
            float: Averange loss.
            float: Average accuracy.
        """        
        dl = self.data.test_dl if test else self.data.val_dl
        episodes_n = len(dl)

        loss = 0; acc = 0
        self.model.eval()
        for i in range(epochs):
            for episode in iter(dl):
                support, query = episode
                support_X, support_y = support; query_X, query_y = query
                support_X = support_X.to(self.device); query_X = query_X.to(self.device)
                
                log_probs = self.model(support_X, support_y, query_X).cpu()
                _loss = self.loss_fn(log_probs, query_y)
                preds = torch.argmax(log_probs, dim=1)
                _acc = (preds == query_y).float().mean()
              
                loss += _loss.item()
                acc += _acc.item()
                
        loss /= (epochs * episodes_n)
        acc /= (epochs * episodes_n)
        return loss, acc