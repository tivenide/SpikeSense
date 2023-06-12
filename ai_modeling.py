from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class WindowedFrameDataset(Dataset):
    """
    A PyTorch dataset that represents windowed frames of data and their labels.

    Parameters
    ----------
    data : numpy.ndarray or list
        The input data array of shape (num_frames, num_channels, height, width).
    labels : numpy.ndarray or list
        The target labels array of shape (num_frames,).

    Attributes
    ----------
    data : numpy.ndarray or list
        The input data array.
    transform : torchvision.transforms
        The transform applied to the input data.
    labels : numpy.ndarray or list
        The target labels array.

    Methods
    -------
    __len__()
        Returns the length of the dataset.
    __getitem__(idx)
        Returns the data and label of the specified index.

    Notes
    -----
    This class takes a numpy array or a list of windowed frames of data and their labels, and transforms them into a PyTorch dataset.
    It also applies a transform to the input data to convert it to a tensor.

    Example
    -------
    # create a dataset
    time_rows = np.random.randn(100, 3, 32, 32)
    labels = np.random.randint(0, 2, size=(100,))
    windowed_frame_dataset = WindowedFrameDataset(time_rows, labels)

    # create a dataloader
    batch_size = 32
    windowed_frame_dataloader = DataLoader(windowed_frame_dataset, batch_size=batch_size, shuffle=True)
    """

    def __init__(self, data, labels):
        self.data = data.astype('float32')
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.labels = labels.astype('float32')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]
        return data, labels

def create_dataloader_simple(data, labels, batch_size=32):
    from torch.utils.data import DataLoader
    time_rows = data
    windowed_frame_dataset = WindowedFrameDataset(time_rows, labels)
    windowed_frame_dataloader = DataLoader(windowed_frame_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return windowed_frame_dataloader

import numpy as np

class EarlyStopper:
    """
    Early stopper that stops training if the validation loss does not improve
    for a given number of epochs.
    Parameters
    ----------
        patience (int):
            The number of epochs to wait if the validation loss does not improve before stopping early.
        min_delta (float):
            The minimum change in validation loss to be considered as an improvement.
    Attributes
    ----------
        patience (int):
            The number of epochs to wait if the validation loss does not improve before stopping early.
        min_delta (float):
            The minimum change in validation loss to be considered as an improvement.
        counter (int):
            The number of epochs that the validation loss has not improved.
        min_loss (float):
            The best validation loss so far.
    Examples
    --------
    early_stop = EarlyStopper(patience=15, min_delta=0.01)
    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, early_stop=early_stop)
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = np.inf

    def early_stop(self, validation_loss, verbose=False):
        if validation_loss < self.min_loss - self.min_delta:
            self.counter = 0
            self.min_loss = validation_loss
            if verbose:
                print(f"Early Stop: Counter reset. Min loss is {self.min_loss}.")
        else:
            self.counter += 1
            if verbose:
                print(f"Early Stop: Counter increased to {self.counter}. Min loss is {self.min_loss}. Patience is set to {self.patience}.")
            if self.counter >= self.patience:
                if verbose:
                    print("Early Stop: Patience reached. Stopping early.")
                return True
        return False

import torch

class modeling():
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader,
                 loss_fn, early_stop, optimizer, scheduler, n_classes,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        self.model = model.to(device)

        self.device = device

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.eval_dataloader = eval_dataloader

        self.loss_fn = loss_fn
        self.early_stop = early_stop
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.epoch = 0
        self.epochs = 25

        self.train_loss = 0
        self.eval_loss = 0

        self.n_classes = n_classes

        self.train_some_metrics = []
        self.eval_some_metrics = []
        self.test_some_metrics = None

    def run(self):
        for self.epoch in range(self.epochs):
            print(f"Epoch {self.epoch + 1} of {self.model.__class__.__name__}\n-------------------------------")
            self.train()
            self.eval()
            if self.early_stop.early_stop(validation_loss=self.eval_loss):
                break
        self.test()
        self.plot_metrics_over_epochs(self.train_some_metrics, title='Train metrics over epochs')
        self.plot_metrics_over_epochs(self.eval_some_metrics, title='Eval metrics over epochs')
        """
        for metrics in self.train_some_metrics:
            confusion_matrix = metrics['confusion_matrix']
            roc_curve = metrics['roc_curve']
            pr_curve = metrics['pr_curve']
            #self.plot_roc_curve(roc_curve['fpr'], roc_curve['tpr'], roc_curve['auc'])
            #self.plot_precision_recall_curve(pr_curve['precision'], pr_curve['recall'])
            self.plot_confusion_matrix(confusion_matrix, ['Class 0', 'Class 1'])
        """

    def train(self):
        n_total_steps = len(self.train_dataloader)
        train_target = []
        train_pred = []
        self.model.train()
        for batch, (feature, target) in enumerate(self.train_dataloader):
            feature, target = feature.to(self.device), target.type(torch.LongTensor).to(self.device)

            target_one_hot = torch.zeros((target.size(0), self.n_classes))
            target_one_hot = target_one_hot.to(self.device)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1)

            prediction = self.model(feature)

            self.train_loss = self.loss_fn(prediction, target)
            self.optimizer.zero_grad()
            self.train_loss.backward()
            self.optimizer.step()

            train_target.append(target)
            train_pred.append(prediction)

            verbose_step = int(n_total_steps/5)
            if (batch + 1) % verbose_step == 0:
                print(f"Epoch [{self.epoch + 1}/{ self.epochs}], Step [{batch + 1}/{n_total_steps}], Loss: {self.train_loss.item():.4f}, Learning Rate {self.optimizer.param_groups[0]['lr']}")

        train_target = torch.cat(train_target, dim=0)
        train_pred = torch.cat(train_pred, dim=0)
        print(f"Calculating train metrics")
        train_metrics = self.calculate_metrics(target=train_target.detach().cpu().numpy(), prediction=train_pred.argmax(1).detach().cpu().numpy(), probabilities=train_pred.detach().cpu().numpy())
        self.train_some_metrics.append(train_metrics)

    def eval(self):
        eval_target = []
        eval_pred = []
        with torch.no_grad():
            self.model.eval()
            for batch, (feature, target) in enumerate(self.eval_dataloader):
                feature, target = feature.to(self.device), target.type(torch.LongTensor).to(self.device)

                target_one_hot = torch.zeros((target.size(0), self.n_classes))
                target_one_hot = target_one_hot.to(self.device)
                target_one_hot.scatter_(1, target.unsqueeze(1), 1)

                prediction = self.model(feature)
                self.eval_loss = self.loss_fn(prediction, target)
                eval_target.append(target)
                eval_pred.append(prediction)
            self.scheduler.step(self.eval_loss)
            eval_target = torch.cat(eval_target, dim=0)
            eval_pred = torch.cat(eval_pred, dim=0)
        print(f"Calculating evaluation metrics")
        eval_metrics = self.calculate_metrics(target=eval_target.detach().cpu().numpy(), prediction=eval_pred.argmax(1).detach().cpu().numpy(), probabilities=eval_pred.detach().cpu().numpy())
        self.eval_some_metrics.append(eval_metrics)
    def test(self):
        test_target = []
        test_pred = []
        with torch.no_grad():
            self.model.eval()
            for batch, (feature, target) in enumerate(self.test_dataloader):
                feature, target = feature.to(self.device), target.type(torch.LongTensor).to(self.device)
                target_one_hot = torch.zeros((target.size(0), self.n_classes))
                target_one_hot = target_one_hot.to(self.device)
                target_one_hot.scatter_(1, target.unsqueeze(1), 1)
                prediction = self.model(feature)
                test_target.append(target)
                test_pred.append(prediction)

            test_target = torch.cat(test_target, dim=0)
            test_pred = torch.cat(test_pred, dim=0)
        print(f"Calculating test metrics")
        self.test_some_metrics = self.calculate_metrics(target=test_target.detach().cpu().numpy(), prediction=test_pred.argmax(1).detach().cpu().numpy(), probabilities=test_pred.detach().cpu().numpy())

    def calculate_metrics(self, target, prediction, probabilities):
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report
        from sklearn.metrics import precision_recall_curve, roc_curve, auc

        metrics = {}

        self.cm = confusion_matrix(y_true=target, y_pred=prediction)
        metrics['confusion_matrix'] = self.cm
        print(self.cm)

        # Calculate precision and recall using classification report
        report = classification_report(target, prediction, output_dict=True)

        #print(classification_report(target, prediction))

        precision = report['1']['precision']
        recall = report['1']['recall']
        f1_score = report['1']['f1-score']

        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1_score
        print(f"Precision: {(precision):>0.2f} \tRecall: {(recall):>0.2f} \tF1: {(f1_score):>0.2f}")

        # Calculate PR curve
        precision, recall, _ = precision_recall_curve(target, probabilities[:,1])
        metrics['pr_curve'] = {'precision': precision, 'recall': recall}
        #print(f"Precision: {precision} \tRecall: {recall}")

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(target, probabilities[:,1])
        metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr)}
        #print(f"FPR: {fpr} \tTPR: {tpr}")
        print(f"AUC: {(auc(fpr, tpr)):>0.2f}")

        return metrics


    def plot_precision_recall_curve(self, precision, recall):
        import matplotlib.pyplot as plt
        plt.plot(recall, precision, color='b', label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()

    def plot_roc_curve(self, fpr, tpr, auc):
        import matplotlib.pyplot as plt
        plt.plot(fpr, tpr, color='b', label=f'ROC Curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='r', linestyle='--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self, confusion_matrix, classes):
        import matplotlib.pyplot as plt
        import numpy as np
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes) #, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title('Confusion Matrix')

        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                plt.text(j, i-0.1, str(confusion_matrix[i, j]), ha='center', va='center', color='black')
                plt.text(j, i+0.1, f'{confusion_matrix[i, j] / np.sum(confusion_matrix) * 100:.2f}%', ha='center', va='center', color='black', fontweight='bold')

        plt.show()

    def plot_f1_score_over_epochs(self, metrics, title='F1-Score over Epochs'):
        import matplotlib.pyplot as plt
        epochs = range(1, len(metrics) + 1)
        f1_scores = [metric['f1'] for metric in metrics]

        plt.plot(epochs, f1_scores, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('F1-Score')
        plt.title(title)
        plt.grid(True)
        plt.show()

    def plot_metrics_over_epochs(self, metrics, title='Metrics over Epochs'):
        import matplotlib.pyplot as plt
        epochs = range(1, len(metrics) + 1)
        f1_scores = [metric['f1'] for metric in metrics]
        precision = [metric['precision'] for metric in metrics]
        recall = [metric['recall'] for metric in metrics]

        plt.plot(epochs, f1_scores, marker='o', label='F1-Score')
        plt.plot(epochs, precision, marker='o', label='Precision')
        plt.plot(epochs, recall, marker='o', label='Recall')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_model(self):
        path = "model.pth"
        torch.save(self.model, path)

