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

def load_datasets_from_h5(file_path):
    import h5py
    with h5py.File(file_path, 'r') as file:
        datasets_group = file['datasets']
        x_train = datasets_group['x_train'][()]
        y_train = datasets_group['y_train'][()]
        x_test = datasets_group['x_test'][()]
        y_test = datasets_group['y_test'][()]
        x_val = datasets_group['x_val'][()]
        y_val = datasets_group['y_val'][()]
    return x_train, y_train, x_test, y_test, x_val, y_val



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

    def early_stop(self, validation_loss, verbose=True):
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
                 epochs, loss_fn, early_stop, optimizer, scheduler, n_classes,
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
        self.epochs = epochs

        self.train_loss = 0
        self.eval_loss = 0
        self.train_loss_total = []
        self.eval_loss_total = []

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
        self.save_metrics(self.train_some_metrics, filename='train_metrics.json')
        self.save_metrics(self.eval_some_metrics, filename='eval_metrics.json')
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

    def run_and_save_best_and_last_model(self, path, datafile):
        import os
        best_f1 = 0.0
        best_epoch = 0

        for self.epoch in range(self.epochs):
            print(f"Epoch {self.epoch + 1} of {self.model.__class__.__name__}\n-------------------------------")
            self.train()
            self.eval()
            if self.early_stop.early_stop(validation_loss=self.eval_loss):
                break

            #torch.save(self.model.state_dict(), "epo_"+str(self.epoch + 1)+"_"+str(self.model.__class__.__name__)+".pth")

            # Calculate F1-score from evaluation metrics
            f1 = self.eval_some_metrics[-1]['f1']
            # Check if current epoch has the best F1-score so far
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = self.epoch
                torch.save(self.model.state_dict(), os.path.join(path, "best_"+str(self.model.__class__.__name__)+".pth"))

        torch.save(self.model.state_dict(), os.path.join(path, "last_"+str(self.model.__class__.__name__)+".pth"))
        #self.test()

        self.save_model_meta_to_h5(file_path= os.path.join(path, datafile))
        self.save_loss_to_h5(file_path=os.path.join(path, datafile))
        self.save_metrics_to_h5(file_path=os.path.join(path, datafile), mode='train')
        self.save_metrics_to_h5(file_path=os.path.join(path, datafile), mode='eval')

        print(f"Loading and testing best model\n-------------------------------")
        print(f"Best epoch: {best_epoch + 1}")
        print(f"F1-score: {best_f1}")
        best_metrics = self.eval_some_metrics[best_epoch]
        print(f"Precision: {best_metrics['precision']}")
        print(f"Recall: {best_metrics['recall']}")
        self.model.load_state_dict(torch.load(os.path.join(path, "best_"+str(self.model.__class__.__name__)+".pth")))
        self.test()

        self.save_metrics_test_to_h5(file_path=os.path.join(path, datafile))


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
            self.train_loss_total.append(self.train_loss.item())
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
                self.eval_loss_total.append(self.eval_loss.item())
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
        import numpy as np

        metrics = {}

        # Check for NaN values
        # if prediction equals NaN then prediction.argmax(1) resulting in 0 for instance
        if np.isnan(prediction).any() and np.isnan(probabilities).any():
            print("Error: Input data contains NaN values.")
            return metrics

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

        if not np.isnan(probabilities).any():
            # Calculate PR curve
            precision, recall, _ = precision_recall_curve(target, probabilities[:,1])
            metrics['pr_curve'] = {'precision': precision, 'recall': recall}
            #print(f"Precision: {precision} \tRecall: {recall}")

            # Calculate ROC curves
            # class/neuron 1: spike
            fpr_1, tpr_1, thr_1 = roc_curve(target, probabilities[:,1])
            metrics['roc_curve_1'] = {'fpr': fpr_1, 'tpr': tpr_1, 'thr': thr_1, 'auc': auc(fpr_1, tpr_1)}
            #print(f"FPR: {fpr} \tTPR: {tpr}")
            print(f"class 1: AUC: {(auc(fpr_1, tpr_1)):>0.2f}")

            # class/neuron 0: noise
            fpr_0, tpr_0, thr_0 = roc_curve(target, probabilities[:,0])
            metrics['roc_curve_0'] = {'fpr': fpr_0, 'tpr': tpr_0, 'auc': auc(fpr_0, tpr_0)}
            #print(f"FPR: {fpr} \tTPR: {tpr}")
            print(f"class 0: AUC: {(auc(fpr_0, tpr_0)):>0.2f}")

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

    def plot_confusion_matrix_v2(self, confusion_matrix):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd
        cm_sum = np.sum(confusion_matrix, axis=1)
        cm_perc = confusion_matrix / cm_sum.astype(float) * 100
        annot = np.empty_like(confusion_matrix).astype(str)
        nrows, ncols = confusion_matrix.shape
        for i in range(nrows):
            for j in range(ncols):
                c = confusion_matrix[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (p, c)
        cm = pd.DataFrame(confusion_matrix)
        cm.index.name = 'True Class'
        cm.columns.name = 'Predicted Class'
        sns.heatmap(cm, annot=annot, fmt='', cmap='rocket_r')
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

    def plot_losses(self):
        import matplotlib.pyplot as plt
        import numpy as np
        train_epochs = range(1, len(self.train_loss_total) + 1)
        eval_epochs = np.linspace(1, len(self.train_loss_total), len(self.eval_loss_total))
        plt.plot(self.train_loss_total, 'b', label='Training Loss')
        plt.plot(self.eval_loss_total, 'r', label='Evaluation Loss')
        plt.title('Training and Evaluation Loss')
        plt.xlabel('Epochs (All Batches seen)')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


    def save_model(self):
        path = "model.pth"
        torch.save(self.model, path)

    def save_metrics_json(self, metrics, filename ='metrics.json'):
        import json
        import numpy as np
        def convert_ndarray(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert ndarray to list
            raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

        with open(filename, 'w') as file:
            json.dump(metrics, file, indent=4, default=convert_ndarray)
        print(f"Metrics saved to {filename}")


    def save_metrics_to_h5(self, file_path, mode):
        import h5py
        import numpy as np

        if mode == 'train':
            metrics = self.train_some_metrics
            group_name = 'metrics_train'
        elif mode == 'eval':
            metrics = self.eval_some_metrics
            group_name = 'metrics_evaluation'
        else:
            raise ValueError(f"Mode {mode} not supported. Please choose 'train' or 'eval'")

        with h5py.File(file_path, 'a') as file:
            metrics_group = file.create_group(group_name)
            for epoch, metric_dict in enumerate(metrics):
                epoch_group = metrics_group.create_group(f'epoch_{epoch}')
                for key, value in metric_dict.items():
                    if isinstance(value, np.ndarray):
                        epoch_group.create_dataset(key, data=value)
                    elif isinstance(value, (int, float)):
                        epoch_group.attrs[key] = value
                    elif isinstance(value, dict):
                        sub_group = epoch_group.create_group(key)
                        for sub_key, sub_value in value.items():
                            sub_group.create_dataset(sub_key, data=sub_value)
                    else:
                        raise ValueError(f"Unsupported data type for metric: {type(value)}")

    def save_metrics_test_to_h5(self, file_path):
        import h5py
        import numpy as np
        with h5py.File(file_path, 'a') as file:
            metric_group = file.create_group('metrics_test')
            for key, value in self.test_some_metrics.items():
                if isinstance(value, np.ndarray):
                    metric_group.create_dataset(key, data=value)
                elif isinstance(value, (int, float)):
                    metric_group.attrs[key] = value
                elif isinstance(value, dict):
                    sub_group = metric_group.create_group(key)
                    for sub_key, sub_value in value.items():
                        sub_group.create_dataset(sub_key, data=sub_value)
                else:
                    raise ValueError(f"Unsupported data type for metric: {type(value)}")
    def save_loss_to_h5(self, file_path):
        import h5py
        import numpy as np
        with h5py.File(file_path, 'a') as file:
            loss_group = file.create_group('loss')
            training_loss_stack = np.stack(self.train_loss_total)
            evaluation_loss_stack = np.stack(self.eval_loss_total)
            loss_group.create_dataset('training_loss', data=training_loss_stack)
            loss_group.create_dataset('evaluation_loss', data=evaluation_loss_stack)

    def save_model_meta_to_h5(self, file_path):
        import h5py
        with h5py.File(file_path, 'a') as file:
            model_meta_group = file.create_group('model_meta')
            if hasattr(self.model, 'get_model_metadata') and callable(self.model.get_model_metadata):
                model_metadata = self.model.get_model_metadata()
                for key, value in model_metadata.items():
                    model_meta_group.attrs[key] = value
            else:
                model_meta_group.attrs['error'] = 'Model metadata is not available'

    def save_trainer_meta_to_h5(self, file_path, trainer_meta_dict=None):
        import h5py
        with h5py.File(file_path, 'a') as file:
            trainer_meta_group = file.create_group('trainer_meta')
            if trainer_meta_dict is not None:
                for key, value in trainer_meta_dict.items():
                    trainer_meta_group.attrs[key] = value
            else:
                trainer_meta_group.attrs['error'] = 'Trainer metadata is not available'

    def save_model_to_h5(self, file_path):
        #TODO: function does not properly work
        import h5py
        with h5py.File(file_path, 'a') as file:
            model_group = file.create_group('model')
            model_group.attrs['model_summary'] = str(self.model.summary())
            # Save the model weights, configuration, etc.
            self.model.save(file.create_group('model_data'))

import torch

class using():
    def __init__(self, loaded_model, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        self.loaded_model = loaded_model.to(device)
        self.device = device

    def devide_2_vectors_into_equal_windows_with_step(self, x1, x2, window_size, step_size=None):
            """
            Devides vectors x1, x2 into windows with one window_size. step_size is used to generate more windows with overlap.
            :param x1: Input list to be devided.
            :param x2: Input list to be devided.
            :param window_size: Size of each window.
            :param step_size: If the step_size is not provided, it defaults to the window_size.
                If the step_size is set to True, it is set to half of the window_size.
                If the step_size is set to any other value, it is used directly as the step_size.
            :return: Returns for every input a list of lists. Each included list represents a window.
            """
            if step_size is None:
                step_size = window_size
            elif step_size is True:
                step_size = window_size // 2
            elif step_size is not None:
                step_size = int(step_size)
            x1_windows = []
            x2_windows = []
            for i in range(0, len(x1) - window_size + 1, step_size):
                x1_windows.append(x1[i:i + window_size])
                x2_windows.append(x2[i:i + window_size])
            return x1_windows, x2_windows

    def is_spike_in_window(self, window):
        # old version (pytorch unrecommented, here not working)
        #loaded_model = torch.load(self.model, map_location=torch.device(self.device))
        # new version with state_dict:
        # this has to be done outside the using class. The loaded model has to be given to using class.
        #from custom_models import TransformerModel
        #loaded_model = TransformerModel(input_dim=1, hidden_size=64, num_classes=2, num_layers=12, num_heads=8, dropout=0.1)
        #loaded_model.load_state_dict(torch.load('path_to_model.pth'))
        self.loaded_model.eval()
        window_tensor = torch.tensor(window, dtype=torch.float32)
        window_tensor = window_tensor.to(self.device)
        with torch.no_grad():
            output = self.loaded_model(window_tensor.unsqueeze(0))

        predicted_class = torch.argmax(output).item()

        # return True if the predicted class is 1 (spike), False otherwise
        return predicted_class == 1

    def detect_spikes_and_get_timepoints(self, windowed_data, timestamps):
        spikes = []
        for window_index in range(len(windowed_data)):
            window = windowed_data[window_index]
            window_timestamps = timestamps[window_index]

            timepoint_of_interest_idx = abs(window).argmax()
            timepoint_of_interest = window_timestamps[timepoint_of_interest_idx]
            # old method:
            #timepoint_of_interest = window_timestamps[9] if len(window_timestamps) >= 10 else None

            is_spike = self.is_spike_in_window(window)

            if is_spike:
                spikes.append(timepoint_of_interest)

        return spikes

    def application_of_model(self, signal_raw, timestamps):
        import numpy as np
        spiketrains = []
        for electrode_index in range(signal_raw.shape[1]):
            print(f"current electrode index: {electrode_index}")
            electrode_data = signal_raw[:, electrode_index]
            electrode_data_windowed, timestamps_windowed = self.devide_2_vectors_into_equal_windows_with_step(electrode_data, timestamps, window_size=20, step_size=None)
            spiketrains.append(self.detect_spikes_and_get_timepoints(electrode_data_windowed, timestamps_windowed))
        return np.array(spiketrains, dtype=object)

    def save_results_to_h5(self, file_path):
        import h5py
        with h5py.File(file_path, 'a') as file:
            spiketrains_group = file.create_group('spiketrains')


