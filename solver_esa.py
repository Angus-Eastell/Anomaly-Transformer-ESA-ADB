import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader_esa import get_loader_segment, ESALabelsParser
from ESA_metrics import ESAScores, ADTQC, ChannelAwareFScore
import pandas as pd
import gc

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.val_loss2_min = np.inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}
    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        if self.mode == 'train':
          self.train_loader = get_loader_segment(self.data_path, train_length = self.train_length, test_length = self.test_length,
                                                batch_size=self.batch_size, win_size=self.win_size,
                                                mode='train',
                                                dataset=self.dataset)
          self.vali_loader = get_loader_segment(self.data_path, train_length = self.train_length, test_length = self.test_length,
                                                batch_size=self.batch_size, win_size=self.win_size,
                                                mode='val',
                                                dataset=self.dataset)

        if self.mode == 'test':  

          self.train_loader = get_loader_segment(self.data_path, train_length = self.train_length, test_length = self.test_length,
                                                batch_size=self.batch_size, win_size=self.win_size,
                                                mode='train',
                                                dataset=self.dataset)
                                                                                  
          self.test_loader = get_loader_segment(self.data_path, train_length = self.train_length, test_length = self.test_length,
                                                batch_size=self.batch_size, win_size=self.win_size,
                                                mode='test',
                                                dataset=self.dataset)
          self.thre_loader = get_loader_segment(self.data_path, train_length = self.train_length, test_length = self.test_length,
                                                batch_size=self.batch_size, win_size=self.win_size,
                                                mode='thre',
                                                dataset=self.dataset)

          # Load labels
          self.labels_parser = ESALabelsParser(self.labels_csv_path)

        # channel names
        self.channel_names = self.target_channels
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.build_model()
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=7, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.vali_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def _compute_esa_metrics(self, predictions, ground_truth, anomaly_scores):
        """
        Compute ESA metrics
        
        Args:
            predictions: (n_samples, n_channels) - binary predictions
            ground_truth: (n_samples, n_channels) - ground truth
            anomaly_scores: (n_samples, n_channels) - anomaly scores
        """
        print("\n" + "="*60)
        print("ESA ANOMALY DETECTION BENCHMARK METRICS")
        print("="*60)
        
        # Get timestamps
        timestamps = self.test_loader.dataset.get_timestamps()
        full_range = (timestamps.iloc[0], timestamps.iloc[-1])
        
        
        
        # Get ground truth labels in ESA format
        y_true_df = self.labels_parser.get_labels_dataframe(
            channel_filter=self.channel_names
        )
        
        start, end = full_range

        # keep only events that overlap the telemetry time range
        y_true_df = y_true_df[
            (y_true_df["EndTime"] >= start) &
            (y_true_df["StartTime"] <= end)
        ].copy()

        # clip event bounds so all events lie inside full_range (satisfies metric assertions)
        y_true_df["StartTime"] = y_true_df["StartTime"].clip(lower=start)
        y_true_df["EndTime"]   = y_true_df["EndTime"].clip(upper=end)
        print("y_true_df columns:", y_true_df.columns.tolist())
        print("Filtered labels:", len(y_true_df),
            "range:", y_true_df["StartTime"].min(), "to", y_true_df["EndTime"].max())
        

        print(f"\nGround truth events: {len(y_true_df)}")
        print(f"Unique anomaly IDs: {y_true_df['ID'].nunique()}")
        print(f"Time range: {full_range[0]} to {full_range[1]}")
        
        """# Create predictions in ESA format (multi-channel)
        y_pred_dict = {}
        for ch_idx, ch_name in enumerate(self.channel_names):
            y_pred_channel = []
            for i in range(len(predictions)):
                y_pred_channel.append([timestamps.iloc[i], int(predictions[i, ch_idx])])
            y_pred_dict[ch_name] = y_pred_channel

        # single channel format
        y_any_pred_dict = {}
        global_pred = predictions.any(axis=1).astype(int)
        y_any_pred = []
        for i in range(len(global_pred)):
            y_any_pred.append([timestamps.iloc[i], int(global_pred[i])])

        #y_any_pred_dict["is_anomaly"] = y_any_pred"""

        ts = timestamps.tolist()  # avoid repeated .iloc

        y_pred_dict = {
            ch_name: [[t, int(p)] for t, p in zip(ts, predictions[:, ch_idx])]
            for ch_idx, ch_name in enumerate(self.channel_names)
        }

        global_pred = predictions.any(axis=1).astype(int)
        y_any_pred = [[t, int(p)] for t, p in zip(ts, global_pred)]

                
        # 1. ESA Scores (Event-wise and Affiliation-based)
        print("\n--- Event-wise and Affiliation-based Scores ---")
        try:
            # Use first channel for basic ESA scores
            esa_metric = ESAScores(
                betas=self.beta,
                full_range=full_range
            )
            print("Telemetry range:", full_range[0], "to", full_range[1])
            print("Labels range:", y_true_df["StartTime"].min(), "to", y_true_df["EndTime"].max())
            # Convert single channel for ESAScores
            #y_pred_first = y_pred_dict[self.channel_names[0]]
            esa_results = esa_metric.score(y_true_df, y_any_pred)
            
            for metric_name, value in esa_results.items():
                print(f"{metric_name:30s}: {value:8.4f}")
                
        except Exception as e:
            print(f"Error computing ESA scores: {e}")
            import traceback
            traceback.print_exc()
        
        # 2. Channel-Aware F-Score
        print("\n--- Channel-Aware F-Score ---")
        try:
            channel_metric = ChannelAwareFScore(
                beta=self.beta if isinstance(self.beta, float) else self.beta,
                full_range=full_range
                
            )
            
            channel_results = channel_metric.score(y_true_df, y_pred_dict)
            
            for metric_name, value in channel_results.items():
                print(f"{metric_name:30s}: {value:8.4f}")
                
        except Exception as e:
            print(f"Error computing channel-aware metrics: {e}")
        
        # 3. ADTQC (Latency metrics)
        print("\n--- ADTQC Latency Metrics ---")
        try:
            adtqc_metric = ADTQC(
                full_range=full_range
            )
            
            adtqc_results = adtqc_metric.score(y_true_df, y_pred_dict)
            
            for metric_name, value in adtqc_results.items():
                if isinstance(value, float):
                    print(f"{metric_name:30s}: {value:8.4f}")
                else:
                    print(f"{metric_name:30s}: {value}")
                    
        except Exception as e:
            print(f"Error computing ADTQC metrics: {e}")
        
        print("="*60 + "\n")

        return esa_results, channel_results, adtqc_results

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        start_time = time.time()
        criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            # Metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.test_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = criterion(input, output)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            metric = metric.unsqueeze(-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1, self.output_c)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1, self.output_c)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)

        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        """
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1
        """

        pred = np.array(pred)
        gt = np.array(gt)
        gt = (gt == 2).astype(int)

        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        end_time = time.time()

        inference_time = end_time - start_time

        print('Inference Time:', inference_time)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        gt_any = gt.any(axis=1)
        pred_any = pred.any(axis=1)

        accuracy = accuracy_score(gt_any, pred_any)

        precision, recall, f_score, support = precision_recall_fscore_support(gt_any, pred_any,
                                                                              average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))

        esa_results, channel_results, adtqc = self._compute_esa_metrics(
                predictions=pred,
                ground_truth=gt,
                anomaly_scores=test_energy
            )
        
        timestamps = self.test_loader.dataset.get_timestamps().reset_index(drop=True)

        df_pred = pd.DataFrame(
            pred,
            columns=self.channel_names
        )
        df_pred.insert(0, "timestamp", timestamps)


        return accuracy, precision, recall, f_score, esa_results, channel_results, adtqc, pred, inference_time

      
    def test_per_channel(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)

        start_time = time.time()

        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            loss = criterion(input, output)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            metric = metric.unsqueeze(-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0)#.reshape(-1)
        #attends_energy = attens_energy.reshape(-1, attens_energy.shape[-1])
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = criterion(input, output)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            # Metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            metric = metric.unsqueeze(-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
  
        attens_energy = np.concatenate(attens_energy, axis=0)#.reshape(-1)
        #attends_energy = attens_energy.reshape(-1, attens_energy.shape[-1])
        test_energy = np.array(attens_energy)

        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        #print(combined_energy.shape)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio, axis = (0,1))
        print("Per channel thresholds :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = criterion(input, output)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            metric = metric.unsqueeze(-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1, self.output_c)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1, self.output_c)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh[None, :]).astype(int)

        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        """
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1
        """

        pred = np.array(pred)
        gt = np.array(gt)
        gt = (gt == 2).astype(int)

        end_time = time.time()

        inference_time = end_time - start_time

        print('Inference Time:', inference_time)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        gt_any = gt.any(axis=1)
        pred_any = pred.any(axis=1)

        accuracy = accuracy_score(gt_any, pred_any)

        precision, recall, f_score, support = precision_recall_fscore_support(gt_any, pred_any,
                                                                              average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))

        esa_results, channel_results, adtqc = self._compute_esa_metrics(
                predictions=pred,
                ground_truth=gt,
                anomaly_scores=test_energy
            )
        
        timestamps = self.test_loader.dataset.get_timestamps().reset_index(drop=True)

        df_pred = pd.DataFrame(
            pred,
            columns=self.channel_names
        )
        df_pred.insert(0, "timestamp", timestamps)

        return accuracy, precision, recall, f_score, esa_results, channel_results, adtqc, df_pred, inference_time

    def test_low_mem(self):

      self.model.load_state_dict(
          torch.load(
              os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth'),
              map_location=self.device
          )
      )
      self.model.eval()
      temperature = 50

      print("======================TEST MODE (LOW MEMORY)======================")

      criterion = nn.MSELoss(reduction='none')
      start_time = time.time()

      def compute_assoc_loss(series, prior):
          series_loss = 0.0
          prior_loss = 0.0

          for u in range(len(prior)):
              norm_prior = prior[u] / torch.sum(prior[u], dim=-1, keepdim=True)
              norm_prior = norm_prior.expand(-1, -1, -1, self.win_size)

              series_loss += my_kl_loss(series[u], norm_prior.detach()) * temperature
              prior_loss += my_kl_loss(norm_prior, series[u].detach()) * temperature

          return series_loss, prior_loss

      # ===================== (1) TRAIN ENERGY =====================
      train_energy = []

      with torch.no_grad():
          for i, (input_data, _) in enumerate(self.train_loader):
              input = input_data.float().to(self.device)

              output, series, prior, _ = self.model(input)
              loss = criterion(input, output)

              series_loss, prior_loss = compute_assoc_loss(series, prior)

              metric = torch.softmax(-(series_loss + prior_loss), dim=-1)
              metric = metric.unsqueeze(-1)

              cri = (metric * loss).cpu().numpy()
              train_energy.append(cri)

              del input, output, series, prior, loss, metric, cri

      train_energy = np.concatenate(train_energy, axis=0)

         
      del self.train_loader
      gc.collect()
      torch.cuda.empty_cache()

      # ===================== (2) THRESHOLD =====================
      test_energy_tmp = []

      with torch.no_grad():
          for input_data, _ in self.thre_loader:
              input = input_data.float().to(self.device)

              output, series, prior, _ = self.model(input)
              loss = criterion(input, output)

              series_loss, prior_loss = compute_assoc_loss(series, prior)

              metric = torch.softmax(-(series_loss + prior_loss), dim=-1)
              metric = metric.unsqueeze(-1)

              cri = (metric * loss).cpu().numpy()
              test_energy_tmp.append(cri)

              del input, output, series, prior, loss, metric, cri

      test_energy_tmp = np.concatenate(test_energy_tmp, axis=0)

      combined_energy = np.concatenate([train_energy, test_energy_tmp], axis=0)
      thresh = np.percentile(combined_energy, 100 - self.anormly_ratio, axis=(0, 1))

      print("Per channel thresholds:", thresh)

      del self.thre_loader
      gc.collect()
      torch.cuda.empty_cache()

      # ===================== (3) TEST SET =====================
      attens_energy = []
      test_labels = []

      with torch.no_grad():
          for input_data, labels in self.test_loader:
              input = input_data.float().to(self.device)

              output, series, prior, _ = self.model(input)
              loss = criterion(input, output)

              series_loss, prior_loss = compute_assoc_loss(series, prior)

              metric = torch.softmax(-(series_loss + prior_loss), dim=-1)
              metric = metric.unsqueeze(-1)

              cri = (metric * loss).cpu().numpy()

              attens_energy.append(cri)
              test_labels.append(labels)

              del input, output, series, prior, loss, metric, cri


      attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1, self.output_c)
      test_labels = np.concatenate(test_labels, axis=0).reshape(-1, self.output_c)

      pred = (attens_energy > thresh[None, :]).astype(int)
      gt = (test_labels == 2).astype(int)

      end_time = time.time()
      inference_time = end_time - start_time

      print("Inference Time:", inference_time)

      from sklearn.metrics import accuracy_score, precision_recall_fscore_support

      gt_any = gt.any(axis=1)
      pred_any = pred.any(axis=1)

      accuracy = accuracy_score(gt_any, pred_any)
      precision, recall, f_score, _ = precision_recall_fscore_support(
          gt_any, pred_any, average='binary'
      )

      print(
          "Accuracy : {:.4f}, Precision : {:.4f}, Recall : {:.4f}, F-score : {:.4f}".format(
              accuracy, precision, recall, f_score
          )
      )

      esa_results, channel_results, adtqc = self._compute_esa_metrics(
          predictions=pred,
          ground_truth=gt,
          anomaly_scores=attens_energy
      )

      timestamps = self.test_loader.dataset.get_timestamps().reset_index(drop=True)

      df_pred = pd.DataFrame(pred, columns=self.channel_names)
      df_pred.insert(0, "timestamp", timestamps)

      return accuracy, precision, recall, f_score, esa_results, channel_results, adtqc, df_pred, inference_time

