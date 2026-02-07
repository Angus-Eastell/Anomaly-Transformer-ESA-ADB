import os
import argparse
import json
import pandas as pd
import csv
import numpy as np

from torch.backends import cudnn
from utils.utils import *

from solver_esa import Solver


def str2bool(v):
    return v.lower() in ('true')

def write_results_csv(
    csv_path,
    accuracy,
    precision,
    recall,
    f_score,
    esa_results,
    channel_results,
    adtqc,
    inference_time,
    channel_thresholds,
    run_name):


    row = {
        "run_name": run_name,
        "total_inference_time": inference_time,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
    }

    # flatten ESA metrics
    for k, v in esa_results.items():
        row[f"esa_{k}"] = v

    for k, v in channel_results.items():
        row[f"channel_{k}"] = v

    for k, v in adtqc.items():
        row[f"adtqc_{k}"] = v

    for ch, thr in channel_thresholds.items():
        row[f"threshold_{ch}"] = thr

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main(config):
    cudnn.benchmark = True

    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if (not os.path.exists(config.results_path)):
        mkdir(config.results_path)

    run_dir = os.path.join(config.results_path, config.run_name)
    os.makedirs(run_dir, exist_ok=True)


    if config.mode == 'train':
        solver.train()
        print('Saving train config')
        with open(os.path.join(run_dir, "train_config.json"), "w") as f:
            json.dump(vars(config), f, indent=4)
    elif config.mode == 'test':
        print('Saving test config')
        with open(os.path.join(run_dir, "test_config.json"), "w") as f:
            json.dump(vars(config), f, indent=4)

        accuracy, precision, recall, f_score, esa_results, channel_results, adtqc, thresh, pred_energy, pred_df, inference_time = solver.test_low_mem_overlapping_new()

        # Guard against global vs per-channel threshold
        if np.isscalar(thresh) or len(np.atleast_1d(thresh)) == 1:
            # Global threshold → apply to all channels
            channel_thresholds = {
                ch: float(thresh) for ch in config.target_channels
            }
        else:
            # Per-channel thresholds
            assert len(thresh) == len(config.target_channels), (
                f"Threshold length {len(thresh)} does not match "
                f"number of channels {len(config.target_channels)}"
            )
            channel_thresholds = dict(zip(config.target_channels, thresh))

        print('Writing Results')

        pred_path = os.path.join(run_dir, 'pred.csv')
        pred_df.to_csv(pred_path, index = False)

        energy_path = os.path.join(run_dir, 'pred_energy.csv')
        pred_energy.to_csv(energy_path, index = False)


        write_results_csv(
            csv_path=os.path.join(run_dir, "results.csv"),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f_score=f_score,
            esa_results=esa_results,
            channel_results=channel_results,
            adtqc=adtqc,
            inference_time = inference_time,
            channel_thresholds= channel_thresholds,
            run_name= config.run_name
            )

    elif config.mode == 'thresh_analysis':
      vali_energy = solver.thresh_analysis()

      print('Writing Results')

      vali_path = os.path.join(run_dir, 'vali_energy.csv')
      vali_energy.to_csv(vali_path, index = False)


    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'thresh_analysis'])
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--train_length', type=str, default='3_months')
    parser.add_argument('--test_length', type=str, default='84_months')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--results_path', type=str, default='results')
    parser.add_argument('--run_name', type=str, default='model')
    parser.add_argument('--anormly_ratio', type=float, default=1.00)
    parser.add_argument('--temperature', type=int, default=50)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--step', type =  int, default = 100)
    parser.add_argument('--mission', type= str, choices = ['mission_1','mission_2'])


    config = parser.parse_args()


    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
