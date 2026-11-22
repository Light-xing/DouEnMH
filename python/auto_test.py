import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from douzero.evaluation.autosimu import evaluate

random.seed(78419)
np.random.seed(78419)
torch.manual_seed(78419)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(78419)
torch.set_default_tensor_type('torch.FloatTensor')


def get_latest_model(dir_path, position='landlord'):
    """获取指定位置的最新模型文件 (格式: {position}_{frame}.ckpt)."""
    files = os.listdir(dir_path)
    prefix = position + '_'
    model_files = [f for f in files if f.startswith(prefix)]

    if not model_files:
        return None
    # Parse frame number from filename like "landlord_100000.ckpt"
    return max(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))


def main(checkpoint_dir, output_csv, output_all_csv, best_txt, dire, position):
    print("start_" + position)
    output_csv = os.path.join(dire, output_csv)
    output_all_csv = os.path.join(dire, output_all_csv)
    best_txt = os.path.join(dire, best_txt)
    max_rows = 200

    if not os.path.exists(output_csv):
        pd.DataFrame(columns=["model_id", "win_rate", "score"]).to_csv(output_csv, index=False)

    if not os.path.exists(output_all_csv):
        pd.DataFrame(columns=["model_id", "win_rate", "score"]).to_csv(output_all_csv, index=False)

    player_1_bid = 'random'
    player_2_bid = 'random'
    player_3_bid = 'random'

    # 对手出牌模型使用同目录下最新的 landlord_down/up 模型
    latest_down = get_latest_model(checkpoint_dir, 'landlord_down')
    latest_up = get_latest_model(checkpoint_dir, 'landlord_up')
    player_2_playcard = os.path.join(checkpoint_dir, latest_down) if latest_down else 'random'
    player_3_playcard = os.path.join(checkpoint_dir, latest_up) if latest_up else 'random'
    print(f'Opponent models: {player_2_playcard}, {player_3_playcard}')

    while True:
        latest_model = get_latest_model(checkpoint_dir, position)

        if not latest_model:
            print("No model found. Waiting for the next check...")
            time.sleep(60)
            continue

        model_id = int(latest_model.split('_')[-1].split('.')[0])

        latest_model_path = os.path.join(checkpoint_dir, latest_model)

        df = pd.read_csv(output_csv)
        if model_id in df["model_id"].values:
            time.sleep(60)
            print('no new model')
            continue

        num_workers = 2
        eval_data = 'eval_data.pkl'

        ti = time.time()
        win_rate, score = evaluate(
            player_1_bid,
            player_2_bid,
            player_3_bid,
            latest_model_path,
            player_2_playcard,
            player_3_playcard,
            eval_data,
            num_workers,
            position)
        print('eval ' + position + ':', time.time() - ti)

        new_row = pd.DataFrame([[model_id, win_rate, score]], columns=["model_id", "win_rate", "score"])

        df = pd.read_csv(output_csv)
        df = pd.concat([df, new_row], ignore_index=True)
        df = df.tail(max_rows)
        df.to_csv(output_csv, index=False)

        df_all = pd.read_csv(output_all_csv)
        df = pd.concat([df_all, new_row], ignore_index=True)
        df_all.to_csv(output_all_csv, index=False)

        best_win_rate_model = df.loc[df["win_rate"].idxmax()]["model_id"]
        best_score_model = df.loc[df["score"].idxmax()]["model_id"]
        with open(best_txt, "w") as f:
            f.write(f"Best win rate model: {best_win_rate_model}\n")
            f.write(f"Best score model: {best_score_model}\n")

        plt.figure(figsize=(10, 6))
        plt.plot(df["model_id"], df["win_rate"], marker='o')
        plt.xlabel("Model ID")
        plt.ylabel("Win Rate")
        plt.title("Win Rate over Models")
        plt.savefig(dire + "win_rate.jpg")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(df["model_id"], df["score"], marker='o', color='r')
        plt.xlabel("Model ID")
        plt.ylabel("Score")
        plt.title("Score over Models")
        plt.savefig(dire + "score.jpg")
        plt.close()

        time.sleep(60)


if __name__ == "__main__":
    checkpoint_dir = "douzero_checkpoints/douzero/"
    output_csv = "landlord_eval_out.csv"
    output_all_csv = "landlord_eval_out_all.csv"
    best_txt = "landlord_best.txt"
    dire = "landlord/"
    position = 'landlord'
    main(checkpoint_dir, output_csv, output_all_csv, best_txt, dire, position)
