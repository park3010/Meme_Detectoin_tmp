import os
import glob
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="예: /home/sujin/psj2003/meme_detection/result/5_seed/comparison/covid/resnet")
    args = parser.parse_args()

    metric_files = sorted(glob.glob(os.path.join(args.root, "v*", "metrics.csv")))

    if not metric_files:
        raise FileNotFoundError(f"No metrics.csv found under: {args.root}/v*/metrics.csv")

    dfs = []
    for fp in metric_files:
        df = pd.read_csv(fp)
        df["run"] = os.path.basename(os.path.dirname(fp))  # v1, v2 ...
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # 저장: 개별 run 전체 모음
    all_out = os.path.join(args.root, "all_metrics.csv")
    all_df.to_csv(all_out, index=False)

    # 숫자형 컬럼만 집계
    numeric_cols = all_df.select_dtypes(include="number").columns.tolist()

    # seed 같은 컬럼은 평균/표준편차 대상에서 빼고 싶으면 제외
    exclude_cols = {"seed"}
    metric_cols = [c for c in numeric_cols if c not in exclude_cols]

    summary = pd.DataFrame({
        "metric": metric_cols,
        "mean": [all_df[c].mean() for c in metric_cols],
        "std": [all_df[c].std(ddof=1) for c in metric_cols],   # sample std
    })

    # mean ± std 문자열 컬럼 추가
    summary["mean±std"] = summary.apply(
        lambda x: f"{x['mean']:.4f} ± {x['std']:.4f}", axis=1
    )

    summary_out = os.path.join(args.root, "summary_mean_std.csv")
    summary.to_csv(summary_out, index=False)

    print("\n[Collected files]")
    for fp in metric_files:
        print(fp)

    print("\n[Per-run metrics]")
    print(all_df)

    print("\n[Summary: mean ± std]")
    print(summary[["metric", "mean±std"]])

    print(f"\nSaved: {all_out}")
    print(f"Saved: {summary_out}")


if __name__ == "__main__":
    main()