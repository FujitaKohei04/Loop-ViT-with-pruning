# bViT 論文と本実験の設定比較

- 論文: Byra et al., "Investigating Single-Block Recurrence in Vision Transformers for Image Recognition", [arXiv:2605.10661 (HTML v2)](https://arxiv.org/html/2605.10661v2)
- 本実験の参照元: [train_bvit.py](../../train_bvit.py), [models/bViT/bViT.py](../../models/bViT/bViT.py), [dl_dataset/registry.py](../../dl_dataset/registry.py), `outputs/bvit/*/config.json`
- 対象 run: `bvit_s` / `bvit_b` / `vit_s_scratch` / `vit_b_scratch` on CIFAR-100（2026-10〜11 実施）
- 作成日: 2026-11-10

凡例: ✅ 同じ ／ ⚠️ 違う ／ ❓ 論文に記載がなく、本実験が仮定で決めた値

## 0. 大前提：論文には CIFAR をスクラッチで学習する実験がない

論文がスクラッチで学習しているのは **ImageNet-1K だけ**である。CIFAR-100 は、ImageNet-1K で事前学習した重みからの**転移学習**としてしか出てこない。

| 論文 Table 2 (CIFAR-100, IN-1K 事前学習から) | ViT-B | bViT-B |
|---|---|---|
| Linear probing | 0.788 | 0.772 |
| Full fine-tuning | 0.888 | 0.880 |

したがって、本実験の結果（bViT-B 0.6988 / ViT-B 0.6747）に直接対応する論文の数値はない。本実験は「論文の ImageNet 用レシピを CIFAR-100 のスクラッチ学習に当てはめたもの」である。

参考として、論文の ImageNet-1K での結果（Table 1）と本実験の CIFAR-100 での結果を並べる。

| モデル | 論文 IN-1K val acc | 本実験 CIFAR-100 top1 (EMA, 1 run) |
|---|---|---|
| ViT-S | 0.782 ± 0.003 | 0.7326 |
| ViT-B | 0.789 ± 0.003 | 0.6747 |
| bViT-S | 0.681 ± 0.003 | 0.6875 |
| bViT-B | 0.779 ± 0.002 | 0.6988 |

## 1. データ

| 項目 | 論文 | 本実験 | |
|---|---|---|---|
| データセット | ImageNet-1K（1.28M 枚、1000 クラス） | CIFAR-100（50k 枚、100 クラス） | ⚠️ |
| 入力解像度 | 224×224 | 32×32 | ⚠️ |
| 正規化 | ImageNet の mean/std | CIFAR-100 の mean/std | ⚠️（データに合わせた変更） |
| クロップ | RandomResizedCrop(224, scale 0.08–1, ratio 3/4–4/3) | RandomCrop(32, padding=4, reflect)（`augment=auto`→`small`） | ⚠️ |
| 水平反転 | あり | あり | ✅ |
| RandAugment | あり（パラメータの記載なし） | n=2, m=9（torchvision のデフォルト） | ❓ |
| Random Erasing | あり（確率の記載なし） | p=0.25（DeiT の値） | ❓ |
| 評価時の前処理 | 記載なし（ImageNet の慣例なら Resize→CenterCrop） | 前処理なし（正規化のみ） | ❓ |
| Repeated augmentation | 記載なし | なし | ❓ |

## 2. モデル

| 項目 | 論文 | 本実験 | |
|---|---|---|---|
| パッチサイズ | 16 | 4 | ⚠️ |
| トークン数 | 196+1 = 197 | 64+1 = 65 | ⚠️ |
| S / B / L の幅・ヘッド数 | 384/6, 768/12, 1024/16 | 同じ | ✅ |
| 深さ（ステップ数） | ViT 12 ブロック ↔ bViT 1 ブロック×12 ステップ（L は 24） | 同じ | ✅ |
| ブロック構造 | pre-norm MHSA + FFN、残差接続あり | 同じ | ✅ |
| 正規化層 | RMSNorm | RMSNorm | ✅ |
| RMSNorm の eps | 記載なし | 1e-6 | ❓ |
| 活性化関数 | GELU | GELU | ✅ |
| CLS トークン・位置埋め込み | 学習可能 | 学習可能 | ✅ |
| ヘッド | 最終 CLS → 線形層 | 最終 RMSNorm → 線形層 | ❓（最終 norm の有無は記載なし） |
| MLP ratio | 記載なし（パラメータ数から逆算すると 4） | 4 | ❓ |
| qkv bias | 記載なし | あり（q/k/v は別々の Linear） | ❓ |
| 初期化 | 記載なし | trunc_normal(0.02)、ヘッドはゼロ初期化 | ❓ |
| Stochastic depth | なし | なし | ✅ |
| Dropout | 「強い dropout は避けた」（値の記載なし） | なし | ❓ |
| Step embedding・register | 付録 B の変種のみ | デフォルトはオフ | ✅ |
| パラメータ数 | ViT-B 86.6M / bViT-B 8.6M / ViT-S 22.0M / bViT-S 2.5M | 85.2M / 7.25M / 21.4M / 1.86M | ⚠️（パッチ埋め込み・位置埋め込み・ヘッドのサイズ差による） |

## 3. 最適化

| 項目 | 論文 | 本実験 | |
|---|---|---|---|
| Optimizer | AdamW | AdamW | ✅ |
| ベース LR | 5e-4 | 5e-4 | ✅ |
| Weight decay | 0.05 | 0.05（norm・bias・cls・pos には掛けない） | ✅ / ❓（除外対象の記載なし） |
| エポック数 | 300 | 300 | ✅ |
| Warmup | 30 エポック、0.033×LR から線形 | 同じ | ✅ |
| スケジュール | cosine で 0 まで | 同じ（step 単位で更新） | ✅ |
| 実効バッチ | 1024（256 × 4 GPU） | 1024（128 × 勾配累積 8、1 GPU） | ✅（値）/ ⚠️（実現方法） |
| 1 エポックの step 数 / 総 step 数 | 約 1251 / 約 375k | 48 / **14,400** | ⚠️（約 26 分の 1） |
| Label smoothing | 0.1 | 0.1 | ✅ |
| Mixup / CutMix | α=0.8 / α=1.0 | 同じ | ✅ |
| Mixup と CutMix の切り替え | 記載なし | バッチごとに 50/50 | ❓ |
| 勾配クリッピング | 記載なし | 1.0 | ❓ |
| 混合精度 | 記載なし | fp16 AMP + GradScaler | ❓ |
| EMA decay | 0.99995 | 0.998696（`ema_decay=auto`：EMA の時定数が総 step 数に占める割合 ≒5.3% を論文に合わせた） | ⚠️ |
| EMA の更新頻度 | 記載なし | optimizer step ごと | ❓ |

## 4. 評価・実行環境

| 項目 | 論文 | 本実験 | |
|---|---|---|---|
| 報告する値 | 平均 ± 標準偏差（Table 1、複数 run） | seed=42 の 1 run | ⚠️ |
| 評価に使うモデル | 記載なし | EMA モデルの最終エポック（`final_eval.json`） | ❓ |
| GPU | H100 × 4 | RTX 2080 Ti × 1 | ⚠️ |
| フレームワーク | PyTorch（「確立された PyTorch の ViT 学習実装に沿う」） | PyTorch + timm（Mixup、EMA） | ✅ / ❓ |
| コード公開 | なし（論文中は "[\*\*\*]"） | — | — |

## 5. 論文の転移学習の設定（付録 F）

論文と同じ形で CIFAR-100 の数値を比べたい場合の参考。

| 項目 | 値 |
|---|---|
| 初期化 | ImageNet-1K で学習したチェックポイント。ヘッドは新しい線形層に置き換え |
| エポック数 | 40 |
| バッチ | CIFAR-10/100 は 128、その他は 64 |
| スケジュール | 5 エポックの線形 warmup → cosine decay |
| Optimizer / 損失 | AdamW / cross-entropy |
| Linear probing | LR 1e-3, WD 0.001 |
| Full fine-tuning | LR 5e-5, 最小 LR 1e-6, WD 0.05 |
| LoRA (r=8, α=8, dropout なし) | LR 5e-4, 最小 LR 5e-6, WD 1e-4 |
| Time-embedding tuning（bViT のみ） | 約 9K パラメータ、ゼロ初期化、共有ブロックは固定 |
| 報告 | 3 run の平均 ± 標準偏差、H100 × 1 |
| 記載なし | CIFAR の入力リサイズ、fine-tuning 時のデータ拡張 |

## 6. 別 GPU で試すときの注意点

- **bf16**：RTX 2080 Ti は bf16 を使えないため fp16 にしている。A100 や H100 なら bf16 にすると、オーバーフローの心配がなくなり GradScaler も不要になる。論文に記載はないが、H100 なら bf16 を使った可能性が高いと推測される。現状は `train_bvit.py` の `autocast()` と `evaluate()` で dtype が fp16 に固定されているので、切り替えるにはコードの変更が必要。
- **バッチの組み方**：GPU を増やして 256×4 の DDP にすると論文と同じ形になる。BatchNorm がないので勾配累積でも数学的には同じはずだが、実効バッチ 1024 は変えないこと。
- **データと解像度**：論文との差がいちばん大きいのはここ（データ規模、解像度、パッチサイズ、総 step 数）。論文の数値と直接比べるには、次のどちらかが必要。
  1. ImageNet-1K（または ImageNet-100）を 224px・パッチ 16 でスクラッチ学習する
  2. 論文と同じく、IN-1K 事前学習 → CIFAR-100 fine-tuning（§5 の設定）にする
