---
marp: true
theme: academic
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Denoising Diffusion Probabilistic Models (DDPM)

Ho et al., 2020

<br>

Kenichiro Goto
西川研究室 B3
2025-11-14

---

<!-- _header: Agenda -->



---

<!-- _class : lead -->







---

<!-- _header: 参考文献 -->




# Note (this is only for me to understand the paper)

latent空間でdiffusionするというのが肝
- 計算量が激減し、10~100倍早くなる
stable diffusionなどをPCで動かせる理由

latent構造には**意味的構造**がすでに圧縮されている
autoencoderが
- high-level な形状
- semantic consistency
- perceptual similarity
をlatentに押し込めるためdiffusionが学習するべき分布が扱いやすくなる

画像以外への応用が容易
- audio latent
- video latent
- 3D latent
- text latent (in progress)
- text latent (in progress)
高次元データの生成モデルはlatent表現を持つのが常識になりつつある

逆方向の理論 (score, 情報保存) が理解しやすくなる
pixel空間よりlatent空間の方が生成すべき確立分布が平滑なのでsocre (∇logp(x))の推定が安定

以下は論文の重要だと思った部分を抜き出したもの

## Abstract

既存のDMsはpixel空間で計算をしていたので大量のGPUを使用し、連続評価により推論は高価だった
LDMは latent空間で計算をすることで複雑性の減少と詳細維持・忠実性の間のほぼ最適と言えるバランスに到達した(という触れ込み)
- cross attention layer
- covolutional manner
などの導入により全般的な条件付き入力（テキストとかやbounding boxes）を可能にし、また高画質な画像の生成も同時に達成している

## Introduction
既存のdiffusion model
- 尤度ベースのクラスに属する
- GPUによる電力消費が激しい (e.g. 150-1000 V in 100 days)
これによって以下の状況をもたらした
1. GPUがたくさん必要
2. すでに学習されたモデルの評価にも、時間的・金銭的に負担がかかる。同じアーキテクチャのものでも最初から同じことをしないといけない (e.g. 25-1000 steps)
訓練とサンプリング（生成）の両方の複雑性を排除しつつ、性能は低下させないようにできないかというモチベーション

### Departure to Latent Space
すでにpixel空間で訓練されたdiffusion modelを分析することから始める
学習の段階
1. perceptual compression: high-frequencryな細かい部分を取り除き、少し意味的な変分も学習する
2. semantic compression: データの意味的で、概念的な組み合わせを学習する

学習を2段階に分ける
1. 知覚的にデータ空間に等しい低次元な表現空間を出力するautoencoderを訓練する
メモ：多分ここで出力される低次元空間をlatent spaceと呼んでいる
2. DMsをlatent spaceで学習するときには、既存の手法が行っていたような超過空間圧縮(訳し方が合っているかはわからない)を使用しない
このように複雑性を減らしてlatent spaceからの効率的な画像生成をできるようにしている
このようなモデル群をLDMsと名付ける

利点
- autoencoderの学習は1度でいい
- DMが訓練結果は使いまわせる
- または計算的に難しい問題のための探索に使える（初期リソース消費を抑えられる分、より深くまで探索できる）

contributions
1. ピュアなtransformer-basedモデルよりもスケールし、高次元データを扱える
  - 低次元なlatent spaceで動くのでより忠実で細かい画像が作れる
  - 高画質画像を効率的に作れる
2. pixelベースの既存手法やデータと同等の性能を、計算コスト、推論コストを抑えつつ実現
3. encoder/decoderとscore-basedの事前分布を同時に学習するということはしない（わけている）ので、繊細な作業が必要ない
4. $~1025^2 $ pxの高画質生成ができる
5. cross-attentionベースの条件付きの学習もできる
6. 使いまわせるようになったモデルを公開（これまでは公開のしようがなかった？）

## 2. Related work
時間がないので一度飛ばす

## 3. Method
- 計算コストがかかるという弱点を克服するため、明示的に生成の学習過程を分ける
  - 知覚的に画像空間と等しい空間を学習するautoencoderを活用する
効果
1. サンプリング（生成）が低次元空間で行われるので計算効率の良いDMが得られる
2. UNet由来のinductive biasを利用する（よくわかっていない）
3. general-purposeな圧縮モデルが得られる。その潜在空間はいろんな生成モデルの訓練や応用に使える




## 3.1 Perceptual Image Compression
perceptual compression modelは
- transformer (Patrick Esser, Robin Rombach, and Bj¨ orn Ommer. Taming
transformers for high-resolution image synthesis. CoRR,
abs/2012.09841, 2020.)
- autoencoder (perceptual lossを使う)
- patch-based adversarial objective
の組み合わせ。

1) Train an autoencoder:
Encoder: 
E(x)→z where $\mathbf{x} \in \mathcal{R}^{H\times W \times 3}$, $z \in \mathcal{R}^{h \times w \times c}$
Decoder: 
D(z)→$\tilde{x}$
The encoder downsamples by factor:
f = $2^m$
So H×W collapses to (H/f) × (W/f).
The latent has channels c but far fewer pixels.
👉 The goal is to remove only imperceptible detail (“perceptual compression”), while preserving all visible / semantic content.

2) Why perceptual loss + patch-GAN?
The paper uses:
- VGG perceptual loss (LPIPS-style)
- Patch-based discriminator (GAN loss)
This ensures:
- No blurriness (common with L2 autoencoders)
- Reconstructions stay on the “image manifold”
- Local texture realism is preserved
Pixel-space L1/L2 alone → blurry reconstructions.
Perceptual + patch-GAN → sharp, natural reconstructions.