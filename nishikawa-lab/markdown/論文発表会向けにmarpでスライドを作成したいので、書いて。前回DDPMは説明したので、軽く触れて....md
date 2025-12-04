

# **High-Resolution Image Synthesis with Latent Diffusion Models**

論文紹介  
Rombach et al., CVPR 2022  
(CompVis, LMU Munich & Runway ML)

## **1\. 概要 (Abstract)**

本論文は、**Latent Diffusion Models (LDMs)** を提案。

* **背景:** 従来の拡散モデル（DDPM等）はピクセル空間で動作するため、学習・推論の計算コストが極めて高い。  
* **提案:** 学習済みのAutoencoderを用いて画像を低次元の\*\*潜在空間（Latent Space）\*\*へ圧縮し、その空間上で拡散モデルを学習する。  
* **成果:**  
  * 画質を維持しつつ、計算コストを劇的に削減。  
  * Cross-Attention層の導入により、テキストやレイアウトなどの柔軟な条件付けが可能に（Stable Diffusionの基盤）。  
  * Inpainting, Super-resolution, Text-to-ImageでSOTAまたはそれに匹敵する性能を達成。

## **2\. 背景と課題**

### **ピクセル空間での拡散モデル (Pixel-Space DMs) の課題**

前回触れたDDPMなどは、高解像度画像の生成において優れた性能を示すが、以下の課題がある。

1. **計算コスト:** RGB画像（高次元）の各ピクセルに対してデノイズ処理を行うため、学習には数百GPU days、推論も低速。  
2. **知覚的冗長性:** 画像の細部（高周波成分）の学習に多くの計算リソースを消費しているが、これらは意味的な内容（Semantic）とは必ずしも直結しない。

### **解決へのアプローチ**

\*\*「知覚的な圧縮 (Perceptual Compression)」**と**「意味的な生成 (Semantic Generation)」\*\*のプロセスを分離する。

## **3\. 提案手法: Latent Diffusion Models (LDM)**

LDMは大きく**2つのステージ**に分かれる。

1. **Perceptual Compression (Autoencoder):**  
   * ピクセル空間 $x$ を知覚的に等価な低次元の潜在空間 $z$ に圧縮する。  
   * 高周波成分（ノイズに近い詳細）を取り除き、計算効率を上げる。  
2. **Latent Diffusion (DMs in Latent Space):**  
   * 圧縮された潜在空間 $z$ 上で拡散プロセス（ノイズ付加と除去）を行う。  
   * 空間的な次元が減るため、高解像度画像の生成も効率的に行える。

## **4\. Perceptual Image Compression (Stage 1\)**

学習済みAutoencoder ($\\mathcal{E}, \\mathcal{D}$) を利用する。

* **Encoder** $\\mathcal{E}$**:** 画像 $x \\in \\mathbb{R}^{H \\times W \\times 3}$ を潜在表現 $z \= \\mathcal{E}(x) \\in \\mathbb{R}^{h \\times w \\times c}$ にエンコード。  
  * ダウンサンプリング係数 $f \= H/h \= W/w$ (例: $f=4, 8, 16$)。  
* **Decoder** $\\mathcal{D}$**:** 潜在表現から画像を再構成 $\\tilde{x} \= \\mathcal{D}(z)$。  
* **正則化:** 潜在空間の分散を抑えるため、KL正則化（VAEライク）またはVQ正則化（VQGANライク）を使用。

**メリット:** このモデルは一度学習すれば、様々なタスクのDM学習に再利用可能。

## **5\. Latent Diffusion Models (Stage 2\)**

潜在空間 $z$ に対する拡散モデル。

* **Forward Process:** $z$ にガウシアンノイズを徐々に追加し、純粋なノイズにする（DDPMと同様）。  
* **Reverse Process:** ノイズから $z$ を復元するよう、Denoising U-Net $\\epsilon\_\\theta$ を学習。

### **目的関数 (Objective)**

ピクセル $x$ ではなく、潜在変数 $z$ (つまり $\\mathcal{E}(x)$) に対して最適化を行う。

[![][image1]](https://www.codecogs.com/eqnedit.php?latex=L_%7BLDM%7D%20%3A%3D%20%5Cmathbb%7BE%7D_%7B%5Cmathcal%7BE%7D\(x\)%2C%20%5Cepsilon%20%5Csim%20%5Cmathcal%7BN%7D\(0%2C1\)%2C%20t%7D%20%5Cleft%5B%20%7C%7C%20%5Cepsilon%20-%20%5Cepsilon_%5Ctheta\(z_t%2C%20t\)%20%7C%7C_2%5E2%20%5Cright%5D#0)

* $t$: タイムステップ  
* $z\_t$: ノイズが加わった時点 $t$ の潜在表現  
* $\\epsilon\_\\theta$: ノイズ予測ネットワーク (Time-conditional U-Net)

## **6\. Conditioning Mechanisms (条件付け)**

LDMの強力な特徴は、テキスト、画像、レイアウトなど多様な入力 $y$ で生成を制御できる点。

Cross-Attention の導入:  
U-Netの中間層にCross-Attention機構を組み込み、条件 $y$ を注入する。

1. ドメイン固有のエンコーダ $\\tau\_\\theta$ (例: BERT, CLIP) で $y$ を中間表現に変換。  
2. U-Netの特徴マップ $\\varphi\_i(z\_t)$ と Attention をとる。

[![][image2]](https://www.codecogs.com/eqnedit.php?latex=Attention\(Q%2C%20K%2C%20V\)%20%3D%20softmax%5Cleft\(%5Cfrac%7BQK%5ET%7D%7B%5Csqrt%7Bd%7D%7D%5Cright\)%20%5Ccdot%20V#0)[![][image3]](https://www.codecogs.com/eqnedit.php?latex=Q%20%3D%20W_Q%5E%7B\(i\)%7D%20%5Ccdot%20%5Cvarphi_i\(z_t\)%2C%20%5Cquad%20K%20%3D%20W_K%5E%7B\(i\)%7D%20%5Ccdot%20%5Ctau_%5Ctheta\(y\)%2C%20%5Cquad%20V%20%3D%20W_V%5E%7B\(i\)%7D%20%5Ccdot%20%5Ctau_%5Ctheta\(y\)#0)

## **7\. アーキテクチャ概略図**

*(論文 Figure 3 相当の概念図)*

1. **Pixel Space:** 入力画像 $x$ $\\xrightarrow{\\mathcal{E}}$ **Latent Space:** $z$  
2. **Diffusion Process:** $z \\xrightarrow{Noise} z\_T$  
3. **Denoising U-Net:** $z\_T \\xrightarrow{Denoise} z$  
   * ここに **Conditioning** (Text, Semantic Map等) が Cross-Attention で入る。  
4. **Pixel Space:** $z$ $\\xrightarrow{\\mathcal{D}}$ 出力画像 $\\tilde{x}$

## **8\. 実験結果: 知覚的圧縮のトレードオフ**

ダウンサンプリング係数 $f$ (圧縮率) が生成品質と効率にどう影響するか？

* $f$ **が小さい (**$1, 2$**):** ピクセル空間に近く、計算コスト削減効果が薄い。学習も遅い。  
* $f$ **が大きすぎる (**$32$**):** 情報が失われすぎ、画質（Fidality）が停滞する。  
* **最適なバランス:** $f \\in \\{4, 8, 16\\}$ が最も良いトレードオフ（画質・効率）を示した。

**結果:** LDM-4 や LDM-8 が、従来のPixel-based DM (LDM-1) よりも低いFID（高画質）と高いスループットを達成。

## **9\. アプリケーションと成果**

1. **Text-to-Image Synthesis:**  
   * LAION-400Mデータセットで学習。  
   * 1.45Bパラメータのモデルで、ARモデルやGANと同等以上の性能。  
   * ユーザー入力プロンプトに対して忠実で高解像度な画像を生成可能。  
2. **Inpainting:**  
   * 欠損部分の補完。高解像度でも整合性の取れた補完が可能。  
   * 畳み込み的なサンプリングにより、$512^2$ 以上の解像度にも対応。  
3. **Super-Resolution:**  
   * 低解像度画像を入力条件として連結 (Concatenate) して学習。  
   * SR3 (Pixel-based DM) に匹敵するFIDを達成しつつ、推論は高速。

## **10\. まとめ (Conclusion)**

* **Latent Diffusion Models (LDMs)** を提案。  
  * 画像生成プロセスを「知覚的圧縮」と「潜在拡散」に分離。  
  * ピクセル空間ではなく潜在空間で拡散モデルを学習することで、**計算コストを大幅に削減**しつつ**高解像度・高画質**を実現。  
* **Cross-Attention** による柔軟な条件付けメカニズムを導入し、Text-to-Imageなど多様なタスクでSOTA級の性能を達成。  
* 学習済みモデルとコードは公開されており、その後の画像生成AIブーム（Stable Diffusion等）の火付け役となった。

<!-- _class: lead -->

## **Q & A**

\\frac{2}{3}  