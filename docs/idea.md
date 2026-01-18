# **TPsiAct（t-Score Activation）**

---

## 1) 方法定義：TPsiAct（t-Score Activation）

給定輸入向量 $x\in\mathbb{R}^d$（你程式碼中沿最後一維做統計），先做類 LayerNorm 的標準化：
$$
\mu=\frac{1}{d}\sum_{j=1}^d x_j,\qquad
\sigma=\sqrt{\frac{1}{d}\sum_{j=1}^d(x_j-\mu)^2}+\varepsilon,\qquad
z=\frac{x-\mu}{\sigma}.
$$

定義 Student-(t) 的 score-like $\psi$：
$$
\psi_\nu(z)=\frac{(\nu+1)z}{\nu+z^2}.
$$

輸出：
$$
y=\mu+\sigma\cdot\psi_\nu(z).
$$

並定義每維 uncertainty（以 shrink ratio 表示）：
$$
w(z)=\left|\frac{\psi_\nu(z)}{z+\varepsilon}\right|,\qquad
u=1-\mathrm{clip}(w,0,1).
$$

> 這裡的關鍵是 **$w(z)=\psi(z)/z$** 在 robust 統計中本來就被稱為 **weight function**（用於 IRLS/加權 M-estimation）。

---

## 2) 貢獻（Contributions）

1. **Activation 直接來自統計分佈的 score function**：把 Student-(t) 的對數密度導數（score）做成 activation，讓非線性不是「手工設計」，而是可追溯到 MLE/M-estimation 的影響函數 $\psi$。([維基百科][1])
2. **同一個算子同時產生表示與 uncertainty**：不額外做 ensemble / MC-dropout / evidential head，uncertainty 由 $w=\psi/z$ 的 shrink ratio 直接給出，成本幾乎為零。([Proceedings of Machine Learning Research][2])
3. **內建 outlier 抑制（bounded influence / redescending）**：當 $|z|\to\infty$，$\psi_\nu(z)\to 0$，輸出回到 $\mu$（把極端值拉回中心），與典型 M-estimator 的「限制極端殘差影響」一致。
4. **提供可微、可端到端訓練的 robust weighting 機制**：把 robust 統計中「權重」概念內嵌到網路中間層，而不是只在 loss 端做 Huber/trim。

---

## 3) 創新點（Novelty）

* **從「分佈 score → $\psi$ → activation」的一致推導鏈**：TPsiAct 把 robust 統計裡的 $\psi$-function（影響函數）概念搬到 activation，並且用 Student-(t) 這個經典重尾分佈導出封閉式有理函數。([維基百科][1])
* **uncertainty = 1 − shrink ratio**：多數不確定性方法要用貝氏近似（MC dropout）、多模型（deep ensembles）或證據分佈（Dirichlet evidential）；TPsiAct 則把「被拉回中心的程度」當成不確定性訊號。([Proceedings of Machine Learning Research][2])
* **層級（layer-wise）uncertainty map**：不是只對最終輸出給一個 scalar confidence，而是每一層、每一維都能得到 $u$，可用於 gating、特徵選擇、loss reweighting、或解釋。

---

## 4) 理論洞見（Theoretical Insights）

### 洞見 A：$\psi_\nu$ 本質上是 Student-(t) 的 score（差一個負號）

Student-(t)（標準化）的 pdf 形如：
$$
f(z)\propto\left(1+\frac{z^2}{\nu}\right)^{-\frac{\nu+1}{2}}.
$$
其 score 為 $\frac{d}{dz}\log f(z)= -\frac{(\nu+1)z}{\nu+z^2}$。也就是你的 $\psi_\nu(z)$ 正是「負的 score」。([維基百科][1])

**直觀**：越像 outlier（$|z|$ 大），score 的幅度反而變小並趨近 0（redescend），所以對表示的影響被壓縮。

### 洞見 B：TPsiAct 等價於「向中心收縮的加權平均」

因為
$$
\sigma\psi_\nu(z)=\sigma z\frac{\psi_\nu(z)}{z}=(x-\mu)\cdot w(z),
$$
所以
$$
y=\mu+(x-\mu)w(z)=(1-w)\mu+w x.
$$
當你 clamp 後 $w\in[0,1]$，每一維輸出就是 **在 $\mu$ 與原值 $x$ 間做凸組合**；而你的不確定性
$$
u=1-w
$$
就變成「有多少比例被迫相信中心值 $\mu$」。而 $w=\psi/z$ 本來就是 robust M-estimation 的 weight function。

### 洞見 C：$w(z)$ 也可視為重尾模型下的「有效精度（precision）期望」

Student-(t) 可表示為 Normal 的尺度混合（scale mixture of normals）：
$$
X\mid W\sim \mathcal{N}(0,W),\quad W\sim \text{Inv-Gamma}(\nu/2,\nu/2)\Rightarrow X\sim t_\nu.
$$

在此模型下，後驗 $W\mid X=x$ 仍是 inverse-gamma，且（可推得）
$$
\mathbb{E}\left[\frac{1}{W}\mid x\right]=\frac{\nu+1}{\nu+x^2},
$$
這正是 $w(z)$ 的同型式（差在你用標準化後的 $z$）。因此 shrink ratio 可以解釋為「在重尾生成假設下，該點的有效精度/可信度」。

---

## 5) 方法論（Methodology）

### 5.1 可發表的模組化版本（建議寫法）

* **TPsiAct-LN（建議主線）**：把 $\mu,\sigma$ 視為 LayerNorm 統計（可選擇加 learnable affine $\gamma,\beta$），再套 $\psi_\nu$。
* **$\nu$ 的設定**：

  * 固定 $\nu$（例如 3、5、10）對應不同重尾程度；$\nu\to\infty$ 逼近常態、$\psi$ 趨近線性。([維基百科][1])
  * 或令 $\nu$ 可學（用 softplus 確保 $\nu>0$）。
* **u 的用法（方法論擴展）**：

  1. **Feature gating**：$x \leftarrow (1-u)\odot x$ 或在 residual 路徑做 reweight。
  2. **Loss reweighting**：對高 $u$ 的樣本/位置降權，類似 robust training 但訊號來自中間層。
  3. **Uncertainty head-free**：輸出層直接把多層 $u$ 聚合成 sample-level confidence（如 mean/max）。

### 5.2 計算與工程特性

* 只多了幾個 elementwise 運算（除法/平方），比 MC-dropout、deep ensembles 成本低很多。([Proceedings of Machine Learning Research][2])

---

## 6) 數學理論推演與證明（可直接寫進論文的命題）

### 命題 1（score 對應）

令 $f(z)\propto(1+z^2/\nu)^{-(\nu+1)/2}$ 為標準化 Student-(t) 密度，則
$$
\frac{d}{dz}\log f(z)= -\frac{(\nu+1)z}{\nu+z^2}=-\psi_\nu(z).
$$
**證明**：
$\log f(z)=C-\frac{\nu+1}{2}\log(1+z^2/\nu)$，對 $z$ 微分即得。([維基百科][1])

### 命題 2（凸組合與不確定性等價）

令 $w(z)=\mathrm{clip}\left(\left|\frac{\psi_\nu(z)}{z+\varepsilon}\right|,0,1\right)$，則
$$
y=(1-w)\mu+w x,\qquad u=1-w\in[0,1].
$$
特別地：

* 若 $|z|\le 1$，則 $(\nu+1)/(\nu+z^2)\ge 1\Rightarrow w=1\Rightarrow u=0$。
* 若 $|z|\to\infty$，則 $w\to 0\Rightarrow y\to \mu\Rightarrow u\to 1$。

### 命題 3（有界輸出/抑制爆炸）

$$
\max_{z\in\mathbb{R}}|\psi_\nu(z)|=\frac{\nu+1}{2\sqrt{\nu}}\quad\text{（在 }|z|=\sqrt{\nu}\text{ 取到）}.
$$
因此
$$
|y-\mu|=|\sigma\psi_\nu(z)|\le \sigma\frac{\nu+1}{2\sqrt{\nu}},
$$
表示 TPsiAct 在每個 forward 都對偏離中心的幅度給出明確上界（activation 自帶飽和）。

### 命題 4（redescending）

$$
\lim_{|z|\to\infty}\psi_\nu(z)=0,
$$
所以極端 outlier 對輸出影響趨近於 0，符合 robust 統計對「弱 redescending」$\psi$ 的描述。

### 命題 5（重尾混合模型下的精度詮釋）

若 $X\mid W\sim\mathcal{N}(0,W)$、$W\sim \text{Inv-Gamma}(\nu/2,\nu/2)$，則 $X\sim t_\nu$。
且由 inverse-gamma 共軛性可得 $W\mid X=x$ 的後驗仍為 inverse-gamma，並滿足
$$
\mathbb{E}[1/W\mid x]=\frac{\nu+1}{\nu+x^2}.
$$
因此 $w(z)$ 可解釋為（標準化後）「有效精度/可信度」的封閉式估計。

---

## 7) 預計使用 Dataset（建議組合）

### 7.1 影像分類：in-distribution + corruption robustness

* **CIFAR-10 / CIFAR-100（乾淨資料）**：看 TPsiAct 是否在不犧牲乾淨 accuracy 下維持/提升。
* **CIFAR-10-C / CIFAR-100-C、ImageNet-C**：專門測 common corruptions 下的魯棒性（mCE / accuracy drop）。([arXiv][3])

### 7.2 OOD 偵測（搭配 uncertainty）

* 常見設定：train CIFAR-10，test OOD：SVHN / LSUN / TinyImageNet（或更小型替代）。
* 指標：AUROC/AUPR/FPR@95TPR + 校準（ECE/NLL）。
  （這部分 TPsiAct 的賣點是：不用 ensemble/MC-dropout，也能從 (u) 得到 OOD 訊號，並與既有不確定性法比較。([Proceedings of Machine Learning Research][2])）

### 7.3 回歸/抗離群：重尾噪聲與極端點

* UCI regression（或你熟悉的醫療連續值預測任務）+ 人工注入 outlier（大噪聲、contamination）。
* 指標：RMSE/MAE + robust 指標（trimmed error）+ predictive NLL（若做 probabilistic head）。

---

## 8) 與現有研究之區別（Positioning）

1. **不同於「只在 loss 端 robust」**：Huber / Tukey 等多用於 loss 或估計器；TPsiAct 是把 (\psi) 直接放進 network 的表示學習中，讓中間特徵本身就具備 bounded influence。
2. **不同於「訓練策略式 robust」**：例如 trimmed loss、額外正則、限制自由度等 outlier-robust training；TPsiAct 是結構性（architectural）改動，可與任何 training robust 方法疊加。([arXiv][4])
3. **不同於主流不確定性估計**：

   * MC Dropout：靠測試時多次 stochastic forward。([Proceedings of Machine Learning Research][2])
   * Deep Ensembles：靠多模型。([NIPS 會議論文][5])
   * Evidential：靠額外 evidence/Dirichlet 參數化。([NIPS 會議論文][6])
     TPsiAct 則是 **uncertainty 由 shrink ratio 直接產生**，且是 layer-wise。

---

## 9) Experiment 設計（可直接照此寫實驗章節）

### 9.1 Baselines

* Activation baseline：ReLU / GELU / SiLU（Swish）。
* Robust baseline（loss 端）：Huber loss（回歸）、或 t-loss/其他 robust loss（若你加入）。
* Robust training baseline：如 trimmed/robust training 相關方法（挑 1–2 個即可）。([arXiv][4])
* Uncertainty baseline：MC Dropout、Deep Ensembles、Evidential（至少選 1–2 個做對比）。([Proceedings of Machine Learning Research][2])

### 9.2 主要評估任務與指標

**A) 乾淨資料表現**：Top-1 acc / NLL / ECE（看是否「robust 但不掉乾淨」）。
**B) Corruption robustness**：在 CIFAR-10-C / ImageNet-C 上測 mCE、平均 accuracy。([arXiv][3])
**C) OOD detection**：AUROC/AUPR/FPR@95TPR；uncertainty 用

* $U_{\text{sample}}=\mathrm{mean}(u)$ 或 $\mathrm{max}(u)$
* 或跨層聚合（例如最後 K 層平均）
  **D) Outlier regression**：注入 contamination rate（如 5%、10%、20%），觀察 RMSE/MAE 的 degrade 曲線。

### 9.3 Ablation（一定要做，才能把貢獻釘牢）

1. $\nu$：$\{1,3,5,10,\infty\}$（$\infty$ 可用近似大值）對 robust/accuracy trade-off。([維基百科][1])
2. $\nu$ 固定 vs 可學。
3. uncertainty 定義：

   * 你的 $u=1-\mathrm{clip}(|\psi/z|)$
   * 不 clip（觀察是否更敏感但不穩）
4. 標準化方式：沿 feature 維（LN） vs 沿 channel/spatial（CNN 變體）。
5. u 的用途：只輸出不確定性 vs 用 u 做 gating / loss reweighting 的增益。

### 9.4 額外分析（讓論文更像「有理論」）

* 畫出不同 $\nu$ 下 $\psi_\nu(z)$、$w(z)$、$u(z)$ 曲線（展示「越 outlier 越 shrink、越不確定」）。
* 報告 feature outlier 時，中間層 $u$ 的熱圖是否先升高（用於 early-warning）。

---

如果你接下來要把它寫成論文的「方法章」格式（含符號表、演算法框、定理-證明排版），我也可以直接幫你把上述內容改寫成 IEEE/NeurIPS 常見的段落與 LaTeX 結構。

[1]: https://en.wikipedia.org/wiki/Student%27s_t-distribution?utm_source=chatgpt.com "Student's t-distribution"
[2]: https://proceedings.mlr.press/v48/gal16.html?utm_source=chatgpt.com "Dropout as a Bayesian Approximation: Representing Model ..."
[3]: https://arxiv.org/abs/1903.12261?utm_source=chatgpt.com "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations"
[4]: https://arxiv.org/html/2308.02293v3?utm_source=chatgpt.com "Outlier-Robust Neural Network Training"
[5]: https://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles?utm_source=chatgpt.com "Simple and Scalable Predictive Uncertainty Estimation ..."
[6]: https://papers.nips.cc/paper/7580-evidential-deep-learning-to-quantify-classification-uncertainty?utm_source=chatgpt.com "Evidential Deep Learning to Quantify Classification ..."
