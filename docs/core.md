# t-Score Activation（tΨAct）——用「score function」當 activation，uncertainty 直接是 shrink ratio

### 理論來源

Robust 統計裡 M-estimator 會用 (\psi)-function（影響函數）來限制 outlier 影響；而且很多 (\psi) 其實就是某些分佈的 score/導數形式（例如 t 分佈）([CRAN][3])。

### Layer 定義（你會覺得很像 “activation 但有理論”）

用 t 分佈的 score-like 形式（有名的簡單有理函數）：

* 標準化：(z=(x-\mu)/\sigma)
* (\psi_\nu(z)=\frac{(\nu+1)z}{\nu+z^2})
* 輸出：(y=\mu+\sigma,\psi_\nu(z))
* uncertainty（每維）：(u=1-\left|\frac{\psi_\nu(z)}{z+\epsilon}\right|)
  （越接近 1 代表越被 shrink，越不確定）

```python
class tPsiAct(nn.Module):
    def __init__(self, nu=5.0, eps=1e-5):
        super().__init__()
        self.nu = nu
        self.eps = eps

    def forward(self, x, return_u=False):
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True) + self.eps
        z = (x - mu) / sigma

        psi = (self.nu + 1.0) * z / (self.nu + z * z)
        y = mu + sigma * psi

        shrink = (psi.abs() / (z.abs() + self.eps)).clamp(0, 1)
        u = 1.0 - shrink
        return (y, u) if return_u else y
```