"""
第二次作业：MLP 学习参数的实现与可视化
课程：人工智能概论 · 第二篇 第 15 节「人工神经元网络」（课件 66/171 页）

内容：用纯 Python 标准库实现 2-2-1 MLP（2 输入 → 2 个 sigmoid 隐藏单元 → 1 输出），
     在 XOR 四个样本上训练，记录每一步的参数轨迹，导出为 JSON 供网页可视化。

与课件 PyTorch 版本严格对应：
  - 同一组初始化参数
  - 同一损失（binary cross entropy with logits）
  - 同一优化器（Adam，lr = 0.03）
  - 同一训练轮数（300 epoch，全批量）
  - 同一分类阈值（0.5）

额外做了一次数值梯度检验，验证手写反向传播的正确性。

运行：python3 mlp_train.py
输出：mlp_trace.json
"""

import json
import math
import os

# ---------------------------------------------------------------- 数据

X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y = [0.0, 1.0, 1.0, 0.0]

N = len(X)


def sigmoid(z):
    """数值稳定的 sigmoid。"""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


# ---------------------------------------------------------------- 模型
# 参数命名与课件一致：
#   W[i][j]  = 输入 i 到隐藏单元 j 的权重
#   bh[j]    = 隐藏单元 j 的偏置
#   v[j]     = 隐藏单元 j 到输出的权重
#   bo       = 输出偏置

INIT = {
    "W": [[2.0, -2.0], [-2.0, 2.0]],
    "bh": [-1.0, -1.0],
    "v": [3.0, 3.0],
    "bo": -2.5,
}


def flatten(p):
    """把参数字典压成扁平列表，方便数值梯度检验。"""
    out = []
    for i in range(2):
        out.extend(p["W"][i])
    out.extend(p["bh"])
    out.extend(p["v"])
    out.append(p["bo"])
    return out


def unflatten(vec):
    return {
        "W": [[vec[0], vec[1]], [vec[2], vec[3]]],
        "bh": [vec[4], vec[5]],
        "v": [vec[6], vec[7]],
        "bo": vec[8],
    }


def forward(p, x):
    """返回 (隐藏层激活 h, 输出 logits, 输出概率)。"""
    h = [
        sigmoid(x[0] * p["W"][0][j] + x[1] * p["W"][1][j] + p["bh"][j])
        for j in range(2)
    ]
    z = h[0] * p["v"][0] + h[1] * p["v"][1] + p["bo"]
    return h, z, sigmoid(z)


def batch_loss_and_grad(p):
    """全批量 BCE-with-logits 的损失与解析梯度（手写反向传播）。"""
    grad = {"W": [[0.0, 0.0], [0.0, 0.0]], "bh": [0.0, 0.0], "v": [0.0, 0.0], "bo": 0.0}
    loss = 0.0

    for x, y in zip(X, Y):
        h, z, prob = forward(p, x)

        # ③ 损失：- [y·log σ(z) + (1-y)·log(1-σ(z))]，用 log-sum-exp 形式更稳
        #   softplus(z) - y·z 等价于上面的交叉熵
        loss += max(z, 0.0) + math.log1p(math.exp(-abs(z))) - y * z

        # ④ 反向传播：dL/dz = σ(z) - y（BCE + sigmoid 的经典结果）
        dz = (prob - y) / N

        # 输出层梯度
        for j in range(2):
            grad["v"][j] += h[j] * dz
        grad["bo"] += dz

        # 传到隐藏层：dL/dh = dz · v，再过 sigmoid' = h(1-h)
        for j in range(2):
            dh = dz * p["v"][j]
            dpre = dh * h[j] * (1.0 - h[j])
            grad["W"][0][j] += x[0] * dpre
            grad["W"][1][j] += x[1] * dpre
            grad["bh"][j] += dpre

    return loss / N, grad


def loss_only(vec):
    return batch_loss_and_grad(unflatten(vec))[0]


# ---------------------------------------------------------------- Adam

class Adam:
    """与 torch.optim.Adam 默认超参一致。"""

    def __init__(self, params, lr=0.03, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = [0.0] * len(params)
        self.v = [0.0] * len(params)
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        new = []
        b1, b2 = self.b1, self.b2
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            m_hat = self.m[i] / (1 - b1 ** self.t)
            v_hat = self.v[i] / (1 - b2 ** self.t)
            new.append(p - self.lr * m_hat / (math.sqrt(v_hat) + self.eps))
        return new


# ---------------------------------------------------------------- 训练

def train(epochs=300, lr=0.03, init=None, record=True):
    p = json.loads(json.dumps(init or INIT))
    opt = Adam(flatten(p), lr=lr)
    trace = []

    def snapshot(epoch, loss):
        h_all = [forward(p, x)[0] for x in X]
        prob = [forward(p, x)[2] for x in X]
        return {
            "e": epoch,
            "loss": round(loss, 6),
            "W": [[round(p["W"][0][0], 5), round(p["W"][0][1], 5)],
                  [round(p["W"][1][0], 5), round(p["W"][1][1], 5)]],
            "bh": [round(p["bh"][0], 5), round(p["bh"][1], 5)],
            "v": [round(p["v"][0], 5), round(p["v"][1], 5)],
            "bo": round(p["bo"], 5),
            "h": [[round(h[0], 5), round(h[1], 5)] for h in h_all],
            "p": [round(pr, 5) for pr in prob],
        }

    loss0, _ = batch_loss_and_grad(p)
    if record:
        trace.append(snapshot(0, loss0))

    for epoch in range(1, epochs + 1):
        loss, grad = batch_loss_and_grad(p)
        flat = flatten(p)
        gflat = []
        for i in range(2):
            gflat.extend(grad["W"][i])
        gflat.extend(grad["bh"])
        gflat.extend(grad["v"])
        gflat.append(grad["bo"])

        flat = opt.step(flat, gflat)
        p = unflatten(flat)

        if record:
            trace.append(snapshot(epoch, loss))

    return p, trace


# ---------------------------------------------------------------- 梯度检验

def grad_check(p, eps=1e-6):
    """中心差分 vs 解析梯度，返回每个分量的相对误差。"""
    _, grad = batch_loss_and_grad(p)
    analytic = []
    for i in range(2):
        analytic.extend(grad["W"][i])
    analytic.extend(grad["bh"])
    analytic.extend(grad["v"])
    analytic.append(grad["bo"])

    base = flatten(p)
    numeric = []
    for i in range(len(base)):
        up = base[:]
        dn = base[:]
        up[i] += eps
        dn[i] -= eps
        numeric.append((loss_only(up) - loss_only(dn)) / (2 * eps))

    errs = []
    for a, n in zip(analytic, numeric):
        denom = max(1e-8, abs(a) + abs(n))
        errs.append(abs(a - n) / denom)
    return analytic, numeric, errs


# ---------------------------------------------------------------- 主程序

def main():
    here = os.path.dirname(os.path.abspath(__file__))

    # 1) 梯度检验
    a, n, errs = grad_check(unflatten(flatten(INIT)))
    print("== 梯度检验（初始参数处，中心差分 eps=1e-6）==")
    names = ["W11", "W12", "W21", "W22", "bh1", "bh2", "v1", "v2", "bo"]
    for nm, av, nv, er in zip(names, a, n, errs):
        print(f"  {nm:>4}  解析 {av:+.8f}   数值 {nv:+.8f}   相对误差 {er:.2e}")
    print(f"  最大相对误差 = {max(errs):.3e}")

    # 2) 训练
    final, trace = train(epochs=300, lr=0.03)

    print("\n== 训练结果 ==")
    print(f"  初始 loss = {trace[0]['loss']:.6f}")
    print(f"  最终 loss = {trace[-1]['loss']:.6f}")
    for x, y, pr in zip(X, Y, trace[-1]["p"]):
        pred = 1 if pr >= 0.5 else 0
        print(f"  x={x}  target={int(y)}  prob={pr:.4f}  pred={pred}  {'OK' if pred==int(y) else 'WRONG'}")

    # 3) 导出
    payload = {
        "meta": {
            "task": "XOR",
            "arch": "2-2-1",
            "activation": "sigmoid",
            "loss": "BCEWithLogits",
            "optimizer": "Adam",
            "lr": 0.03,
            "epochs": 300,
            "batch": 4,
            "threshold": 0.5,
        },
        "data": {"X": X, "Y": Y},
        "init": INIT,
        "trace": trace,
        "gradcheck": {
            "names": names,
            "analytic": a,
            "numeric": n,
            "rel_err": errs,
            "max_rel_err": max(errs),
        },
        "final": {
            "W": final["W"],
            "bh": final["bh"],
            "v": final["v"],
            "bo": final["bo"],
            "loss": trace[-1]["loss"],
            "prob": trace[-1]["p"],
        },
    }

    out = os.path.join(here, "mlp_trace.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    size = os.path.getsize(out)
    print(f"\n已导出 {out}  ({size/1024:.1f} KB, {len(trace)} 帧)")

    # 同时产出一份 JS 包装，页面用 <script> 引入即可，避免 file:// 下的 CORS 限制
    js = os.path.join(here, "mlp_data.js")
    with open(js, "w", encoding="utf-8") as f:
        f.write("/* 由 mlp_train.py 自动生成，请勿手动修改 */\n")
        f.write("window.MLP_TRACE = ")
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        f.write(";\n")
    print(f"已导出 {js}  ({os.path.getsize(js)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
