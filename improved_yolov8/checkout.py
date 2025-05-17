import torch, numpy as np, tempfile
from pathlib import Path
from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils.metrics import DetMetrics
from model import ImprovedYoloV8s
from train import CustomDetectionLoss, compute_loss, export_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = ImprovedYoloV8s().to(device).eval()

# --- fake batch ---------------------------------------------------------
B, H, W = 2, 640, 640
imgs = torch.randn(B, 3, H, W, device=device)
targets = torch.tensor([[0, 1, .5, .5, .2, .3],
                        [1, 2, .4, .4, .25, .25]], device=device)

# --- 1. forward / backward ---------------------------------------------
loss_fn = CustomDetectionLoss(model)
preds   = model(imgs)
loss, _ = compute_loss(preds, targets, loss_fn, {})
loss.backward()
print(f'✅ forward/backward ok  (loss={loss.item():.3f})')

# --- 2. NMS + DetMetrics -----------------------------------------------
preds_cat = torch.cat([p.reshape(p.size(0), -1, p.size(-1))
                       for p in preds], 1)
dets = non_max_suppression(preds_cat, 0.001, 0.6)

tp, conf, pred_cls = [], [], []
n_iou = 10                                # Ultralytics 默认 iou thresholds
for det in dets:
    if det is None or not len(det):
        continue
    tp.append(torch.zeros((len(det), n_iou), dtype=torch.uint8))
    conf.append(det[:, 4].cpu())
    pred_cls.append(det[:, 5].cpu())

tp         = torch.cat(tp).numpy() if tp else np.zeros((0, n_iou), dtype=np.uint8)
tp_m       = np.zeros_like(tp)             # 分割/关键点占位，全 0
conf       = torch.cat(conf).numpy() if conf else np.array([])
pred_cls   = torch.cat(pred_cls).numpy() if pred_cls else np.array([])
target_cls = targets[:, 1].cpu().numpy()

metrics = DetMetrics()
metrics.process(tp, tp_m, conf, pred_cls, target_cls)
print('✅ metrics:', metrics.results_dict)

# --- 3. ONNX export -----------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    ckpt = Path(tmp) / 'best.pt'
    torch.save({'model': model.state_dict()}, ckpt)
    ok = export_model(model, 640, ckpt, Path(tmp) / 'model.onnx')
    print('✅ onnx export', 'passed' if ok else 'failed')
