# PyTorch zero optimization

- Research repo to deep dive into a couple of zero-order optimization techniques.
- We explore how directional sharpness can provide actionable optimization insights.

## Zero-order methods
- Memory-efficient zerothorder optimizer (MeZO). [Paper](https://arxiv.org/abs/2305.17333); [official repo](https://github.com/princeton-nlp/MeZO).
- Smart evolutionary strategy (SmartES).

## Run experiments

Define a config yaml file in the `config` directory; Then
```bash
python main.py --config $path --epochs $n
```
More options are available (see main file)

## How to use MeZO
Simply copy paste `mezo.py` in your repo and import the optimizer.

```python
from zeroptim.optim.mezo import MeZO

opt = MeZO(torch.optim.SGD(model.parameters(), lr=0.05), eps=1e-3) 
opt = MeZO(torch.optim.AdamW(model.parameters(), lr=0.005), eps=1e-3)  
```

## Disclaimer
Work in progress. May have bugs. Use at your discretion.
