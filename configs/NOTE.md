## How to

```python
from retnet.configuration_retnet import RetNetConfig, load_config_from_json

size = '3b'
config = RetNetConfig.from_pretrained(f'configs/retnet-{size}')
# or
config = load_config_from_json(f'configs/retnet-{size}/config.json')
...
```

## Size Notes

|  name  |   size  |
|-------:|--------:|
| base   |  44.81M |
| medium | 286.38M |
| 300m   | 339.61M |
| xl     |    1.3B |
| 3b     |      3B |
| 7b     |      7B |
| 13b    |     13b |
| 65b    |     65b |