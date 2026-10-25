# 价目表（手填）

这里的 JSON 是**手填**的，不是调用供应商接口取的。四份表：

| 文件 | 模型 | 档位 |
| --- | --- | --- |
| `deepseek-flash.peak.json` | deepseek-flash | 高峰 |
| `deepseek-flash.offpeak.json` | deepseek-flash | 空闲 |
| `deepseek-v4-pro.peak.json` | deepseek-v4-pro | 高峰 |
| `deepseek-v4-pro.offpeak.json` | deepseek-v4-pro | 空闲 |

数字来源：<https://api-docs.deepseek.com/zh-cn/quick_start/pricing>（2026-10-25 抓取，
CNY / 每百万 token）。价格会变，**用之前先核对 `effective_date` 与出处**，过期就重抓再改数。

`config/prices.example.json` 只是字段模板，里面的数字不是任何供应商的真实报价。

## 字段

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `currency` | 是 | 如 `CNY`。`--max-cost` 的币种必须与它一致 |
| `unit_tokens` | 是 | 1 / 1000 / 1000000，按供应商报价口径；猜错差 1000 倍 |
| `input` | 是 | 缓存未命中的输入价 |
| `cached_input` | 否 | 缓存命中的输入价；省略时按 `input` 计并在结果里标注 |
| `output` | 是 | 输出价 |
| `source` | 是 | 价目页链接。没有出处的价格无法复核 |
| `effective_date` | 否 | 抓取或生效日期 |
| `model` | 否（建议） | 与实际调用的模型比对；不一致会标 `model_match=false` |
| `tier` | 否（建议） | `peak` / `off_peak` |
| `note` | 否 | 一句话说明 |

## 为什么峰谷要拆两份

供应商按时间段计价，一张表只能表示一档。取平均会让每次估算都偏；只写高峰会让空闲
时段高估一倍。所以拆开，按实际运行时间选一份。

## 用法

```bash
python -m mtsa.cli run --mode agent --scene weather \
    --input fixtures/weather_v1.csv --question "分析温度的趋势与异常" \
    --price-table prices/deepseek-flash.peak.json --max-cost 0.05
```

`--max-cost` 超支**只标记、不中止**，见 `doc/接口与字段.md` 的「成本语义（E2）」。
