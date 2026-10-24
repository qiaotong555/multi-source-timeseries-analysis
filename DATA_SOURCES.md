# 数据来源与署名

代码与文档的授权见 `LICENSE`。**本文件管的是第三方数据**：代码许可证不覆盖这些数据；
即使代码选择「保留所有权利」，下列署名要求仍然必须保留，删掉或改掉就是违反数据条款。

> 条款以各来源站点当前公示为准；本表整理于 2026-10-24，对外交付前请再核对一次（链接见 §3）。

## 1. 数据清单

| 编号 | 数据集 | 来源 | 时间范围 | 行数 | 许可证 |
| --- | --- | --- | --- | --- | --- |
| R01 | berlin_hourly_2024 | Open-Meteo Archive API | 2024-01-01 ~ 2024-12-31 | 8,784 | CC BY 4.0 |
| R02 | singapore_hourly_2024 | Open-Meteo Archive API | 2024-01-01 ~ 2024-12-31 | 8,784 | CC BY 4.0 |
| R03 | phoenix_daily_2015_2024 | Open-Meteo Archive API | 2015-01-01 ~ 2024-12-31 | 3,653 | CC BY 4.0 |
| R04 | reykjavik_hourly_2024 | Open-Meteo Archive API | 2024-01-01 ~ 2024-12-31 | 8,784 | CC BY 4.0 |
| R05 | three_cities_hourly_2024 | Open-Meteo Archive API | 2024-01-01 ~ 2024-12-31 | 26,352 | CC BY 4.0 |
| R06 | nyc_central_park_daily_2018_2024 | NOAA GHCN-Daily | 2018-01-01 ~ 2024-12-31 | 2,557 | 公有领域（美国政府作品） |
| R07 | melbourne_daily_2018_2024 | NOAA GHCN-Daily | 2018-01-01 ~ 2024-12-31 | 2,557 | 公有领域（美国政府作品） |
| R08 | berlin_tegel_hourly_2024 | DWD Open Data | 2024-01-01 ~ 2024-12-31 | 8,780 | CC BY 4.0 |
| R09 | munich_hourly_2024 | DWD Open Data | 2024-01-01 ~ 2024-12-31 | 8,784 | CC BY 4.0 |
| R10 | toronto_hourly_2024 | ECCC Bulk Data | 2024-01-01 ~ 2024-12-31 | 8,783 | Open Government Licence – Canada |
| R11 | la_paz_hourly_2024 | Open-Meteo Archive API | 2024-01-01 ~ 2024-12-31 | 8,784 | CC BY 4.0 |
| R12 | quito_hourly_2024 | Open-Meteo Archive API | 2024-01-01 ~ 2024-12-31 | 8,784 | CC BY 4.0 |
| R13 | addis_ababa_hourly_2024 | Open-Meteo Archive API | 2024-01-01 ~ 2024-12-31 | 8,784 | CC BY 4.0 |
| R14 | nairobi_hourly_2024 | Open-Meteo Archive API | 2024-01-01 ~ 2024-12-31 | 8,784 | CC BY 4.0 |
| R15 | cairo_hourly_2024 | Open-Meteo Archive API | 2024-01-01 ~ 2024-12-31 | 8,784 | CC BY 4.0 |
| R16 | ushuaia_hourly_2024 | Open-Meteo Archive API | 2024-01-01 ~ 2024-12-31 | 8,784 | CC BY 4.0 |
| R17 | nyc_asos_hourly_2018_2024 | NOAA/NWS ASOS via Iowa Environmental Mesonet | 2018-01-01 ~ 2024-12-30 | 61,098 | 公有领域 + IEM 引用要求 |

## 2. 必须原样保留的署名文本

- **Open-Meteo Archive API**（CC BY 4.0）
  `Contains information from Open-Meteo (https://open-meteo.com/), licensed under CC BY 4.0.`
- **NOAA GHCN-Daily**（公有领域（美国政府作品））
  `Data from NOAA NCEI GHCN-Daily (U.S. Government work, public domain).`
- **DWD Open Data**（CC BY 4.0）
  `Contains data from Deutscher Wetterdienst (DWD), licensed under CC BY 4.0.`
- **ECCC Bulk Data**（Open Government Licence – Canada）
  `Contains information licensed under the Open Government Licence – Canada.`
- **NOAA/NWS ASOS via Iowa Environmental Mesonet**（公有领域 + IEM 引用要求）
  `ASOS observations from NOAA/NWS via the Iowa Environmental Mesonet (IEM); please cite IEM.`

## 3. 条款链接

- Open-Meteo Archive API — CC BY 4.0 — https://open-meteo.com/en/license
- NOAA GHCN-Daily — 公有领域（美国政府作品） — https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
- DWD Open Data — CC BY 4.0 — https://www.dwd.de/EN/service/copyright/copyright_node.html
- ECCC Bulk Data — Open Government Licence – Canada — https://eccc-msc.github.io/open-data/license-canada-open-data/
- NOAA/NWS ASOS via Iowa Environmental Mesonet — 公有领域 + IEM 引用要求 — https://mesonet.agron.iastate.edu/

## 4. 复核与更新

`fixtures/real_weather/registry.json` 记录了每份数据的 `source`、`start`/`end`、
`rows` 与 `data_sha1`。增删数据后请同步更新本表，并重新核对条款是否仍然适用。
