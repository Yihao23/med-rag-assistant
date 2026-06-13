# M10:SBOM + Benchmark 报告

生成日期:2026-06-12
环境:Linux,16 核 CPU,15GB 内存,无 GPU(全部离线配置:echo LLM + hashing embedder)

## 1. SBOM(软件物料清单)

工具:syft 1.45.1,格式 CycloneDX JSON,文件在 `reports/sbom/`。

| 对象 | 文件 | 组件数 | 构成 |
|---|---|---|---|
| Docker 镜像 `medrag-api:dev` | `image.cdx.json` | 2907 | deb 87、pypi 49、其余为文件级条目 |
| 仓库源码(三语言生态) | `repo.cdx.json` | 425 | cargo 170、pypi 64、npm 4 等 |

再生成:`syft medrag-api:dev -o cyclonedx-json=reports/sbom/image.cdx.json`

## 2. 漏洞扫描

工具:grype 0.114.0(基于上述 SBOM)。

| 对象 | Critical | High | Medium | Low | 备注 |
|---|---|---|---|---|---|
| 镜像 | 0 | 0 | 0 | 0 | 仅 Debian 基础包的 Negligible 级 CVE(无修复版,风险 <0.1%) |
| 仓库 | 0 | 0 | 2 | 0 | 均为 pip 26.0.1 自身,升级 pip>=26.1 即修 |

再扫描:`grype sbom:reports/sbom/image.cdx.json`

## 3. 检索质量基准(eval)

评估集:`data/eval/medical_qa.jsonl`(6 条),离线配置,指标为 hit@k 与关键词召回。

| retrieval | k=1 | k=2 | k=4 |
|---|---|---|---|
| dense(纯向量) | 0.50 | 0.83 | 1.00 |
| **hybrid(向量+BM25/RRF)** | **0.67** | **1.00** | 1.00 |
| graph(实体图谱) | 0.50 | 0.50 | 0.83 |

结论:
- **hybrid 全面占优**:k=2 即达 1.00(dense 需要 k=4),BM25 的精确术语匹配补上了向量检索的盲区。
- graph 落后是已知现象:默认 `KeywordEntityExtractor` 对中文长句只能切二元组,实体质量粗糙;
  其差异化能力(多跳召回)在本评估集不体现,需 `LLMEntityExtractor` 才能发挥(见 M6 备注)。
- 注意:hashing embedder 是无语义的测试实现,绝对数值仅供配置间相对比较。

再生成:`python -m medrag.eval --llm echo --embedder hashing --retrieval hybrid --k 2`

## 4. 服务延迟基准

负载:`POST /ask`(echo LLM,k=4),100 次请求 + 10 次预热,单客户端串行。

| 路径 | P50 | P95 | max |
|---|---|---|---|
| 直连 Python FastAPI :8000 | 1.7 ms | 4.3 ms | 5.1 ms |
| 经 Rust 网关 :3001 | 3.0 ms | 4.5 ms | 5.4 ms |

结论:**网关代理开销 P50 仅 ~1.3ms**,对真实场景可忽略——接真实 LLM 后生成耗时以秒计,
网关开销占比 <0.1%。横切能力(日志/超时/健康聚合/静态托管)的代价极低。

注意:echo LLM 不调用真实模型,本基准测的是**框架与网络路径开销**,不含模型推理时间。

## 5. 已知局限

- 评估集仅 6 条,统计意义有限——扩充评估集是后续工作。
- 单客户端串行压测,未测并发吞吐(QPS)与资源占用曲线。
- SBOM 未接入 CI(每次构建自动生成),当前为手动快照。
