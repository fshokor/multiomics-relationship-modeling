# nb07 results review — donor 15078

Reviewed 2026-10-07 using the executed notebook and copied `associations.csv`,
`direct_gene_associations.csv`, and `coverage.csv` from `results/single_donor/protein/`.
The run completed without notebook errors and retained 29,688 CITE cells.
All correlations below are descriptive; sites are subdivisions of one donor, not
independent donor replication. Pooled adjusted Pearson controls RNA/ADT library
depth and site; within-site adjusted Pearson controls the two library depths.

## Inflammatory response: consistent measured-subset support

In 7,244 CD14 monocytes, 14 of 200 pathway genes have directly mapped ADTs (7%).
The strongest positive adjusted module associations are CD14, CD88 and CD54.
All three remain positive in every site, with varying magnitudes.

| RNA measurement versus ADT | All sites | Site 1 (734 cells) | Site 2 (2,958) | Site 3 (2,915) | Site 4 (637) |
|---|---:|---:|---:|---:|---:|
| Inflammatory module–CD14 | 0.279 | 0.488 | 0.228 | 0.213 | 0.364 |
| Inflammatory module–CD88 | 0.236 | 0.392 | 0.200 | 0.153 | 0.375 |
| Inflammatory module–CD54 | 0.171 | 0.285 | 0.111 | 0.180 | 0.265 |
| CD14 RNA–CD14 ADT | 0.168 | 0.399 | 0.118 | 0.164 | 0.323 |
| C5AR1 RNA–CD88 ADT | 0.249 | 0.510 | 0.224 | 0.160 | 0.452 |
| ICAM1 RNA–CD54 ADT | 0.038 | 0.143 | 0.004 | 0.027 | 0.086 |

Values are adjusted Pearson correlations. CD14 and CD88 therefore have both
module-level association and positive same-gene agreement across sites. CD54 has
consistent module-level association but weak same-gene agreement, especially in
sites 2 and 3. These observations support a measured subset of the RNA program;
they do not validate all 200 genes or establish translation dynamics.

SELL/CD62L illustrates why the two comparisons must remain separate. Its same-gene
adjusted correlation is positive (0.194 overall; 0.149–0.309 across sites), whereas
its inflammatory-module association is negative (-0.154 overall; -0.256 to -0.099
across sites). This is not a failure of same-gene agreement. Its module association
also depends on normalization: pooled uncentered-log ADT Spearman is +0.176,
compared with centered-log Spearman +0.073 and adjusted Pearson -0.154. Different
transformations and adjustment answer different questions; no causal interpretation.

Large pooled raw effects can be misleading. Inflammatory module–CD48 Pearson drops
from 0.556 to 0.008 after adjustment; adjusted site effects range -0.079 to 0.025.
CD86 is 0.001 after adjustment, and HLA-DR is -0.079 (negative in all four sites).
These phenotypic markers do not supply consistent positive adjusted support.
The joint adjustment does not isolate how much attenuation comes from site versus
depth, and may remove biological as well as technical variation.

## TNF/NF-kB: weaker, overlapping support

Six of 200 genes have directly mapped ADTs (3%). Module–CD54 is positive in every
site (overall 0.153; sites 0.080–0.275), as is module–CD44 (overall 0.110; sites
0.064–0.203). Same-gene CD44 agreement is weak (0.084 overall; sites 0.047–0.116),
and ICAM1/CD54 agreement is the same weak result above. CD86/HLA-DR provide no
consistent positive adjusted support. The two CD14 programs overlap and share
cells and markers; they are not independent confirmations.

## Interferon-alpha, E2F and Myc

- **CD16 interferon-alpha:** three of 97 genes measured (3.1%), 865 cells. Direct
  module associations are weak overall: CD47 -0.077, CD62L -0.035, CD124 -0.021.
  CD47 is negative in all three testable sites (-0.119, -0.006, -0.286), despite
  positive pooled raw Pearson (0.387). Same-gene CD47 agreement is weakly positive
  (0.065 overall). Site 1 has 45 cells, below the 50-cell threshold: its missing
  correlations are **untested**, never zero. CD86 and HLA-DR context is inconsistent.
  No consistent positive adjusted protein support is established by this panel.
- **Lymphoid progenitor E2F:** only TFRC/CD71 is measured (1/200; 0.5%), 449 cells.
  Module–CD71 changes from raw Pearson 0.308 to adjusted -0.148. Adjusted site
  effects are +0.148, -0.205, -0.001 and -0.229. TFRC RNA–CD71 ADT is weakly
  positive overall (0.095), but site effects also vary (-0.096 to 0.207). This is
  insufficient evidence for pathway-wide protein support or biological decoupling.
- **MK/E Myc:** zero of 200 genes have direct ADT coverage. Protein support is
  unmeasured; no negative biological conclusion is justified.

## Implications for nb08

Prioritize the CD14 inflammatory program as the clearest measured-subset protein
result, retaining CD14 and C5AR1/CD88 as the strongest same-gene examples. Include
CD54 as module-level support with weak same-gene agreement. Retain TNF/NF-kB as
weaker overlapping evidence; label interferon/E2F as lacking consistent positive
adjusted support and Myc as unmeasured.

These results can accompany the previously reviewed RNA and ATAC findings in a
descriptive multilayer summary. They do not resolve site-specific **cross-capture
RNA pathway** replication or the seven unavailable nb05 enrichment scores, which
remain separate checks. RNA module scores include the matched gene: a future
leave-one-gene-out sensitivity analysis would separate broader-program association
from that gene's own contribution. Limited antibody coverage, relative ADT
normalization, no background correction and single-donor sampling remain limitations.
