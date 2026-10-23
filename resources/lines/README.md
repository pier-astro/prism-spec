# Transition weights and their physical meaning

Ready-made line sets default to integrated flux. A weight multiplies its template
flux once; weights are not sum-normalized group fractions. Explicit amplitude
mode interprets them as peak multipliers. Generic LineSet imposes no such policy.

## Optical Fe II

[Ilić, Rakić & Popović (2023), ApJS 267, 19](https://doi.org/10.3847/1538-4365/acd783)
extend the optical model of [Kovačević, Popović & Dimitrijević (2010), ApJS 189, 15](https://arxiv.org/abs/1004.2212)
to 3700–11000 Å. The [published machine-readable Table 1](https://cdsarc.cds.unistra.fr/ftp/J/ApJS/267/19/table1.dat)
and [CDS column definitions](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/ApJS/267/19?format=html)
provide transition wavelengths, atomic levels and relative intensities.

All 283 positions in feII_model.ecsv match this table within wavelength rounding;
all weights agree within the published intensity rounding (maximum absolute
difference 0.000499). No numerical weights were changed in this migration.
The paper groups lines by lower energy level and uses an assumed excitation
temperature of 10^4 K. These ratios are a model assumption, not universal for Fe II.

The [2010 paper, Appendix C](https://arxiv.org/html/1004.2212v2#A3) derives line
intensity from transition probability times photon energy times emitter column.
This is a line-integrated energy intensity; it does not include the profile width
factor needed for a peak density. It supports applying the atomic ratios to
integrated fluxes. Do not multiply those published ratios by widths merely to
preserve an older numerical implementation.

The [FANTASY source](https://fantasy-agn.readthedocs.io/en/latest/_modules/fantasy_agn/models.html)
uses Int times a peak-normalized Gaussian in its Fe II routines. Explicit
normalization="amplitude" retains that computational convention for comparisons;
it is distinct from the atomic-intensity interpretation. Existing amplitude-based
examples select that mode explicitly, rather than silently changing their models.

## Other lists

This audit does not establish the source or physical meaning of every packaged
weight. The UV template requires its own bibliographic verification; the optical
2023 table does not validate it. Tables remain editable and weights are never
converted automatically. For atomic doublets, use integrated energy-flux ratios
under the appropriate optical-depth/excitation assumptions; for example, the
[O III] calculation in [Storey & Zeippen (2000)](https://academic.oup.com/mnras/article/312/4/813/1017382).
