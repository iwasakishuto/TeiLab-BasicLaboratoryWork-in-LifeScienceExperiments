#coding: utf-8
r"""
**Gene set enrichment analysis (GSEA)** (or **functional enrichment analysis**) is a method to determine whether an a priori defined set of genes or proteins (i.e. classes of genes or proteins that are over-represented in a large set of genes or proteins) show statistically significant differences, and may have an association with disease phenotypes.

.. warning::

    Enrichment Analysis is a method that **relies on a database** created based on the information published in a paper. Therefore, it is a technique for giving an interpretation to the experimental results rather than making a new discovery. Also, it is necessary to make a strong assumption that **"all past treatises are correct"**, so be careful about that.

############
Instructions
############

    Gene set enrichment analysis uses a priori gene sets that have been grouped together by their involvement in the same biological pathway, or by proximal location on a chromosome. A database of these predefined sets can be found at the Molecular signatures database (`MSigDB <https://www.gsea-msigdb.org/gsea/msigdb/>`_). In GSEA, DNA microarrays, or now RNA-Seq, are still performed and compared between two cell categories, but instead of focusing on individual genes in a long list, the focus is put on a gene set. Researchers analyze whether the majority of genes in the set fall in the extremes of this list: the top and bottom of the list correspond to the largest differences in expression between the two cell types. If the gene set falls at either the top (over-expressed) or bottom (under-expressed), it is thought to be related to the phenotypic differences.

    In the method that is typically referred to as standard GSEA, there are three steps involved in the analytical process. The general steps are summarized below:

    1. Calculate the enrichment score (ES) that represents the amount to which the genes in the set are over-represented at either the top or bottom of the list. This score is a Kolmogorov–Smirnov-like statistic.
    2. Estimate the statistical significance of the ES. This calculation is done by a phenotypic-based permutation test in order to produce a null distribution for the ES. The P value is determined by comparison to the null distribution.
        - Calculating significance this way tests for the dependence of the gene set on the diagnostic/phenotypic labels.
    3. Adjust for multiple hypothesis testing for when a large number of gene sets are being analyzed at one time. The enrichment scores for each set are normalized and a false discovery rate is calculated.


**************************************
Gene Ontology (GO) Enrichment Analysis
**************************************

Most genes have a keyword (GO term) called **Gene Ontology (GO)**. The GO term focuses on the biological process of the gene, the cellular components, and the molecular function. By performing GO enrichment analysis after identifying differentially expressed genes (DEGs), it will lead to elucidation of molecular functions and cell localization peculiar to DEGs.

****************
Pathway Analysis
****************

A pathway is a biological process or pathway that is composed of the interaction of multiple ecological molecules and controls the biological phenomena necessary to maintain ecological activity. It contains not only metabolic pathways but also information such as signal transduction systems, protein-protein interactions, and gene regulatory relationships (including enzymatic reactions, transcription, phosphorylation, and binding).

By performing **Pathway Analysis** after identifying DEGs, it will lead to elucidation of pathways peculiar to DEGs.

.. _target to over-representation analysis approaches:

Over-Representation Analysis Approaches
---------------------------------------

Identify the DEGs using FDR etc. and, classify into a following :math:`2\times2` matrix. Then, test them using  geometric distribution, or chi-square.

+--------+--------+--------+
|        |     Pathway     |
+========+========+========+
|        | target | others |
+--------+--------+--------+
| DEFs   |   a    |   b    |
+--------+--------+--------+
| others |   c    |   d    |
+--------+--------+--------+

.. warning::

    - Treats each gene independently.
    - Treats each pathway independently
    - Treats all genes equally regardless of the expression level.

.. _target to functional class scoring approaches:

Functional Class Scoring Approaches
-----------------------------------

Create a :math:`2\times2` matrix as in :ref:`Over-Representation Analysis Approaches <target to over-representation analysis approaches>`, but add a ranking of the statistics (p-value, fold-change, etc.) obtained as a result of the expression level comparison to weight each gene.

.. warning::

    - Treats each pathway independently
    - Statistics are ranked, not used as they are.

.. _target to pathway topology based approaches:

Pathway Topology Based Approaches
---------------------------------

This approach **directly** uses the topology of the pathway and the statistics obtained from each gene in :ref:`Functional Class Scoring Approaches <target to functional class scoring approaches>`

.. _target to impact factor analytic approaches:

Impact Factor Analytic Approaches
---------------------------------

Integrate the following information into :ref:`Pathway Topology Based Approachess <target to pathway topology based approaches>`

- the structure of the entire pathway
- important regulators
- gene expression level
- gene localization

##############
Python Objects
##############

Several tools have been developed to do GSEA in python.

- `GSEApy <https://gseapy.readthedocs.io/en/latest/introduction.html>`_ : Python implementation for GSEA and wrapper for Enrichr.

    .. code-block:: python

        >>> import gseapy
        >>> gl = [
        ...     'SCARA3', 'LOC100044683', 'CMBL', 'CLIC6', 'IL13RA1', 'TACSTD2', 'DKKL1', 'CSF1',
        ...     'SYNPO2L', 'TINAGL1', 'PTX3', 'BGN', 'HERC1', 'EFNA1', 'CIB2', 'PMP22', 'TMEM173'
        >>> ]
        >>> gseapy.enrichr(gene_list=gl, description='pathway', gene_sets='KEGG_2016', outdir='test')
"""