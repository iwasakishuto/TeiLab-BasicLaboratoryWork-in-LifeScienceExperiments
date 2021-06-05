#coding: utf-8
"""

Differential gene expression can be an outcome of true biological variability or experimental artifacts. Normalization techniques have been used to minimize the effect of experimental artifacts on differential gene expression analysis.

###############################
Robust Multichip Analysis (RMA)
###############################

In microarray analysis, many algorithms have been proposed, but the most widely used one is :fa:`file-pdf-o` `Robust Multichip Analysis (RMA) <https://academic.oup.com/biostatistics/article/4/2/249/245074>`_ , where the signal value of each spot ( ``RawData`` ) is processed and normalized according to the following flow.

.. graphviz::

    digraph RAMPreprocessingGraph {

        graph [
            charset   = "UTF-8";
            label     = "Preprocessing (RMA)",
            labelloc  = "t",
            labeljust = "c",
            bgcolor   = "#1f441e",
            fontcolor = "white",
            fontsize  = 18,
            style     = "filled",
            rankdir   = TB,
            margin    = 0.2,
            ranksep   = 1.0,
            nodesep   = 0.9,
            layout    = dot,
            compound = true,
        ];

        node [
            style     = "solid,filled",
            fontsize  = 16,
            fontcolor = 6,
            fontname  = "Migu 1M",
            color     = "#cee6b4",
            fillcolor = "#9ecca4",
            fixedsize = false,
            margin    = "0.2,0.1",
        ];

        edge [
            style         = solid,
            fontsize      = 14,
            fontcolor     = white,
            fontname      = "Migu 1M",
            color         = white,
            labelfloat    = true,
            labeldistance = 2.5,
            labelangle    = 70
        ];

        RawData [shape=doublecircle margin="0" fillcolor="#29bb89" fontcolor="#be0000" color="#29bb89"];
        Ave   [shape=circle margin="0"  fontcolor="#233e8b" fillcolor="#b6c9f0" color="#233e8b" label="ave"];
        RawData -> Ave;
        Ave -> gMeanSignal;

        subgraph cluster_0 {
            label     = "Background Subtraction";
            labelloc  = "t";
            labeljust = "l";
            fillcolor = "#89898989";
            fontcolor = "#ffd56b";
            margin    = 20;

            gMeanSignal                [shape=box fontname="monaco" fontcolor="#e74c3c" fillcolor="white" color="#e1e4e5"];
            gBGUsed                    [shape=box fontname="monaco" fontcolor="#e74c3c" fillcolor="white" color="#e1e4e5"];
            gBGSubSignal               [shape=box fontname="monaco" fontcolor="#e74c3c" fillcolor="white" color="#e1e4e5"];
            rBGSubSignal               [shape=box fontname="monaco" fontcolor="#e74c3c" fillcolor="white" color="#e1e4e5"];
            gDyeNormSignal             [shape=box fontname="monaco" fontcolor="#e74c3c" fillcolor="white" color="#e1e4e5"];
            rDyeNormSignal             [shape=box fontname="monaco" fontcolor="#e74c3c" fillcolor="white" color="#e1e4e5"];
            gProcessedSignal_A1        [shape=box fontname="monaco" fontcolor="#e74c3c" fillcolor="white" color="#e1e4e5" margin="0.35,0.1"];
            Minus [shape=circle margin="0" fontcolor="#233e8b" fillcolor="#b6c9f0" color="#233e8b" label="-"];
            BackgroundSubtraction     [shape=record  label="Background Subtraction|DFA"];
            DyeNormalization          [shape=diamond margin=0.05];
            SurrogateVariableAnalysis [shape=box     margin=0.1 label="SVA"];

            gMeanSignal -> BackgroundSubtraction;
            BackgroundSubtraction -> gBGUsed;
            gMeanSignal -> Minus;
            gBGUsed -> Minus;
            Minus -> gBGSubSignal;
            gBGSubSignal -> DyeNormalization;
            rBGSubSignal -> DyeNormalization;
            DyeNormalization -> gDyeNormSignal;
            DyeNormalization -> rDyeNormSignal;
            gDyeNormSignal -> SurrogateVariableAnalysis;
            SurrogateVariableAnalysis -> gProcessedSignal_A1;
        };

        subgraph cluster_1 {
            label     = "Normalization Between Samples";
            labelloc  = "t";
            labeljust = "l";
            fillcolor = "#89898989";
            fontcolor = "#ffd56b";
            margin    = 20;

            gProcessedSignal_B         [shape=box fontname="monaco" fontcolor="#e74c3c" fillcolor="white" color="#e1e4e5" margin="0.35,0.1"];
            gProcessedSignal_A2        [shape=box fontname="monaco" fontcolor="#e74c3c" fillcolor="white" color="#e1e4e5" margin="0.35,0.1"];
            gProcessedSignal_A1_normed [shape=box fontname="monaco" fontcolor="#e74c3c" fillcolor="white" color="#e1e4e5" margin="0.4,0.1"];
            gProcessedSignal_B_normed  [shape=box fontname="monaco" fontcolor="#e74c3c" fillcolor="white" color="#e1e4e5" margin="0.4,0.1"];
            gProcessedSignal_A2_normed [shape=box fontname="monaco" fontcolor="#e74c3c" fillcolor="white" color="#e1e4e5" margin="0.4,0.1"];
            QuantileNormalization     [shape=diamond margin=0.05];

            gProcessedSignal_A1 -> QuantileNormalization;
            gProcessedSignal_B -> QuantileNormalization;
            gProcessedSignal_A2 -> QuantileNormalization;
            QuantileNormalization -> gProcessedSignal_A1_normed;
            QuantileNormalization -> gProcessedSignal_B_normed;
            QuantileNormalization -> gProcessedSignal_A2_normed;
        };

        subgraph cluster_2 {
            label     = "Summarization";
            labelloc  = "t";
            labeljust = "l";
            fillcolor = "#89898989";
            fontcolor = "#ffd56b";
            margin    = 20;

            gProcessedSignal_A_normed  [shape=box fontname="monaco" fontcolor="#e74c3c" fillcolor="white" color="#e1e4e5" margin="0.4,0.1"];
            Summarization              [shape=diamond margin=0.05];

            gProcessedSignal_A1_normed -> Summarization;
            gProcessedSignal_A2_normed -> Summarization;
            Summarization -> gProcessedSignal_A_normed;
        };

    }

*************************
1. Background Subtraction
*************************

********************************
2. Normalization Between Samples
********************************

****************
3. Summarization
****************

https://github.com/scipy/scipy/blob/v1.6.3/scipy/signal/signaltools.py#L3384-L3467

"""