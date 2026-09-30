# Experiment DAG

Generated from the Git registry. Solid edges distinguish controls and reused sources; dashed edges require a local comparison conclusion. Draft nodes are plans, not completed experiments.

```mermaid
flowchart TD
  n0["init-linear-s01 | completed"]
  n1["init-absolute-s01 | draft"]
  n2["init-logfc-s01 | draft"]
  n3["init-qc-s01 | draft"]
  n4["init-random-thin-s01 | draft"]
  n5["init-qc-weight-s01 | draft"]
  n6["init-context-balanced-s01 | draft"]
  n7["init-target-balanced-s01 | draft"]
  n8["init-meanvar-s01 | draft"]
  n9["init-set-s01 | draft"]
  n10["init-pca-s01 | completed"]
  n11["init-program-s01 | completed"]
  n12["init-random-program-s01 | completed"]
  n13["init-pretrained-s01 | draft"]
  n14["init-random-encoder-s01 | draft"]
  n15["init-scfoundation-s01 | draft"]
  n16["init-random-scfoundation-s01 | draft"]
  n17["init-go-target-s01 | draft"]
  n18["init-random-go-s01 | draft"]
  n19["init-mlp-s01 | draft"]
  n20["init-direction-loss-s01 | draft"]
  n21["init-context-film-s01 | draft"]
  n22["init-shared-context-s01 | draft"]
  n23["init-twostage-s01 | draft"]
  n24["init-gen-nb-s01 | draft"]
  n25["init-gen-flow-s01 | draft"]
  n26["confirm-qc-s02 | draft"]
  n27["confirm-random-s02 | draft"]
  n28["confirm-qc-s03 | draft"]
  n29["confirm-random-s03 | draft"]
  n30["init-linear-nocontext-s01 | completed"]
  n31["init-linear-nocontext-a01-s01 | completed"]
  n32["init-program-raw-s01 | completed"]
  n33["masked-response-shared-s01 | completed"]
  n34["masked-response-pca-s01 | completed"]
  n35["masked-response-program-s01 | completed"]
  n36["masked-response-random-s01 | completed"]
  n37["masked-response-qc-s01 | completed"]
  n38["masked-response-random-thin-s01 | completed"]
  n39["target-response-zero-s01 | draft"]
  n40["target-response-centroid-s01 | draft"]
  n41["target-response-go-s01 | draft"]
  n42["target-response-random-go-s01 | draft"]
  n43["transport-linear-s01 | draft"]
  n44["transport-mlp-s01 | completed"]
  n45["transport-no-ntc-s01 | draft"]
  n46["composition-lfc-s01 | completed"]
  n47["distribution-nb-s01 | completed"]
  n48["gene-decoder-s01 | completed"]
  n49["cell-cvae-s01 | failed"]
  n50["cell-cvae-aligned-s01 | completed"]
  n51["set-distribution-s01 | draft"]
  n52["task-observation-s01 | failed"]
  n53["task-observation-s02 | completed"]
  n54["local-capability-s01 | failed"]
  n55["local-capability-s02 | failed"]
  n56["local-capability-s03 | ready"]
  n0 -->|control| n1
  n0 -.->|"evidence: E-BASE"| n1
  n0 -->|control| n2
  n0 -.->|"evidence: E-BASE"| n2
  n0 -->|control| n3
  n0 -.->|"evidence: E-BASE"| n3
  n0 -->|control| n4
  n0 -.->|"evidence: E-BASE"| n4
  n0 -->|control| n5
  n0 -.->|"evidence: E-BASE"| n5
  n0 -.->|"evidence: E-QC"| n5
  n3 -.->|"evidence: E-QC"| n5
  n4 -.->|"evidence: E-QC"| n5
  n0 -->|control| n6
  n0 -.->|"evidence: E-BASE"| n6
  n0 -->|control| n7
  n0 -.->|"evidence: E-BASE"| n7
  n0 -->|control| n8
  n0 -.->|"evidence: E-BASE"| n8
  n8 -->|control| n9
  n0 -.->|"evidence: E-MOMENTS"| n9
  n8 -.->|"evidence: E-MOMENTS"| n9
  n32 -->|control| n10
  n0 -->|source| n10
  n0 -.->|"evidence: E-BASE"| n10
  n0 -.->|"evidence: E-CONDITION-ABLATION"| n10
  n30 -.->|"evidence: E-CONDITION-ABLATION"| n10
  n30 -.->|"evidence: E-RIDGE-STRENGTH"| n10
  n31 -.->|"evidence: E-RIDGE-STRENGTH"| n10
  n32 -->|control| n11
  n0 -->|source| n11
  n0 -.->|"evidence: E-BASE"| n11
  n0 -.->|"evidence: E-CONDITION-ABLATION"| n11
  n30 -.->|"evidence: E-CONDITION-ABLATION"| n11
  n30 -.->|"evidence: E-RIDGE-STRENGTH"| n11
  n31 -.->|"evidence: E-RIDGE-STRENGTH"| n11
  n32 -->|control| n12
  n0 -->|source| n12
  n0 -.->|"evidence: E-BASE"| n12
  n0 -.->|"evidence: E-CONDITION-ABLATION"| n12
  n30 -.->|"evidence: E-CONDITION-ABLATION"| n12
  n30 -.->|"evidence: E-RIDGE-STRENGTH"| n12
  n31 -.->|"evidence: E-RIDGE-STRENGTH"| n12
  n10 -->|control| n13
  n32 -.->|"evidence: E-PROGRAM"| n13
  n10 -.->|"evidence: E-PROGRAM"| n13
  n11 -.->|"evidence: E-PROGRAM"| n13
  n12 -.->|"evidence: E-PROGRAM"| n13
  n10 -->|control| n14
  n32 -.->|"evidence: E-PROGRAM"| n14
  n10 -.->|"evidence: E-PROGRAM"| n14
  n11 -.->|"evidence: E-PROGRAM"| n14
  n12 -.->|"evidence: E-PROGRAM"| n14
  n10 -->|control| n15
  n32 -.->|"evidence: E-PROGRAM"| n15
  n10 -.->|"evidence: E-PROGRAM"| n15
  n11 -.->|"evidence: E-PROGRAM"| n15
  n12 -.->|"evidence: E-PROGRAM"| n15
  n10 -->|control| n16
  n32 -.->|"evidence: E-PROGRAM"| n16
  n10 -.->|"evidence: E-PROGRAM"| n16
  n11 -.->|"evidence: E-PROGRAM"| n16
  n12 -.->|"evidence: E-PROGRAM"| n16
  n0 -->|control| n17
  n0 -.->|"evidence: E-BASE"| n17
  n0 -->|control| n18
  n0 -.->|"evidence: E-BASE"| n18
  n0 -->|control| n19
  n0 -.->|"evidence: E-BASE"| n19
  n0 -->|control| n20
  n0 -.->|"evidence: E-BASE"| n20
  n19 -->|control| n21
  n0 -.->|"evidence: E-CAPACITY"| n21
  n19 -.->|"evidence: E-CAPACITY"| n21
  n19 -->|control| n22
  n0 -.->|"evidence: E-CAPACITY"| n22
  n19 -.->|"evidence: E-CAPACITY"| n22
  n19 -->|control| n23
  n0 -.->|"evidence: E-CAPACITY"| n23
  n19 -.->|"evidence: E-CAPACITY"| n23
  n19 -->|control| n24
  n0 -.->|"evidence: E-CAPACITY"| n24
  n19 -.->|"evidence: E-CAPACITY"| n24
  n19 -.->|"evidence: E-CONTEXT"| n24
  n21 -.->|"evidence: E-CONTEXT"| n24
  n19 -->|control| n25
  n0 -.->|"evidence: E-CAPACITY"| n25
  n19 -.->|"evidence: E-CAPACITY"| n25
  n19 -.->|"evidence: E-CONTEXT"| n25
  n21 -.->|"evidence: E-CONTEXT"| n25
  n27 -->|control| n26
  n0 -.->|"evidence: E-QC"| n26
  n3 -.->|"evidence: E-QC"| n26
  n4 -.->|"evidence: E-QC"| n26
  n0 -.->|"evidence: E-QC"| n27
  n3 -.->|"evidence: E-QC"| n27
  n4 -.->|"evidence: E-QC"| n27
  n29 -->|control| n28
  n0 -.->|"evidence: E-QC"| n28
  n3 -.->|"evidence: E-QC"| n28
  n4 -.->|"evidence: E-QC"| n28
  n0 -.->|"evidence: E-QC"| n29
  n3 -.->|"evidence: E-QC"| n29
  n4 -.->|"evidence: E-QC"| n29
  n0 -->|control| n30
  n0 -->|source| n30
  n0 -.->|"evidence: E-BASE"| n30
  n0 -.->|"evidence: E-ANCHOR-AUDIT"| n30
  n30 -->|control| n31
  n0 -->|source| n31
  n0 -.->|"evidence: E-CONDITION-ABLATION"| n31
  n30 -.->|"evidence: E-CONDITION-ABLATION"| n31
  n0 -->|source| n32
  n0 -.->|"evidence: E-BASE"| n32
  n0 -.->|"evidence: E-CONDITION-ABLATION"| n32
  n30 -.->|"evidence: E-CONDITION-ABLATION"| n32
  n30 -.->|"evidence: E-RIDGE-STRENGTH"| n32
  n31 -.->|"evidence: E-RIDGE-STRENGTH"| n32
  n0 -->|source| n33
  n32 -.->|"evidence: E-PROGRAM"| n33
  n10 -.->|"evidence: E-PROGRAM"| n33
  n11 -.->|"evidence: E-PROGRAM"| n33
  n12 -.->|"evidence: E-PROGRAM"| n33
  n30 -.->|"evidence: E-RIDGE-STRENGTH"| n33
  n31 -.->|"evidence: E-RIDGE-STRENGTH"| n33
  n33 -->|control| n34
  n0 -->|source| n34
  n32 -.->|"evidence: E-PROGRAM"| n34
  n10 -.->|"evidence: E-PROGRAM"| n34
  n11 -.->|"evidence: E-PROGRAM"| n34
  n12 -.->|"evidence: E-PROGRAM"| n34
  n30 -.->|"evidence: E-RIDGE-STRENGTH"| n34
  n31 -.->|"evidence: E-RIDGE-STRENGTH"| n34
  n33 -->|control| n35
  n0 -->|source| n35
  n32 -.->|"evidence: E-PROGRAM"| n35
  n10 -.->|"evidence: E-PROGRAM"| n35
  n11 -.->|"evidence: E-PROGRAM"| n35
  n12 -.->|"evidence: E-PROGRAM"| n35
  n30 -.->|"evidence: E-RIDGE-STRENGTH"| n35
  n31 -.->|"evidence: E-RIDGE-STRENGTH"| n35
  n33 -->|control| n36
  n0 -->|source| n36
  n32 -.->|"evidence: E-PROGRAM"| n36
  n10 -.->|"evidence: E-PROGRAM"| n36
  n11 -.->|"evidence: E-PROGRAM"| n36
  n12 -.->|"evidence: E-PROGRAM"| n36
  n30 -.->|"evidence: E-RIDGE-STRENGTH"| n36
  n31 -.->|"evidence: E-RIDGE-STRENGTH"| n36
  n33 -->|control| n37
  n0 -->|source| n37
  n33 -->|source| n37
  n33 -.->|"evidence: E-MASKED-RESPONSE"| n37
  n34 -.->|"evidence: E-MASKED-RESPONSE"| n37
  n35 -.->|"evidence: E-MASKED-RESPONSE"| n37
  n36 -.->|"evidence: E-MASKED-RESPONSE"| n37
  n33 -->|control| n38
  n0 -->|source| n38
  n33 -->|source| n38
  n33 -.->|"evidence: E-MASKED-RESPONSE"| n38
  n34 -.->|"evidence: E-MASKED-RESPONSE"| n38
  n35 -.->|"evidence: E-MASKED-RESPONSE"| n38
  n36 -.->|"evidence: E-MASKED-RESPONSE"| n38
  n33 -.->|"evidence: E-SHARED-QC"| n39
  n37 -.->|"evidence: E-SHARED-QC"| n39
  n38 -.->|"evidence: E-SHARED-QC"| n39
  n39 -->|control| n40
  n33 -.->|"evidence: E-SHARED-QC"| n40
  n37 -.->|"evidence: E-SHARED-QC"| n40
  n38 -.->|"evidence: E-SHARED-QC"| n40
  n39 -->|control| n41
  n40 -->|control| n41
  n33 -.->|"evidence: E-SHARED-QC"| n41
  n37 -.->|"evidence: E-SHARED-QC"| n41
  n38 -.->|"evidence: E-SHARED-QC"| n41
  n39 -->|control| n42
  n40 -->|control| n42
  n33 -.->|"evidence: E-SHARED-QC"| n42
  n37 -.->|"evidence: E-SHARED-QC"| n42
  n38 -.->|"evidence: E-SHARED-QC"| n42
  n33 -->|control| n43
  n0 -->|source| n43
  n33 -->|source| n43
  n33 -.->|"evidence: E-MECHANISM-PORTFOLIO"| n43
  n44 -.->|"evidence: E-MECHANISM-PORTFOLIO"| n43
  n46 -.->|"evidence: E-MECHANISM-PORTFOLIO"| n43
  n47 -.->|"evidence: E-MECHANISM-PORTFOLIO"| n43
  n33 -->|control| n44
  n0 -->|source| n44
  n33 -->|source| n44
  n33 -.->|"evidence: E-SHARED-QC"| n44
  n37 -.->|"evidence: E-SHARED-QC"| n44
  n38 -.->|"evidence: E-SHARED-QC"| n44
  n33 -->|control| n45
  n0 -->|source| n45
  n33 -->|source| n45
  n33 -.->|"evidence: E-MECHANISM-PORTFOLIO"| n45
  n44 -.->|"evidence: E-MECHANISM-PORTFOLIO"| n45
  n46 -.->|"evidence: E-MECHANISM-PORTFOLIO"| n45
  n47 -.->|"evidence: E-MECHANISM-PORTFOLIO"| n45
  n33 -->|control| n46
  n0 -->|source| n46
  n33 -->|source| n46
  n33 -.->|"evidence: E-SHARED-QC"| n46
  n37 -.->|"evidence: E-SHARED-QC"| n46
  n38 -.->|"evidence: E-SHARED-QC"| n46
  n33 -->|control| n47
  n0 -->|source| n47
  n33 -->|source| n47
  n33 -.->|"evidence: E-SHARED-QC"| n47
  n37 -.->|"evidence: E-SHARED-QC"| n47
  n38 -.->|"evidence: E-SHARED-QC"| n47
  n33 -->|control| n48
  n0 -->|source| n48
  n33 -->|source| n48
  n33 -.->|"evidence: E-MECHANISM-PORTFOLIO"| n48
  n44 -.->|"evidence: E-MECHANISM-PORTFOLIO"| n48
  n46 -.->|"evidence: E-MECHANISM-PORTFOLIO"| n48
  n47 -.->|"evidence: E-MECHANISM-PORTFOLIO"| n48
  n33 -->|control| n49
  n0 -->|source| n49
  n33 -->|source| n49
  n46 -->|source| n49
  n33 -.->|"evidence: E-MECHANISM-PORTFOLIO"| n49
  n44 -.->|"evidence: E-MECHANISM-PORTFOLIO"| n49
  n46 -.->|"evidence: E-MECHANISM-PORTFOLIO"| n49
  n47 -.->|"evidence: E-MECHANISM-PORTFOLIO"| n49
  n33 -->|control| n50
  n0 -->|source| n50
  n33 -->|source| n50
  n46 -->|source| n50
  n33 -.->|"evidence: E-MECHANISM-PORTFOLIO"| n50
  n44 -.->|"evidence: E-MECHANISM-PORTFOLIO"| n50
  n46 -.->|"evidence: E-MECHANISM-PORTFOLIO"| n50
  n47 -.->|"evidence: E-MECHANISM-PORTFOLIO"| n50
  n33 -->|control| n51
  n0 -->|source| n51
  n50 -->|source| n51
  n0 -->|source| n52
  n0 -->|source| n53
  n53 -->|source| n54
  n53 -->|source| n55
  n54 -->|source| n55
  n53 -->|source| n56
  n54 -->|source| n56
  n55 -->|source| n56
```
