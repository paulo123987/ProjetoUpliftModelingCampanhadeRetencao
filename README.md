# Projeto Uplift Modeling — Campanha de Retenção

> **Disciplina:** Data Science Aplicada  
> **Objetivo:** Identificar os clientes *Persuadíveis* — os únicos que justificam o investimento em campanhas de retenção.

---

## Problema de Negócio

Empresas investem em campanhas de retenção sem saber exatamente *quem* vai responder. O resultado: recursos desperdiçados em clientes que já ficariam (*Sure Things*) ou que nunca ficariam (*Lost Causes*).

**Uplift Modeling** resolve isso medindo o *efeito incremental* de uma campanha para cada cliente individualmente, não apenas se ele ficou — mas se a campanha *fez ele ficar*.

---

## Os 4 Tipos de Clientes (Radcliffe & Surry, 2011)

| Segmento | Sem Campanha | Com Campanha | Ação |
|----------|-------------|-------------|------|
| **Persuadíveis** | Cancela | Permanece | ✅ Investir — aqui está o ROI |
| **Sure Things** | Permanece | Permanece | ⚠️ Monitorar — não priorizar |
| **Lost Causes** | Cancela | Cancela | ❌ Ignorar — campanha não ajuda |
| **Sleeping Dogs** | Permanece | Cancela | 🚫 EVITAR — campanha prejudica! |

---

## Estrutura do Notebook (80 células)

| Bloco | Conteúdo |
|-------|----------|
| **0–1** | Título, instalação de dependências e imports |
| **2** | Carregamento individual dos 3 datasets + análise de missing values |
| **3** | EDA Univariada (histogramas, pizza, barras) |
| **4** | Join das tabelas + criação da coluna `grupo` |
| **5** | EDA Bivariada/Multivariada (boxplots, heatmap, scatter, faixa etária) |
| **6** | Feature Engineering — 14 features com justificativa de negócio |
| **7** | T-Learner com 4 algoritmos: Naive Bayes · Logistic Regression · Random Forest · Gradient Boosting |
| **8** | Avaliação: Uplift Curves, AUUC, análise por decil, métricas clássicas (AUC/F1/KS) e Cross-Validation |
| **9** | Interpretabilidade SHAP — Top 10 features com nota de negócio |
| **10** | Segmentação nos 4 tipos (Radcliffe & Surry 2011) |
| **11** | Storytelling executivo + simulador de ROI financeiro |
| **12** | Referências acadêmicas |

---

## Metodologia: T-Learner (Two-Model Approach)

```
Score de Uplift = P(Y=1 | X, campanha=SIM) − P(Y=1 | X, campanha=NÃO)
```

Para cada algoritmo, treinamos **dois modelos separados**:
- **Modelo T**: treinado apenas com clientes que receberam a campanha
- **Modelo C**: treinado apenas com clientes que não receberam

O *uplift score* de cada cliente é a diferença das probabilidades preditas.

---

## Feature Engineering (14 features)

| # | Feature | Lógica | Justificativa |
|---|---------|--------|---------------|
| 1 | `genero_num` | M=1, F=0 | ML não processa texto |
| 2 | `renda_por_idade` | renda / idade | Poder aquisitivo relativo à fase de vida |
| 3 | `score_fidelidade` | satisfação × tempo | Captura fidelidade real: satisfeito E antigo |
| 4 | `renda_alta` | renda > mediana | Flag cliente premium |
| 5 | `cliente_antigo` | tempo > 24 meses | Comportamento distinto de clientes consolidados |
| 6 | `renda_log` | log(renda + 1) | Reduz assimetria da distribuição de renda |
| 7 | `interacao_renda_satisfacao` | renda × score / 1000 | Perfil premium: rico + satisfeito |
| 8 | `risco_churn` | (1−score/10) × (1−antigo) | Proxy de risco: novo + insatisfeito |
| 9 | `tempo_por_satisfacao` | tempo / (score + 1) | Fidelidade fragilizada: antigo mas insatisfeito |
| 10 | `valor_cliente_estimado` | (renda/1000) × (tempo/12) | Proxy LTV anual |

---

## Algoritmos Utilizados

| # | Família | Algoritmo | Parâmetros |
|---|---------|-----------|-----------|
| 1 | Probabilístico | `GaussianNB` | Baseline |
| 2 | Probabilístico Linear | `LogisticRegression` | max_iter=1000 |
| 3 | Ensemble | `RandomForestClassifier` | n_estimators=100, **max_depth=5** |
| 4 | Boosting | `GradientBoostingClassifier` | n_estimators=100, learning_rate=0.1, max_depth=3 |

> **Nota:** `max_depth=5` no Random Forest evita overfitting. Sem este parâmetro o modelo memoriza o treino e atinge AUC=1.0 artificialmente.

---

## Métricas de Avaliação

### Métricas de Uplift
- **AUUC** (Area Under Uplift Curve) — métrica global; quanto maior, melhor
- **Análise por Decil** — métrica local; bom modelo concentra retenção nos decis superiores

### Métricas Clássicas por Grupo (Controle e Tratamento)
- AUC-ROC · Accuracy · F1-Score · Precision · Recall · KS (Kolmogorov-Smirnov)
- Calculadas com **Cross-Validation 5-fold** para evitar data leakage

---

## Datasets

```
Datasets_Projeto_Pratico_DSCS/
├── clientes.csv    — 1.000 clientes (id, idade, genero, renda_mensal, tempo, score)
├── campanha.csv    — 1.000 linhas   (id_cliente, recebeu_campanha 0/1)
└── resposta.csv    — 1.000 linhas   (id_cliente, manteve_contrato 0/1)
```

---

## Como Executar

### Pré-requisitos
```bash
pip install -r requirements.txt
```

### Execução
1. Abrir `projeto_uplift_modeling.ipynb` no VS Code com a extensão Jupyter
2. Executar célula a célula com `Shift+Enter`
3. A primeira célula instala as dependências automaticamente

---

## Simulação de ROI

```
Custo por ação (contato + incentivo):  R$ 65,00
Ticket médio mensal estimado:          2% da renda mensal
LTV anual estimado:                    ticket × 12 meses

Cenário Universal:  envia para 100% da base → custo alto, retorno diluído
Cenário Uplift:     envia apenas para Persuadíveis → custo focado, ROI positivo
```

---

## Referências

- Radcliffe, N. J., & Surry, P. D. (2011). *Real-World Uplift Modelling with Significance-Based Uplift Trees*. Stochastic Solutions.
- Gutierrez, P., & Gérardy, J.-Y. (2017). *Causal Inference and Uplift Modeling: A review of the literature*. JMLR Workshop.
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS 30.
- Anderson, E. T., & Simester, D. (2011). *A step-by-step guide to smart business experiments*. Harvard Business Review.
- Scikit-learn developers (2024). *scikit-learn: Machine Learning in Python*. https://scikit-learn.org
- Plotly Technologies Inc. (2015). *Collaborative data science*. https://plotly.com
- McKinney, W. (2010). *Data Structures for Statistical Computing in Python*. SciPy Proceedings.
- Harris, C. R. et al. (2020). *Array programming with NumPy*. Nature, 585, 357–362.

---

## Estrutura do Repositório

```
├── projeto_uplift_modeling.ipynb   ← Notebook principal (80 células)
├── Datasets_Projeto_Pratico_DSCS/
│   ├── clientes.csv
│   ├── campanha.csv
│   └── resposta.csv
├── requirements.txt
└── README.md
```

---

*Projeto desenvolvido com Python 3.x · Pandas · Scikit-learn · Plotly · SHAP*
