# === Imports ===
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import random
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Reprodutibilidade
RANDOM_STATE = 23
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# === Configuração do pré-processamento ===
colunas_categoricas = ["Tipo_Assinatura", "Nivel_Satisfacao", "Cidade", "Genero", "Desconto_Aplicado"]
colunas_numericas = ["Gasto_Total", "Itens_Comprados", "Dias_Sem_Compra", "Idade", "Nota_Media"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]), colunas_numericas),

        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), colunas_categoricas)
    ]
)

# === Carregar e preparar dataset ===
@st.cache_resource
def carregar_dados():
    df = pd.read_csv("Data/Previsao_Churn.csv")

    # Renomear colunas
    df.rename(columns={
        "Customer ID": "ID_Cliente",
        "Gender": "Genero",
        "Age": "Idade",
        "City": "Cidade",
        "Membership Type": "Tipo_Assinatura",
        "Total Spend": "Gasto_Total",
        "Items Purchased": "Itens_Comprados",
        "Average Rating": "Nota_Media",
        "Discount Applied": "Desconto_Aplicado",
        "Days Since Last Purchase": "Dias_Sem_Compra",
        "Satisfaction Level": "Nivel_Satisfacao"
    }, inplace=True)

    # Criar variável alvo (churn real)
    df["Churn"] = (df["Dias_Sem_Compra"] > 30).astype(int)
    return df

# === Treinar modelo ===
@st.cache_resource
def treinar_modelo(df):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    rf_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE))
    ])

    rf_model.fit(X_train, y_train)
    return rf_model

# === App ===
st.set_page_config(page_title="Previsão de Churn", layout="wide")
st.title("📊 Dashboard de Churn em E-commerce")

df = carregar_dados()
rf_model = treinar_modelo(df)

# Adicionar previsões ao dataset
df["Pred_Churn"] = rf_model.predict(df.drop(columns=["Churn"]))
df["Prob_Churn"] = rf_model.predict_proba(df.drop(columns=["Churn"]))[:, 1]

# === Sidebar com filtros ===
st.sidebar.header("🔍 Filtros")
filtros = {}

for col in ["Genero", "Cidade", "Tipo_Assinatura", "Desconto_Aplicado", "Nivel_Satisfacao"]:
    opcoes = ["Todos"] + sorted(df[col].dropna().unique().tolist())
    escolha = st.sidebar.selectbox(f"{col}", opcoes)
    if escolha != "Todos":
        filtros[col] = escolha

# Aplicar filtros
df_filtrado = df.copy()
for col, val in filtros.items():
    df_filtrado = df_filtrado[df_filtrado[col] == val]

# === Métricas gerais ===
st.subheader("📈 Métricas Gerais (após filtros)")
total = len(df_filtrado)
pred_real = df_filtrado["Churn"].sum()
pred_real_pct = (pred_real / total * 100) if total > 0 else 0

pred_prev = df_filtrado["Pred_Churn"].sum()
pred_prev_pct = (pred_prev / total * 100) if total > 0 else 0

prob_media = df_filtrado["Prob_Churn"].mean() if total > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Analisado", total)
col2.metric("Predição (Churn Real)", f"{pred_real} clientes", f"{pred_real_pct:.1f}%")
col3.metric("Previsão (Modelo)", f"{pred_prev} clientes", f"{pred_prev_pct:.1f}%")
col4.metric("Prob. Média de Churn", f"{prob_media*100:.1f}%")

# === Gráficos ===
st.subheader("📊 Análises Gráficas")

# Importância das variáveis
importancias = rf_model.named_steps["classifier"].feature_importances_
feature_names_num = colunas_numericas
feature_names_cat = rf_model.named_steps["preprocessor"].transformers_[1][1]\
    .named_steps["onehot"].get_feature_names_out(colunas_categoricas)
feature_names = list(feature_names_num) + list(feature_names_cat)

importancias_df = pd.DataFrame({
    "Variavel": feature_names,
    "Importancia": importancias
}).sort_values(by="Importancia", ascending=False).head(10)

st.write("### 🔑 Top 10 Variáveis Mais Importantes")
st.bar_chart(importancias_df.set_index("Variavel"))

# Dispersão
st.write("### 🌐 Dispersão: Dias sem Compra x Nível de Satisfação")
fig, ax = plt.subplots()

# Garantir mapeamento do nível de satisfação
mapa_satisfacao = {"Baixo": 1, "Médio": 2, "Alto": 3,
                   "Low": 1, "Medium": 2, "High": 3}
df_filtrado["Nivel_Satisfacao_Num"] = df_filtrado["Nivel_Satisfacao"].map(mapa_satisfacao)

# Remover linhas sem dados válidos
df_plot = df_filtrado.dropna(subset=["Dias_Sem_Compra", "Nivel_Satisfacao_Num"])

if df_plot.empty:
    st.warning("⚠️ Nenhum dado disponível para plotar com os filtros selecionados.")
else:
    ax.scatter(df_plot["Dias_Sem_Compra"], df_plot["Nivel_Satisfacao_Num"],
               c=df_plot["Pred_Churn"], cmap="coolwarm", alpha=0.6)
    ax.set_xlabel("Dias sem Compra")
    ax.set_ylabel("Nível de Satisfação (1=Baixo, 2=Médio, 3=Alto)")
    st.pyplot(fig)
