# Hackathon Customer Analysis

[![CI](https://github.com/gildemberg-santos/hackathon_customer_analysis/actions/workflows/pylint.yml/badge.svg)](https://github.com/gildemberg-santos/hackathon_customer_analysis/actions/workflows/pylint.yml)

Protótipo construído em um hackathon (2023) para responder a uma pergunta comercial: **quais leads têm mais chance de fechar a compra?**

O modelo é uma regressão logística treinada com o histórico de vendas de veículos. Cada lead é descrito pelo valor da venda, pelo veículo de interesse e por atributos financeiros, e o modelo estima se a venda foi concluída (`sale_sold`).

## Como funciona

1. Lê o histórico de `data/exported_data.csv`.
2. Separa 75% dos registros para treino e 25% para teste (`random_state=0`, resultado reproduzível).
3. Treina uma `LogisticRegression` do scikit-learn e mede a acurácia no conjunto de teste.
4. Salva o modelo em `data/modelo.pkl` para ser reaproveitado sem novo treino.

Colunas usadas como entrada:

| Coluna | Significado |
|---|---|
| `sale_amount` | Valor da venda |
| `sale_vehicle_id` | Veículo negociado |
| `vehicle_year` | Ano do veículo |
| `vehicle_price` | Preço do veículo |
| `lead_bank_score` | Score bancário do lead |
| `lead_have_children` | Lead tem filhos |
| `lead_years_employment` | Anos no emprego atual |
| `lead_income` | Renda do lead |

## Rodando

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# coloque o histórico em data/exported_data.csv (não versionado)
python main.py
```

O script imprime a acurácia no conjunto de teste e as predições.

## Testes

```sh
pip install pytest pylint
pytest -q
pylint $(git ls-files '*.py')
```

Os testes usam um conjunto sintético, então rodam sem os dados reais. O GitHub Actions executa lint e testes em Python 3.10, 3.11 e 3.12.

## Limitações e próximos passos

Como protótipo de hackathon, o projeto deixa de lado pontos que um modelo em produção precisaria:

- `sale_vehicle_id` entra como número, mas é categórico; deveria ser codificado (one-hot ou target encoding).
- Não há normalização das variáveis numéricas nem tratamento de valores ausentes.
- Acurácia sozinha engana quando poucas vendas fecham; precisão, recall e AUC dariam uma visão melhor.
- O próximo passo natural é expor a predição por uma API e devolver a probabilidade de fechamento, não só a classe.

## Licença

[MIT](LICENSE)
