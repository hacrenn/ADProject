#ADProject — Análise de Dados

##Descrição rápida
ADProject é um projeto de análise de dados que investiga a relação entre indicadores de saúde mental (p.ex. depressão, ansiedade), tempo passado em redes sociais e medidas de bem-estar (p.ex. happiness index). O repositório contém um notebook interativo, um script que automatiza a cadeia de processamento, conjuntos de dados CSV e um relatório com metodologia e conclusões. Repositório: GitHub — Autor / mantenedor: hacrenn Tecnologias principais usadas: Python Software Foundation, Project Jupyter


##Objetivo do projecto

O objetivo é demonstrar um fluxo completo de análise exploratória e visualização para responder perguntas como:
* Existe correlação entre o tempo médio diário em redes sociais e taxas de depressão por país/idade?
* Como evoluíram indicadores de ansiedade e depressão ao longo do tempo (temporalidade por país)?
* Que grupos demográficos (idade/género) apresentam maior exposição a fatores de risco identificados?
O foco é exploratório — identificar padrões, gerar visualizações esclarecedoras e propor hipóteses para análises estatísticas mais robustas.


##Conteúdo do repositório (resumo dos ficheiros)

* ProjetoAD.ipynb — Notebook Jupyter com o fluxo completo: leitura de dados, limpeza, fusão (merge), exploração e visualizações interativas.
* ProjetoAD.py — Script Python que reproduz o pipeline do notebook de forma não interativa (útil para execução automática e geração de figuras/outputs).
* RelatórioAD.pdf — Documento escrito com metodologia, resultados, interpretações e recomendações.
* data/ (ou ficheiros CSV na raiz) — Conjunto de ficheiros CSV usados (prevalência de doenças mentais, tempo em redes sociais, indicadores socioeconómicos, happiness index, etc.).
* docs/ (opcional) — Notas, gráficos exportados e ficheiros auxiliares usados para o relatório.
Nota: alguns ficheiros no repositório podem conter caminhos absolutos; recomenda-se usar caminhos relativos e organizar todos os CSVs numa pasta data/ antes de executar o notebook ou o script.


##Descrição dos dados e fontes

O projeto integra múltiplas fontes para construir uma base harmonizada por país/ano:
* Dados de prevalência de transtornos mentais — percentuais por país/ano para depressão, ansiedade, etc.
* Dados de utilização de redes sociais — métricas de tempo médio diário/semana por faixa etária/plataforma (quando disponível).
* Indicadores de bem-estar — happiness index, PIB per capita, e outros fatores socioeconómicos de contexto.
* Dados demográficos — distribuição por idade e género para estratificação.
Sempre que possível, as fontes originais devem ser documentadas (fonte, ano, link). No RelatórioAD.pdf estão referenciadas as bases utilizadas e as transformações aplicadas.


##Fluxo metodológico (alto nível)

1. Ingestão — carregar CSVs, inspecionar colunas e tipos de dados.
2. Limpeza — harmonizar nomes de colunas, converter tipos (numérico / data), e lidar com valores em falta (missing values) e outliers.
3. Harmonização — renomear colunas e mapear variáveis de diferentes fontes para um esquema comum (ex.: Entity→ Country).
4. Merge / Join — combinar datasets por chaves (geralmente Country + Year), escolhendo estratégias de join que preservem observações relevantes.
5. Transformação — criar variáveis derivadas (normalizações, percentis, categorizações por faixa etária).
6. Exploração — estatísticas descritivas, correlações, pivot tables e visualizações (mapas coropléticos, séries temporais, scatter plots estratificados).
7. Interpretação — sumarizar padrões observados, limitações e hipóteses para investigação adicional.
As decisões de tratamento (p.ex. imputação de missing, remoção de outliers, escolha de agregação temporal) são descritas no relatório.
