# Implementação e Análise de Market Making em LOB: Avellaneda–Stoikov em ABM

Este repositório contém o código-fonte e os experimentos do meu Trabalho de Conclusão de Curso (TCC) em Ciências Econômicas pelo Insper. O projeto foca na implementação do modelo estocástico de Avellaneda-Stoikov (2008) utilizando o simulador de microestrutura de mercado baseado em agentes ABIDES.

**Autor:** Bruno Drezza Reis de Souza  
**Orientador:** Prof. Raul Ikeda Gomes da Silva  
**Instituição:** Insper (Bacharelado em Ciências Econômicas)  
**Ano:** 2025

## Visão Geral do Projeto
A provisão de liquidez em alta frequência (HFT) expõe os formadores de mercado ao risco de inventário e seleção adversa. Este projeto traduz as equações de controle estocástico do modelo clássico de Avellaneda-Stoikov em regras computacionais de tomada de decisão, inserindo o agente em um mercado sintético realista (RMSC03) para avaliar:
- Rentabilidade e controle de risco de inventário.
- Dinâmica de ajuste do *bid-ask spread*.
- Impacto da aversão ao risco ($\gamma$) na formulação de preços.

