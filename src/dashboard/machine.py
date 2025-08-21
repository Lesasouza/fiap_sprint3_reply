import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
  


def _machine_learning_results():
    
    """
    Função para exibir os resultados do processo de machine learning.
    """
    # Verifica se a pasta de resultados existe
    pasta_resultados = os.path.join(os.path.dirname(__file__), "..", "machine_learning")
    
    
    if not os.path.exists(pasta_resultados):
        st.error("Pasta de resultados não encontrada.")
        return 
    st.title("Resultados do Machine Learning")

    st.markdown("##### 🧪 O processo de otimização de modelos de machine learning foi concluído com sucesso. A seguir, apresentamos os resultados dos modelos otimizados, incluindo métricas de avaliação e tempos de treinamento.")

    st.markdown("##### 🧠 Nosso grande desafio foi identificar corretamente a classe de sementes de trigo com base em atributos físicos como área, perímetro e compacidade. Como as classes são balanceadas e os dados possuem características não-lineares, modelos que exploram essa complexidade naturalmente se destacam.")     

    
    resultado_caminho = os.path.join(pasta_resultados, "resultados_modelos_otimizados.csv")
    tempo_caminho = os.path.join(pasta_resultados, "tempos_modelos_otimizados.csv")

    if not os.path.exists(resultado_caminho) or not os.path.exists(tempo_caminho):
        st.error("Arquivos de resultados ou tempos não encontrados.")
        
    else:
        # Carrega o Dataset de resultados
        df_resultados = pd.read_csv(resultado_caminho)
        df_tempos = pd.read_csv(tempo_caminho)
        
        # Pega o numero de modelos para saber quantas cores gerar
        num_modelos = len(df_resultados)
        
        cores_gradiente = plt.colormaps['viridis'](np.linspace(0.2, 0.8, num_modelos))


        st.header("Tabela de Métricas de Avaliação")
        st.dataframe(df_resultados.set_index('Modelo'))

        st.header("Gráficos de Comparação")

        # Gráfico de Acurácia
        st.subheader("Acurácia dos Modelos")
        fig_acuracia, ax_acuracia = plt.subplots(figsize=(8, 4))
        ax_acuracia.barh(df_resultados['Modelo'], df_resultados['Accuracy'], color=cores_gradiente)
        ax_acuracia.set_xlabel('Acurácia')
        ax_acuracia.set_title('Acurácia por Modelo Otimizado')
        ax_acuracia.set_xlim(0, 1) 
        st.pyplot(fig_acuracia)
        
        st.text('''
            Modelos menos precisos:
        - Decision Tree (d3 e dNone) aparecem no início, com acurácia um pouco inferior aos demais (cerca de 0.82 - 0.84). 
        - Esses modelos tendem a ser mais simples e menos robustos, o que explica a menor performance.
        
            Melhor desempenho:
        - MLP (rede neural com 100 neurônios na camada escondida) apresenta a maior acurácia (~0.93).
        - Isso sugere que o problema em questão se beneficia de maior capacidade de modelagem não-linear.
        
        ''')

        # Gráfico de F1-Score
        st.subheader("F1-Score dos Modelos")
        fig_f1, ax_f1 = plt.subplots(figsize=(6, 4))
        ax_f1.barh(df_resultados['Modelo'], df_resultados['F1 Score'], color=cores_gradiente)
        ax_f1.set_xlabel('F1 Score')
        ax_f1.set_title('F1 Score por Modelo Otimizado')
        ax_f1.set_xlim(0, 1)
        st.pyplot(fig_f1)

        st.text('''
            Modelos com menor F1 Score:
        - Decision Tree (d3 e dNone) estão no final da lista novamente, com desempenho inferior (F1 próximo de 0.82 - 0.84).
        - Isso reforça que árvores de decisão simples não conseguem capturar bem a complexidade do problema.

            Melhor desempenho:
        - MLP (rede neural com 100 neurônios na camada escondida) apresenta a maior acurácia (~0.93).
        - Isso sugere que o problema em questão se beneficia de maior capacidade de modelagem não-linear.
        ''')
        # Gráfico de Tempo de Treinamento
        st.subheader("Tempo de Treinamento")
        fig_tempo, ax_tempo = plt.subplots(figsize=(6, 4))
        ax_tempo.bar(df_tempos['Modelo'], df_tempos['Tempo Treinamento (s)'], color=cores_gradiente)
        ax_tempo.set_ylabel('Tempo (s)')
        ax_tempo.set_title('Tempo de Treinamento por Modelo Otimizado')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        
        st.pyplot(fig_tempo)
        
        st.text('''
            Modelos extremamente rápidos:
        - Logistic Regression (86), Decision Tree (d3, dNone), Naive Bayes, QDA, KNN (5 e 7) tiveram tempos de treinamento praticamente desprezíveis (menos de 2 segundos).
        - São modelos leves, adequados quando velocidade é importante.)
        
            Modelos muito pesados:
        - GradBoost 200 (~35s) e principalmente MLP (100) (~110s) foram os mais lentos.
        - O custo da rede neural foi muito superior ao dos demais, confirmando o trade-off: maior acurácia/F1, mas com alto tempo de treinamento.
        
                ''')
        
        
        st.header("🔍 Resumo da Análise dos Resultados")
        
        st.markdown("###### - Modelos lineares como **Logistic Regression** e **LDA** alcançam desempenho razoável, mas não atingem a performance dos ensembles, por serem limitados a relações lineares entre características e classe.")
                
        st.markdown("###### - Random Forest e outras técnicas de ensemble têm maior capacidade de modelar interações complexas e reduzir o overfitting, por isso figuram entre os melhores.")
                
        st.markdown("###### - A melhora significativa do SVM sigmoid sugere que esse kernel específico é particularmente eficaz para nossos dados, capturando não linearidades que kernels lineares não conseguem.")
        
        
        st.header("🎯 Conclusão")
        
        st.markdown("### Modelos simples mantêm utilidade para cenários que priorizam interpretabilidade e rapidez. Para a tarefa de classificação de sementes, recomenda-se priorizar modelos como Random Forest e SVM otimizados, avaliando a relação custo-benefício entre ganho de performance e tempo computacional.")
    