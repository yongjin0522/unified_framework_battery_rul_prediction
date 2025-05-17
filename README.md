# A unified deep learning framework for high-performance RUL prediction in user-behavior batteries

Data Access: https://data.mendeley.com/my-data/

Each of the four neural network based models is trained individually using the selected features. The prediction of RUL is performed at every cycle using historical data accumulated over previous $L$ cycles, where $L$ represents the sliding sequence length for estimating the remaining life at the next cycle (Fig. \ref{fig:intro_method}). For example, if a battery has a total lifespan of 500 cycles, and the remaining life is to be estimated at the $20^{th}$ cycle, the goal is to accurately predict a value of 480 using data from the $10^{th}$ to the $19^{th}$ cycles when $L$ is set to 10. A smaller $L$ value poses a greater challenge, as the prediction must be based on limited data. To balance this, $L$ was set to 10 in the main experiments, ensuring a reliable demonstration of our framework's performance.

A weighted ensemble technique is applied to enhance performance, using predictions from the four deep learning models, each demonstrating high accuracy in RUL predictions, as inputs to the ensemble model. Ensemble methods enhance prediction accuracy by combining the strengths of multiple models, reducing individual model biases and improving overall robustness~\cite{ganaie2022ensemble}. As shown in eqn (\ref{eq:ensemble}), the final estimation $\hat{y}$ is obtained by combining the estimations from the four deep learning models $\hat{y}_i$, where $i$ is the index for the MLP, GRU, LSTM, and Transformer models. The selection of these models is based on their high performance when using 13 HIs. The weights ($w_i$) assigned to each model are optimized by training a single regression layer using the training data one more time.

\begin{align} 
\hat{y} = \sum_{i=1}^{4} w_i\hat{y}_i
\label{eq:ensemble}
\end{align}
