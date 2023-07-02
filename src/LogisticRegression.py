class LogisticRegression:
    """
    Classificador de regressão logística.
    Parametros
    ----------
    n_iterations: int, default=500
        Número máximo de iterações para convergir.
    learning_rate float, default=0.01
        Taxa de aprendizado.
    ----------
    """    
    # Inicializando a função com os parametros learning_rate e n_iterations
    def __init__(self, learning_rate=0.01, n_iterations=500):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
    
    # Implementando a função logistica
    def __sigmoid(self, z):
        np.seterr(all='ignore')
        return 1 / (1 + np.exp(-z))
    
    # Implementando a função de custo: Entropia Cruzada Binária/Log Loss
    def __log_loss(self, y, yhat):
        logloss = (-1 / self.__m) * np.sum(y.T.dot(np.log(yhat)) + (1 - y).T.dot(np.log(1 - yhat)))
        return logloss
    
    # Implementando a função de otimização: Gradiente Descendente
    def __gradient_descent(self, X, y, yhat, theta):
        theta -= self.learning_rate * (1 / self.__m * X.T.dot((yhat - y)))
        return theta
    
    # Definindo a função de ajuste do modelo: processo de treinamento
    def fit(self, X, y, class_weight=None):
        """
        Ajuste o modelo de acordo com os dados de treinamento fornecidos.
        """
        # if self.class_weight:
        #     if self.class_weight =='balanced':
        #         self.w = self.__compute_class_weight(y)
        #     else: self.w = self.class_weight
        # else: self.w = {i:1 for i in np.unique(y)}
        self.classes_ = np.unique(y)
        self.__m = np.float64(X.shape[0])
        self.losses = list()
        theta = np.zeros((X.shape[1]))
        for _ in range(self.n_iterations):
            z = np.dot(X, theta)
            yhat = self.__sigmoid(z)
            loss = self.__log_loss(y, yhat)
            theta = self.__gradient_descent(X, y, yhat, theta)
            self.losses.append(loss)
        self.theta = theta
    
    # Definindo a função de estimador do modelo
    def predict(self, X):
        """
        Prever rótulos de classe para amostras em X.
        """
        z = np.dot(X, self.theta)
        proba = self.__sigmoid(z)
        return np.asarray([1 if p > 0.5 else 0 for p in proba])
    
    
    # # Definindo função para calculo automatico de pesos de classes desbalanceadas
    # def __compute_class_weight(self, y):
    #     """
    #     Estima os pesos das classes para conjuntos de dados não balanceados.
    #     """
    #     np.seterr(all='ignore')
    #     class_weight = dict()
    #     classes = np.unique(y)
    #     n_classes = len(classes)
    #     n_samples = len(y)
    #     for i, c in enumerate(classes):
    #         class_weight[c] = (n_samples / (n_classes * np.bincount(y)[i])).round(8)
    #     return class_weight