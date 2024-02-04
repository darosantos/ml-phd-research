# IA Generate 

################## Primeiro código
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from collections import deque
import numpy as np

class SlidingWindowEnsemble:
    def __init__(self, window_size, k_best_classifiers):
        self.window_size = window_size
        self.k_best_classifiers = k_best_classifiers
        self.classifiers = []
        self.data_window = deque(maxlen=window_size)

    def train_classifier(self, X_train, y_train):
        # Treine um novo classificador (árvore de decisão neste exemplo)
        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)
        self.classifiers.append(classifier)

    def update_ensemble(self):
        # Mantenha apenas os k melhores classificadores
        if len(self.classifiers) > self.k_best_classifiers:
            accuracies = []

            for clf in self.classifiers:
                # Avalie o desempenho de cada classificador no conjunto de dados de janela
                predictions = clf.predict(self.data_window)
                accuracy = accuracy_score(self.data_window[-len(predictions):], predictions)
                accuracies.append(accuracy)

            # Selecione os k melhores classificadores com base nas precisões
            selected_classifiers = np.argsort(accuracies)[-self.k_best_classifiers:]

            # Atualize a lista de classificadores mantendo apenas os selecionados
            self.classifiers = [self.classifiers[i] for i in selected_classifiers]

    def predict(self, X):
        # Atualize a janela de dados e a lista de classificadores
        self.data_window.extend(X)
        self.update_ensemble()

        # Faça previsões usando a votação dos classificadores
        predictions = np.array([clf.predict(X) for clf in self.classifiers])
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

        return majority_vote

# Exemplo de uso
# Suponha que você tenha um fluxo contínuo de dados X e rótulos correspondentes y
# Aqui, o código está assumindo que você já tem um conjunto de dados de treinamento inicial para iniciar os classificadores
X_train_initial, y_train_initial = ...

# Configuração da janela deslizante e número de melhores classificadores
window_size = 100
k_best_classifiers = 3

# Crie uma instância do SlidingWindowEnsemble
ensemble = SlidingWindowEnsemble(window_size, k_best_classifiers)

# Inicialize os classificadores com o conjunto de dados de treinamento inicial
for i in range(k_best_classifiers):
    ensemble.train_classifier(X_train_initial, y_train_initial)

# Fluxo contínuo de dados
for new_data_point, new_label in zip(stream_of_data_X, stream_of_data_y):
    # Atualize o ensemble com novos dados
    ensemble.predict(new_data_point)

    # Treine um novo classificador com os novos dados
    ensemble.train_classifier(new_data_point, new_label)

################## Segundo código
# Modifique esse algoritmo para que seja calculado a entropia de Shannon entre 
# janelas e compare os resultados usando a função de Kullback–Leibler e se o valor 
# da comparação for superior a determinado limite é que seja atualizado o ensemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import numpy as np

class SlidingWindowEnsemble:
    def __init__(self, window_size, k_best_classifiers, kl_threshold):
        self.window_size = window_size
        self.k_best_classifiers = k_best_classifiers
        self.kl_threshold = kl_threshold
        self.classifiers = []
        self.data_window = deque(maxlen=window_size)

    def train_classifier(self, X_train, y_train):
        # Treine um novo classificador (árvore de decisão neste exemplo)
        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)
        self.classifiers.append(classifier)

    def calculate_entropy(self, window):
        # Calcula a entropia de Shannon da janela
        class_probabilities = np.bincount(window) / len(window)
        return entropy(class_probabilities, base=2)

    def kl_divergence(self, p, q):
        # Calcula a divergência de Kullback-Leibler entre duas distribuições de probabilidade
        return np.sum(np.where(p != 0, p * np.log2(p / q), 0))

    def update_ensemble(self):
        # Mantenha apenas os k melhores classificadores
        if len(self.classifiers) > self.k_best_classifiers:
            entropies = [self.calculate_entropy(window) for window in self.data_window]

            # Calcule a KL-divergência entre a última janela e as anteriores
            kl_divergences = [self.kl_divergence(entropies[-1], entropies[i]) for i in range(len(entropies) - 1)]

            # Se a KL-divergência for maior que o limite, atualize o ensemble
            if max(kl_divergences, default=0) > self.kl_threshold:
                accuracies = []

                for clf in self.classifiers:
                    # Avalie o desempenho de cada classificador no conjunto de dados de janela
                    predictions = clf.predict(self.data_window[-1])
                    accuracy = accuracy_score(self.data_window[-1], predictions)
                    accuracies.append(accuracy)

                # Selecione os k melhores classificadores com base nas precisões
                selected_classifiers = np.argsort(accuracies)[-self.k_best_classifiers:]

                # Atualize a lista de classificadores mantendo apenas os selecionados
                self.classifiers = [self.classifiers[i] for i in selected_classifiers]

    def predict(self, X):
        # Atualize a janela de dados e a lista de classificadores
        self.data_window.extend(X)
        self.update_ensemble()

        # Faça previsões usando a votação dos classificadores
        predictions = np.array([clf.predict(X) for clf in self.classifiers])
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

        return majority_vote

# Exemplo de uso
# Suponha que você tenha um fluxo contínuo de dados X e rótulos correspondentes y
# Aqui, o código está assumindo que você já tem um conjunto de dados de treinamento inicial para iniciar os classificadores
X_train_initial, y_train_initial = ...

# Configuração da janela deslizante, número de melhores classificadores e limite da KL-divergência
window_size = 100
k_best_classifiers = 3
kl_threshold = 0.5  # Ajuste conforme necessário

# Crie uma instância do SlidingWindowEnsemble
ensemble = SlidingWindowEnsemble(window_size, k_best_classifiers, kl_threshold)

# Inicialize os classificadores com o conjunto de dados de treinamento inicial
for i in range(k_best_classifiers):
    ensemble.train_classifier(X_train_initial, y_train_initial)

# Fluxo contínuo de dados
for new_data_point, new_label in zip(stream_of_data_X, stream_of_data_y):
    # Atualize o ensemble com novos dados
    ensemble.predict(new_data_point)

    # Treine um novo classificador com os novos dados
    ensemble.train_classifier(new_data_point, new_label)

############## terceira versão
# Para monitorar a entropia em cada atributo e atualizar o ensemble somente
# para as árvores que excedem um limite estipulado de KL-divergência para 
# algum atributo específico, você pode fazer algumas alterações no código. 
# Aqui está uma versão modificada:
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import numpy as np

class SlidingWindowEnsemble:
    def __init__(self, window_size, k_best_classifiers, kl_threshold):
        self.window_size = window_size
        self.k_best_classifiers = k_best_classifiers
        self.kl_threshold = kl_threshold
        self.classifiers = []
        self.data_window = deque(maxlen=window_size)

    def train_classifier(self, X_train, y_train):
        # Treine um novo classificador (árvore de decisão neste exemplo)
        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)
        self.classifiers.append({'classifier': classifier, 'last_entropies': np.zeros(X_train.shape[1])})

    def calculate_entropy(self, window):
        # Calcula a entropia de Shannon para cada atributo
        entropies = np.array([entropy(np.bincount(window[:, i]) / len(window)) for i in range(window.shape[1])])
        return entropies

    def kl_divergence(self, p, q):
        # Calcula a divergência de Kullback-Leibler entre duas distribuições de probabilidade
        return np.sum(np.where(p != 0, p * np.log2(p / q), 0))

    def update_ensemble(self):
        # Mantenha apenas os k melhores classificadores
        if len(self.classifiers) > self.k_best_classifiers:
            entropies = self.calculate_entropy(np.vstack(self.data_window))

            # Calcule a KL-divergência entre as últimas entropias e as anteriores
            for i in range(len(self.classifiers)):
                kl_divergence = self.kl_divergence(entropies, self.classifiers[i]['last_entropies'])

                # Se a KL-divergência for maior que o limite, atualize a árvore correspondente
                if kl_divergence > self.kl_threshold:
                    X_train_window = np.vstack(list(self.data_window))
                    y_train_window = np.array([clf.predict(X_train_window) for clf in self.classifiers])
                    y_train_window = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=y_train_window)

                    self.classifiers[i]['classifier'].fit(X_train_window, y_train_window)
                    self.classifiers[i]['last_entropies'] = entropies

    def predict(self, X):
        # Atualize a janela de dados e a lista de classificadores
        self.data_window.append(X)
        self.update_ensemble()

        # Faça previsões usando a votação dos classificadores
        predictions = np.array([clf['classifier'].predict(X) for clf in self.classifiers])
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

        return majority_vote

# Exemplo de uso
# Suponha que você tenha um fluxo contínuo de dados X e rótulos correspondentes y
# Aqui, o código está assumindo que você já tem um conjunto de dados de treinamento inicial para iniciar os classificadores
X_train_initial, y_train_initial = ...

# Configuração da janela deslizante, número de melhores classificadores e limite da KL-divergência
window_size = 100
k_best_classifiers = 3
kl_threshold = 0.5  # Ajuste conforme necessário

# Crie uma instância do SlidingWindowEnsemble
ensemble = SlidingWindowEnsemble(window_size, k_best_classifiers, kl_threshold)

# Inicialize os classificadores com o conjunto de dados de treinamento inicial
for i in range(k_best_classifiers):
    ensemble.train_classifier(X_train_initial, y_train_initial)

# Fluxo contínuo de dados
for new_data_point, new_label in zip(stream_of_data_X, stream_of_data_y):
    # Atualize o ensemble com novos dados
    ensemble.predict(new_data_point)

    # Treine um novo classificador com os novos dados
    ensemble.train_classifier(new_data_point, new_label)

############## Quarta
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import numpy as np
from collections import deque

class DivergenceMeasures:
    def __init__(self):
        pass

    @staticmethod
    def calculate_entropy(window):
        entropies = np.array([entropy(np.bincount(window[:, i]) / len(window)) for i in range(window.shape[1])])
        return entropies

    @staticmethod
    def kl_divergence(p, q):
        return np.sum(np.where(p != 0, p * np.log2(p / q), 0))

class SlidingWindowEnsemble:
    def __init__(self, window_size, k_best_classifiers, kl_threshold, divergence_measures):
        self.window_size = window_size
        self.k_best_classifiers = k_best_classifiers
        self.kl_threshold = kl_threshold
        self.classifiers = []
        self.data_window = deque(maxlen=window_size)
        self.divergence_measures = divergence_measures

    def train_classifier(self, X_train, y_train):
        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)
        self.classifiers.append({'classifier': classifier, 'last_entropies': np.zeros(X_train.shape[1])})

    def update_ensemble(self):
        if len(self.classifiers) > self.k_best_classifiers:
            entropies = self.divergence_measures.calculate_entropy(np.vstack(self.data_window))

            for i in range(len(self.classifiers)):
                kl_divergence = self.divergence_measures.kl_divergence(entropies, self.classifiers[i]['last_entropies'])

                if kl_divergence > self.kl_threshold:
                    X_train_window = np.vstack(list(self.data_window))
                    y_train_window = np.array([clf.predict(X_train_window) for clf in self.classifiers])
                    y_train_window = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=y_train_window)

                    self.classifiers[i]['classifier'].fit(X_train_window, y_train_window)
                    self.classifiers[i]['last_entropies'] = entropies

    def predict(self, X):
        self.data_window.append(X)
        self.update_ensemble()

        predictions = np.array([clf['classifier'].predict(X) for clf in self.classifiers])
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

        return majority_vote

# Exemplo de uso
X_train_initial, y_train_initial = ...

window_size = 100
k_best_classifiers = 3
kl_threshold = 0.5

divergence_measures = DivergenceMeasures()

ensemble = SlidingWindowEnsemble(window_size, k_best_classifiers, kl_threshold, divergence_measures)

for i in range(k_best_classifiers):
    ensemble.train_classifier(X_train_initial, y_train_initial)

# Fluxo contínuo de dados
for new_data_point, new_label in zip(stream_of_data_X, stream_of_data_y):
    ensemble.predict(new_data_point)
    ensemble.train_classifier(new_data_point, new_label)

########### Quinta
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import numpy as np
from collections import deque

class DivergenceMeasures:
    def __init__(self):
        pass

    @staticmethod
    def calculate_entropy(window):
        entropies = np.array([entropy(np.bincount(window[:, i]) / len(window)) for i in range(window.shape[1])])
        return entropies

    @staticmethod
    def kl_divergence(p, q):
        return np.sum(np.where(p != 0, p * np.log2(p / q), 0))

    @staticmethod
    def js_divergence(p, q):
        m = 0.5 * (p + q)
        return 0.5 * (DivergenceMeasures.kl_divergence(p, m) + DivergenceMeasures.kl_divergence(q, m))

    @staticmethod
    def hellinger_distance(p, q):
        return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))

    @staticmethod
    def total_variation(p, q):
        return 0.5 * np.sum(np.abs(p - q))

    @staticmethod
    def chi_square_divergence(p, q):
        return 0.5 * np.sum(((p - q)**2) / (p + q))

class SlidingWindowEnsemble:
    def __init__(self, window_size, k_best_classifiers, kl_threshold, divergence_measures):
        self.window_size = window_size
        self.k_best_classifiers = k_best_classifiers
        self.kl_threshold = kl_threshold
        self.classifiers = []
        self.data_window = deque(maxlen=window_size)
        self.divergence_measures = divergence_measures

    def train_classifier(self, X_train, y_train):
        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)
        self.classifiers.append({'classifier': classifier, 'last_entropies': np.zeros(X_train.shape[1])})

    def update_ensemble(self):
        if len(self.classifiers) > self.k_best_classifiers:
            entropies = self.divergence_measures.calculate_entropy(np.vstack(self.data_window))

            for i in range(len(self.classifiers)):
                kl_divergence = self.divergence_measures.kl_divergence(entropies, self.classifiers[i]['last_entropies'])

                if kl_divergence > self.kl_threshold:
                    X_train_window = np.vstack(list(self.data_window))
                    y_train_window = np.array([clf.predict(X_train_window) for clf in self.classifiers])
                    y_train_window = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=y_train_window)

                    self.classifiers[i]['classifier'].fit(X_train_window, y_train_window)
                    self.classifiers[i]['last_entropies'] = entropies

    def predict(self, X):
        self.data_window.append(X)
        self.update_ensemble()

        predictions = np.array([clf['classifier'].predict(X) for clf in self.classifiers])
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

        return majority_vote

# Exemplo de uso
X_train_initial, y_train_initial = ...

window_size = 100
k_best_classifiers = 3
kl_threshold = 0.5

divergence_measures = DivergenceMeasures()

ensemble = SlidingWindowEnsemble(window_size, k_best_classifiers, kl_threshold, divergence_measures)

for i in range(k_best_classifiers):
    ensemble.train_classifier(X_train_initial, y_train_initial)

# Fluxo contínuo de dados
for new_data_point, new_label in zip(stream_of_data_X, stream_of_data_y):
    ensemble.predict(new_data_point)
    ensemble.train_classifier(new_data_point, new_label)


##### Sexta
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import numpy as np
from collections import deque

class DivergenceMeasures:
    def __init__(self):
        pass

    @staticmethod
    def calculate_entropy(window):
        entropies = np.array([entropy(np.bincount(window[:, i]) / len(window)) for i in range(window.shape[1])])
        return entropies

    @staticmethod
    def kl_divergence(p, q):
        return np.sum(np.where(p != 0, p * np.log2(p / q), 0))

    @staticmethod
    def js_divergence(p, q):
        m = 0.5 * (p + q)
        return 0.5 * (DivergenceMeasures.kl_divergence(p, m) + DivergenceMeasures.kl_divergence(q, m))

    @staticmethod
    def hellinger_distance(p, q):
        return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))

    @staticmethod
    def total_variation(p, q):
        return 0.5 * np.sum(np.abs(p - q))

    @staticmethod
    def chi_square_divergence(p, q):
        return 0.5 * np.sum(((p - q)**2) / (p + q))

class SlidingWindowEnsemble:
    def __init__(self, window_size, k_best_classifiers, kl_threshold, divergence_measures, n_estimators=10):
        self.window_size = window_size
        self.k_best_classifiers = k_best_classifiers
        self.kl_threshold = kl_threshold
        self.n_estimators = n_estimators
        self.classifiers = []
        self.data_window = deque(maxlen=window_size)
        self.divergence_measures = divergence_measures

    def _fit_base_classifier(self, X_train, y_train):
        # Método interno para treinar um classificador de base (árvore de decisão neste exemplo)
        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)
        return classifier

    def fit(self, X_train, y_train):
        for _ in range(self.n_estimators):
            classifier = self._fit_base_classifier(X_train, y_train)
            self.classifiers.append({'classifier': classifier, 'last_entropies': np.zeros(X_train.shape[1])})

    def update_ensemble(self):
        if len(self.classifiers) > self.k_best_classifiers:
            entropies = self.divergence_measures.calculate_entropy(np.vstack(self.data_window))

            for i in range(len(self.classifiers)):
                kl_divergence = self.divergence_measures.kl_divergence(entropies, self.classifiers[i]['last_entropies'])

                if kl_divergence > self.kl_threshold:
                    X_train_window = np.vstack(list(self.data_window))
                    y_train_window = np.array([clf.predict(X_train_window) for clf in self.classifiers])
                    y_train_window = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=y_train_window)

                    self.classifiers[i]['classifier'].fit(X_train_window, y_train_window)
                    self.classifiers[i]['last_entropies'] = entropies

    def predict(self, X):
        self.data_window.append(X)
        self.update_ensemble()

        predictions = np.array([clf['classifier'].predict(X) for clf in self.classifiers])
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

        return majority_vote

# Exemplo de uso
X_train_initial, y_train_initial = ...

window_size = 100
k_best_classifiers = 3
kl_threshold = 0.5
n_estimators = 10

divergence_measures = DivergenceMeasures()

ensemble = SlidingWindowEnsemble(window_size, k_best_classifiers, kl_threshold, divergence_measures, n_estimators)

# Treinamento inicial
ensemble.fit(X_train_initial, y_train_initial)

# Fluxo contínuo de dados
for new_data_point, new_label in zip(stream_of_data_X, stream_of_data_y):
    ensemble.predict(new_data_point)
    # Atualização incremental do conjunto de treinamento
    ensemble.fit(new_data_point.reshape(1, -1), np.array([new_label]))



############### Sétimo
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import numpy as np
from collections import deque

class DivergenceMeasures:
    def __init__(self):
        pass

    @staticmethod
    def calculate_entropy(window):
        entropies = np.array([entropy(np.bincount(window[:, i]) / len(window)) for i in range(window.shape[1])])
        return entropies

    @staticmethod
    def kl_divergence(p, q):
        return np.sum(np.where(p != 0, p * np.log2(p / q), 0))

    @staticmethod
    def js_divergence(p, q):
        m = 0.5 * (p + q)
        return 0.5 * (DivergenceMeasures.kl_divergence(p, m) + DivergenceMeasures.kl_divergence(q, m))

    @staticmethod
    def hellinger_distance(p, q):
        return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))

    @staticmethod
    def total_variation(p, q):
        return 0.5 * np.sum(np.abs(p - q))

    @staticmethod
    def chi_square_divergence(p, q):
        return 0.5 * np.sum(((p - q)**2) / (p + q))

class SlidingWindowEnsemble:
    def __init__(self, window_size, k_best_classifiers, kl_threshold, divergence_measures, n_estimators=10):
        self.window_size = window_size
        self.k_best_classifiers = k_best_classifiers
        self.kl_threshold = kl_threshold
        self.n_estimators = n_estimators
        self.classifiers = []
        self.data_window = deque(maxlen=window_size)
        self.divergence_measures = divergence_measures

    def _fit_base_classifier(self, X_train, y_train):
        # Método interno para treinar um classificador de base (árvore de decisão neste exemplo)
        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)
        return classifier

    def fit(self, X_train, y_train):
        for _ in range(self.n_estimators):
            # Amostragem bootstrap
            indices = np.random.choice(len(X_train), len(X_train), replace=True)
            X_train_bootstrap, y_train_bootstrap = X_train[indices], y_train[indices]

            classifier = self._fit_base_classifier(X_train_bootstrap, y_train_bootstrap)
            self.classifiers.append({'classifier': classifier, 'last_entropies': np.zeros(X_train_bootstrap.shape[1])})

    def update_ensemble(self):
        if len(self.classifiers) > self.k_best_classifiers:
            entropies = self.divergence_measures.calculate_entropy(np.vstack(self.data_window))

            for i in range(len(self.classifiers)):
                kl_divergence = self.divergence_measures.kl_divergence(entropies, self.classifiers[i]['last_entropies'])

                if kl_divergence > self.kl_threshold:
                    X_train_window = np.vstack(list(self.data_window))
                    y_train_window = np.array([clf.predict(X_train_window) for clf in self.classifiers])
                    y_train_window = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=y_train_window)

                    self.classifiers[i]['classifier'].fit(X_train_window, y_train_window)
                    self.classifiers[i]['last_entropies'] = entropies

    def predict(self, X):
        self.data_window.append(X)
        self.update_ensemble()

        predictions = np.array([clf['classifier'].predict(X) for clf in self.classifiers])
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

        return majority_vote

# Exemplo de uso
X_train_initial, y_train_initial = ...

window_size = 100
k_best_classifiers = 3
kl_threshold = 0.5
n_estimators = 10

divergence_measures = DivergenceMeasures()

ensemble = SlidingWindowEnsemble(window_size, k_best_classifiers, kl_threshold, divergence_measures, n_estimators)

# Treinamento inicial
ensemble.fit(X_train_initial, y_train_initial)

# Fluxo contínuo de dados
for new_data_point, new_label in zip(stream_of_data_X, stream_of_data_y):
    ensemble.predict(new_data_point)
    # Atualização incremental do conjunto de treinamento
    ensemble.fit(np.array([new_data_point]), np.array([new_label]))

#################### Oitavo
import random
import secrets
import numpy as np
from typing import Union, List

def get_sample_data(X, y, seed=100, bootstrap_size=None,
                    bootstrap_feature_size='auto', sample_n_feature='sqrt-',
                    sample_strategy='numpy.random.Generator.integers'):
    n_sample, n_features = X.shape
    chosen_features = random.sample(range(n_features), min(get_sample_n_feature(n_features, sample_n_feature), n_features))

    if bootstrap_size is None:
        bootstrap_size = n_sample
    bootstrap_instance_size = calc_bootstrap_size(bootstrap_size, bootstrap_feature_size)
    chosen_instances = generate_sample_indices(seed, n_sample, bootstrap_instance_size, sample_strategy=sample_strategy)
    
    X_sample = X[chosen_instances][:, chosen_features]
    y_sample = y[chosen_instances]

    return X_sample, y_sample, chosen_features, chosen_instances

def generate_sample_indices(seed, n_samples, n_samples_bootstrap,
                            sample_strategy='numpy.random.Generator.integers',
                            sample_weight=None, replace=True,
                            probabilities=None, endpoint=False,
                            axis=0, shuffle=True) -> List[int]:
    sample_indices = []

    strategies = {
        'numpy.random.RandomState.randint': np.random.RandomState(seed).randint,
        'numpy.random.RandomState.choice': lambda a, size, replace, p: np.random.RandomState(seed).choice(a, size=size, replace=replace, p=p),
        'numpy.random.Generator.integers': lambda low, high, size, endpoint: np.random.default_rng(seed).integers(low=low, high=high, size=size, endpoint=endpoint),
        'numpy.random.Generator.choice': lambda a, size, replace, p, axis, shuffle: np.random.default_rng(seed).choice(a, size=size, replace=replace, p=p, axis=axis, shuffle=shuffle),
        'random.randrange': lambda lim: random.randrange(0, lim),
        'random.choices': lambda a, weights, k: random.choices(a, weights=weights, k=k),
        'random.sample': lambda a, k: random.sample(a, k=k),
        'secrets.choice': lambda a: secrets.choice(a),
        'secrets.randbelow': lambda lim: secrets.randbelow(lim),
    }

    if sample_strategy in strategies:
        sample_indices = strategies[sample_strategy](n_samples, n_samples_bootstrap, replace, probabilities)
    else:
        raise ValueError('Random sampling strategy unrecognized')

    return sample_indices

def get_sample_n_feature(n_features, sample_n_feature='sqrt-') -> int:
    sample_n_feature_mapping = {
        'sqrt-': int(np.sqrt(n_features)),
        'sqrt+': int(np.sqrt(n_features) + 0.5),
        'log2+': int(np.log2(n_features) + 0.5),
        'log2-': int(np.log2(n_features)),
        'log10+': int(np.log10(n_features) + 0.5),
        'log10-': int(np.log10(n_features)),
        'auto': n_features,
    }

    if isinstance(sample_n_feature, (int, float)):
        return int(sample_n_feature * n_features)
    return sample_n_feature_mapping.get(sample_n_feature, None)

def calc_bootstrap_size(size, sample_bootstrap_size='auto') -> int:
    if sample_bootstrap_size == 'auto':
        return size
    if isinstance(sample_bootstrap_size, (int, float)):
        return int(size * sample_bootstrap_size)
    raise ValueError('Impossible to determine bootstrap size.')

# Exemplo de uso
X, y = np.random.rand(100, 5), np.random.randint(2, size=100)
get_sample_data(X, y)