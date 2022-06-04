from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn


class NNIris(nn.Module):
    """Klasa sieci neuronowej dla zbioru Iris."""

    def __init__(self, epochs, lr, momentum,
                 input_layer_size, hidden1_layer_size, hidden2_layer_size, output_layer_size, data_set):
        """Konstruktor klasy przyjmujący parametry sieci
                :param input_layer_size: wymiar warstawy wejściowej
                :param hidden1_layer_size: wymiar pierwszej warstawy ukrytej
                :param hidden2_layer_size: wymiar drugiej warstwy ukrytej
                :param output_layer_size wymiar warstwy wyjściowej
                """
        super(NNIris, self).__init__()

        self.input_layer = nn.Linear(input_layer_size, hidden1_layer_size)
        self.hidden1_layer = nn.Linear(hidden1_layer_size, hidden2_layer_size)
        self.hidden2_layer = nn.Linear(hidden2_layer_size, output_layer_size)
        # Deklaracje warstw

        self.data_set = data_set
        self.lr = lr
        self.momentum = momentum
        self.epochs = epochs
        # Deklaracje hiperparametrów

        self.X_train, self.X_test, self.Y_train, self.Y_test = self.__normalize(self.data_set)

        self.criterion = nn.MSELoss()  # Deklaracja kryerium błędu średniokwadratowego
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)  # Optymalizator

        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        # Deklaracje funkcji sigmoidalnych

    def forward(self, input_x):
        """Funkcja realizująca funkcje przejść neuronów przez warstwy funkcją aktywacji sigmoid"""
        out = self.input_layer(input_x)
        out = self.sigmoid1(out)
        out = self.hidden1_layer(out)
        out = self.sigmoid2(out)
        out = self.hidden2_layer(out)
        return out

    @staticmethod
    def __normalize(data_set):
        """Funkcja normalizująca przekazane dane"""
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = data_set.data
        target = data_set.target
        X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2, random_state=1022)
        X_train = scaler.fit_transform(X_train)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        Y_train = torch.tensor(Y_train, dtype=torch.float32)
        Y_test = torch.tensor(Y_test, dtype=torch.float32)

        return X_train, X_test, Y_train, Y_test

    def learn(self):
        """Funkcja realizujaca uczenie się sieci"""
        loss_trains_list = list()
        loss_tests_list = list()

        with open(file='plot_3d.csv', mode='w') as file:
            for i in range(0, self.epochs):
                file.write(f'{i}.0')
                file.write(',')

            file.write('\n')
            self.train()  # Ustawienie modelu w tryb uczenia

            for epoch in range(0, self.epochs):
                file.write(f'{epoch}.0')
                file.write(',')

                loss_train = 0.0
                loss_tests = 0.0

                for i in range(len(self.X_train)):
                    self.optimizer.zero_grad()  # Zerowanie gradientów
                    outputs = self(self.X_train[i])  # Przejście przez funkcję aktywacji
                    loss = self.criterion(outputs, self.Y_train[i])  # Oblicza błąd

                    loss.backward()  # Obliczanie gradientów
                    self.optimizer.step()  # Wykonanie akutalizacji na podstawie akutalnego gradientu
                    loss_train += loss.item()


                for i in range(len(self.X_test)):
                    output_test = self(self.X_test[i])
                    loss_test = self.criterion(output_test, self.Y_test[i])
                    loss_tests += loss_test.item()

                    file.write(str(loss_tests / (i + 1) * 100))
                    file.write(',')

                file.write('\n')
                print(
                    f'Epoch: {epoch}, Loss: {loss_train / len(self.X_train)}, Test Loss: {loss_tests / len(self.X_test)}')

                loss_trains_list.append(loss_train / len(self.X_train))
                loss_tests_list.append(loss_tests / len(self.X_test))

            self.__draw_plot_2d_epoch_loss(loss_tests_list, loss_trains_list)

    def __draw_plot_2d_epoch_loss(self, losses_test, losses_train):
        """Rysowanie wykresów 2d"""
        plt.plot([epoch for epoch in range(self.epochs)], losses_train, label='Train Losse')
        plt.plot([epoch for epoch in range(self.epochs)], losses_test, label='Test Loss')
        plt.legend(loc="upper right")
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

    @staticmethod
    def draw_plot_3d_test_loss_epoch():
        """Rysowanie wykresu 3D"""
        z_data = pd.read_csv('plot_3d.csv', index_col=0)
        z = z_data.values
        sh_0, sh_1 = z.shape
        x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)

        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])

        fig.update_layout(title='Testy strat podczas uczenia', autosize=False,
                          width=1770, height=930,
                          margin=dict(l=65, r=50, b=65, t=90), )

        fig.show()

    def predict(self):
        """Funkcja testująca nauczoną sieć

        :return float: Zwraca precyzję sieci"""

        self.eval()  # Wyłączenie trybu uczenia się modelu

        with torch.no_grad():
            out = self(self.X_test)
            _, predicted = torch.max(out, 1)
            train_acc = torch.sum(predicted == self.Y_test)

        return f'Accuracy: {round((train_acc.item() / len(self.Y_test)) * 100)}%'


if __name__ == '__main__':
    iris_dataset = datasets.load_iris()

    # Eksperymet 1
    network = NNIris(epochs=100, lr=0.01, momentum=0.01,
                     input_layer_size=4, hidden1_layer_size=11, hidden2_layer_size=5, output_layer_size=3,
                     data_set=iris_dataset)

    network.learn()
    print(network.predict())
    network.draw_plot_3d_test_loss_epoch()

    # Eksperyment 2
    network = NNIris(epochs=200, lr=0.1, momentum=0.01,
                     input_layer_size=4, hidden1_layer_size=50, hidden2_layer_size=30, output_layer_size=3,
                     data_set=iris_dataset)

    network.learn()
    print(network.predict())
    network.draw_plot_3d_test_loss_epoch()

    # Eksperyment 3
    network = NNIris(epochs=500, lr=0.1, momentum=0.8,
                     input_layer_size=4, hidden1_layer_size=50, hidden2_layer_size=30, output_layer_size=3,
                     data_set=iris_dataset)
    network.learn()
    print(network.predict())
    network.draw_plot_3d_test_loss_epoch()

    # Eksperyment 4
    network = NNIris(epochs=200, lr=0.01, momentum=0.01,
                     input_layer_size=4, hidden1_layer_size=10, hidden2_layer_size=5, output_layer_size=3,
                     data_set=iris_dataset)
    network.learn()
    print(network.predict())
    network.draw_plot_3d_test_loss_epoch()
