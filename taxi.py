from IPython.display import clear_output
from time import sleep
import gym
import numpy as np
import random

# creo el entorno
env = gym.make("Taxi-v2").env
env.reset()  # inicio en un estado aleatorio
q_table = np.zeros([env.observation_space.n, env.action_space.n])
# inicializo la matriz con ceros

# Hyperparameters
alpha = 0.1  # Defino la tasa de aprendizaje
gamma = 0.6  # Defino factor de descuento
epsilon = 1  # define el equilibrio entre la exploracion y la explotacion, al comenzar el algoritmo es necesario mucha exploracion, a medida que paan las iteraciones es necesario poca exploracion
max_epsilon = 1  # probabilidad maxima de exploración
min_epsilon = 0.01  # probabilidad minima de exploración
decay_rate = 0.01  # velocidad en la que disminuye la probabilidad de exploracion

episodiosEntreno = 100000  # se definen los episodios de entrenamiento
pasosMaximo = 500  # numero maximo de pasos

for i in range(1, episodiosEntreno):
    state = env.reset()  # reseteo el estado
    done = False
    paso = 0  # para contar los pasos
    while not done:
        paso = paso + 1
        if paso == pasosMaximo - 1:
            done = True
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore espacio de acciones
        else:
            action = np.argmax(q_table[state])  # Explote los valores aprendidos

        next_state, reward, done, info = env.step(action)  # obtengo el nuevo estado, la recompensa y si termino

        # Actualizar valores de la tabla
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value
        # actualizo los valores de la tabla según  alpha y gamma

        state = next_state
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -decay_rate * i)  # cambio el valor de epsilon para que la probabilidad de exploracion disminuya y aumente la probabilidad de explotación

    print(f"Episode: {i}")

print("--------- Entrenamiento finalizado ---------\n")
print(q_table)
sleep(2)

print('----------------------------------------------------')
print('--------------TEST----------------------------------')
print('--------------Estado Inicial Test-------------------')
env.reset()
env.render()  # creo el esenario de prueba
sleep(3)

total_epochs, penalizaciones = 0, 0
episodes = 1
frames = []
for _ in range(episodes):
    # state = env.reset()
    reward = 0

    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
        })
print(f"Results after {episodes} episodes:")


def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.5)


print_frames(frames)
