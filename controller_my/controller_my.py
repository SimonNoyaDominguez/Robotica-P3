from controller import Robot, Motor, DistanceSensor
import random
import numpy as np
import math

# Constantes
TIME_STEP = 32
MAX_SPEED = 10.0  # Velocidad máxima
INITIAL_LEARNING_RATE = 0.3  # Tasa de aprendizaje inicial
DISCOUNT_FACTOR = 0.9  # Mayor valor para considerar mejor las recompensas futuras
ROTATION_FACTOR = 0.7  # Factor para giros menos bruscos

# Variable para controlar la iteración en la que se detiene el aprendizaje
MAX_LEARNING_ITERATIONS = 750  # Puedes modificar este valor según necesites

# Variables para el aprendizaje
last_display_second = 0
mat_q = np.zeros((3, 3))  # Inicializar con valores neutros
visitas = np.zeros((3, 3))  # Matriz para contar visitas a cada par estado-acción
sensors_hist = []  # Historial de lecturas de sensores
total_iterations = 0  # Contador global de iteraciones
q_learning_iterations = 0  # Contador específico de actualizaciones Q-learning
recent_states = []  # Historial de estados recientes

# Inicialización del robot
robot = Robot()

# Obtener motores
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Sensores infrarrojos (incluidos los del suelo)
infrared_sensors = []
infrared_sensors_names = [
    # sensores de torreta
    "rear left infrared sensor", 
    "left infrared sensor", 
    "front left infrared sensor", 
    "front infrared sensor",
    "front right infrared sensor", 
    "right infrared sensor", 
    "rear right infrared sensor", 
    "rear infrared sensor",
    # sensores de suelo
    "ground left infrared sensor", 
    "ground front left infrared sensor", 
    "ground front right infrared sensor",
    "ground right infrared sensor"
]

for name in infrared_sensors_names:
    sensor = robot.getDevice(name)
    sensor.enable(TIME_STEP)
    infrared_sensors.append(sensor)

# Inicialización para exploración aleatoria al principio
estado_actual = 2
accion_actual = 0

def check_sensors():
    """Obtiene los valores de los sensores de suelo"""
    return [infrared_sensors[8].getValue(), infrared_sensors[9].getValue(),
            infrared_sensors[10].getValue(), infrared_sensors[11].getValue()]

def getEstado(sensor_values):
    """Determina el estado basado en los valores de los sensores
    Estado 0: Khepera abandona línea por la izquierda
    Estado 1: Khepera abandona línea por la derecha
    Estado 2: Resto de casos
    """
    # Sensor izquierdo (ground_front_left) detecta línea pero derecho (ground_front_right) no
    if sensor_values[1] > 750 and sensor_values[3] < 500:
        return 0  # Abandona línea por la izquierda (solo sensor izquierdo ve línea)
    
    # Sensor derecho (ground_front_right) detecta línea pero izquierdo (ground_front_left) no
    elif sensor_values[2] > 750 and sensor_values[0] < 500:
        return 1  # Abandona línea por la derecha (solo sensor derecho ve línea)
    
    # En otro caso (ambos sensores detectan línea o ninguno la detecta)
    return 2  # Resto de casos

def check_refuerzo(new_sensor_values, prev_sensor_values):
    # Calcula la fuerza de línea para los sensores centrales
    new_on_line_strength = sum(1 for v in new_sensor_values[1:3] if v > 750)
    prev_on_line_strength = sum(1 for v in prev_sensor_values[1:3] if v > 750)
    # Refuerzo usando las fuerzas de línea
    if prev_on_line_strength == 0 and new_on_line_strength == 0:
        return 1
    elif prev_on_line_strength == 0 and new_on_line_strength > 0:
        return -1
    elif prev_on_line_strength == 2 and new_on_line_strength == 2:
        return -1
    elif prev_on_line_strength == 2 and new_on_line_strength < 2:
        return 1
    elif prev_on_line_strength < new_on_line_strength:
        return -1
    else:
        return 1

# Modificar la función actualizar_refuerzo
def actualizar_refuerzo(refuerzo, action, prev_estado, nuevo_estado):
    """Actualiza la matriz Q usando la ecuación de Q-learning que considera visitas"""
    global visitas, q_learning_iterations, total_iterations
    
    # Incrementar contador de iteraciones Q-learning
    q_learning_iterations += 1
    
    # Si ya alcanzamos el número de iteraciones elegida, no actualizar la matriz Q
    if total_iterations >= MAX_LEARNING_ITERATIONS:
        print(f"Matriz Q congelada (iteración {total_iterations}). Ya no se actualizan los valores.")
        return
    
    # Incrementar contador de visitas para este par estado-acción
    visitas[prev_estado][action] += 1
    
    # Tasa de aprendizaje adaptativa basada en el número de visitas
    # Decrece con más visitas para dar más peso a la experiencia acumulada
    learning_rate = INITIAL_LEARNING_RATE / (1 + 0.1 * math.log(1 + visitas[prev_estado][action]))
    
    # Fórmula Q-learning estándar con tasa de aprendizaje adaptativa
    mat_q[prev_estado][action] = mat_q[prev_estado][action] + learning_rate * (
        refuerzo + DISCOUNT_FACTOR * np.max(mat_q[nuevo_estado]) - mat_q[prev_estado][action]
    )
    
    # Imprimir información sobre la iteración actual
    print(f"Q-Learning Iteración #{q_learning_iterations}")
    print(f"Learning rate para estado {prev_estado}, acción {action}: {learning_rate:.4f}")
    
    # Imprimir estadísticas cada 100 iteraciones de Q-learning
    if q_learning_iterations % 100 == 0:
        print(f"\n==== ESTADÍSTICAS DE APRENDIZAJE (Iteración Q #{q_learning_iterations}) ====")
        print(f"Matriz Q actual:\n{mat_q}")
        print(f"Visitas a pares estado-acción:\n{visitas}")
        
        # Calcular algunas estadísticas útiles
        max_q = np.max(mat_q)
        min_q = np.min(mat_q)
        mean_q = np.mean(mat_q)
        policy = np.argmax(mat_q, axis=1)
        
        print(f"Valor Q máximo: {max_q:.4f}, mínimo: {min_q:.4f}, promedio: {mean_q:.4f}")
        print(f"Política actual: Estado 0 → Acción {policy[0]}, Estado 1 → Acción {policy[1]}, Estado 2 → Acción {policy[2]}")
        print(f"==========================================================\n")

    # Si estamos cerca de congelar la matriz, guardar una copia
    if total_iterations == MAX_LEARNING_ITERATIONS / 2:
        print("\n¡ATENCIÓN! Matriz Q congelada a partir de ahora.")
        print(f"Matriz Q final:\n{mat_q}")
        print(f"Política final: Estado 0 → Acción {np.argmax(mat_q[0])}, Estado 1 → Acción {np.argmax(mat_q[1])}, Estado 2 → Acción {np.argmax(mat_q[2])}")

def pick_action(estado):
    """Estrategia epsilon-greedy con epsilon decreciente linealmente hasta la mitad de iteraciones máximas, luego solo explotación"""
    global total_iterations

    # Epsilon decrece linealmente de 1.0 a 0.0 en la primera mitad de las iteraciones
    if total_iterations < MAX_LEARNING_ITERATIONS / 2:
        current_epsilon = 1.0 - (total_iterations / (MAX_LEARNING_ITERATIONS / 2))
    else:
        current_epsilon = 0.0

    # Imprimir el valor actual de epsilon cada 100 iteraciones
    if total_iterations % 100 == 0:
        print(f"Epsilon actual: {current_epsilon:.4f}")

    # Incrementar contador global
    total_iterations += 1

    # Estrategia epsilon-greedy
    if random.random() < current_epsilon:
        # Exploración: elegir acción aleatoria, ponderada por visitas
        probs = []
        for a in range(3):  # 3 acciones posibles
            probs.append(1.0 / (1.0 + visitas[estado][a]))
        sum_probs = sum(probs)
        if sum_probs > 0:
            probs = [p / sum_probs for p in probs]
            return random.choices([0, 1, 2], weights=probs, k=1)[0]
        else:
            return random.randint(0, 2)
    else:
        # Explotación: elegir mejor acción según matriz Q
        return np.argmax(mat_q[estado])

def perform_action(action):
    """Ejecuta la acción seleccionada"""
    if action == 0:  # Girar derecha
        left_motor.setVelocity(MAX_SPEED)
        right_motor.setVelocity(-MAX_SPEED * ROTATION_FACTOR)
    elif action == 1:  # Girar izquierda
        left_motor.setVelocity(-MAX_SPEED * ROTATION_FACTOR)
        right_motor.setVelocity(MAX_SPEED)
    else:  # Avanzar recto
        left_motor.setVelocity(MAX_SPEED)
        right_motor.setVelocity(MAX_SPEED)

def avoid_obstacles():
    """Detecta y evita obstáculos"""
    if (infrared_sensors[2].getValue() > 300 or 
        infrared_sensors[3].getValue() > 300 or 
        infrared_sensors[4].getValue() > 300):
        speed_offset = 0.2 * (MAX_SPEED - 0.03 * infrared_sensors[3].getValue())
        speed_delta = 0.03 * infrared_sensors[2].getValue() - 0.03 * infrared_sensors[4].getValue()
        left_motor.setVelocity(speed_offset + speed_delta)
        right_motor.setVelocity(speed_offset - speed_delta)
        return True
    return False

# Inicialización para velocidad constante al inicio
left_motor.setVelocity(MAX_SPEED)
right_motor.setVelocity(MAX_SPEED)

# Bucle principal
while robot.step(TIME_STEP) != -1:
    display_second = robot.getTime()
    
    if display_second != last_display_second:
        last_display_second = display_second
        
        # Primero verificamos si hay obstáculos
        if avoid_obstacles():
            # Si estamos evitando obstáculos, saltamos el aprendizaje
            continue
        
        # Obtenemos lecturas de sensores actuales
        sensor_values = check_sensors()
        sensors_hist.append(sensor_values)
        
        # Determinamos el estado actual
        estado_actual = getEstado(sensor_values)
        
        # Guardamos historial de estados recientes
        recent_states.append(estado_actual)
        if len(recent_states) > 8:
            recent_states = recent_states[-8:]

        # Seleccionamos una acción
        accion_actual = pick_action(estado_actual)
        
        # Ejecutamos la acción
        perform_action(accion_actual)
        
        # Esperamos a ver el resultado (no bloqueante)
        robot.step(TIME_STEP)  # Reducido para evitar giros excesivos
        
        # Obtenemos nuevas lecturas de sensores
        new_sensor_values = check_sensors()
        sensors_hist.append(new_sensor_values)
        
        # Si tenemos suficientes lecturas para comparar
        if len(sensors_hist) >= 3:
            # Tomamos lecturas previas para comparación
            prev_sensor_values = sensors_hist[-3]
            
            # Obtenemos el nuevo estado
            nuevo_estado = getEstado(new_sensor_values)
            
            # Calculamos el refuerzo
            refuerzo = check_refuerzo(new_sensor_values, prev_sensor_values)

            # Actualizamos la matriz Q
            actualizar_refuerzo(refuerzo, accion_actual, estado_actual, nuevo_estado)
            
            # Imprimir información para depuración
            print(f"Estado: {estado_actual}, Acción: {accion_actual}, Nuevo estado: {nuevo_estado}, Refuerzo: {refuerzo}")
            print(f"Matriz Q:\n{mat_q}")
            
            # Eliminamos lecturas antiguas para mantener el historial bajo control
            if len(sensors_hist) > 10:
                sensors_hist = sensors_hist[-10:]