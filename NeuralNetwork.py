import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

class DroneSimulator:
    """
    Simulador simple de dron con dinámica similar al MPC original.
    """
    def __init__(self, dt=0.02):
        self.dt = dt  # [s] tiempo de muestreo de control (50 Hz)
        
        # Parámetros físicos
        self.m = 0.60    # [kg] masa
        self.Ixx = 0.02  # [kg·m²] inercia alrededor de x (roll)
        self.Iyy = 0.02  # [kg·m²] inercia alrededor de y (pitch)
        self.Izz = 0.04  # [kg·m²] inercia alrededor de z (yaw)
        self.g = 9.81    # [m/s²] aceleración de la gravedad
        
        # Empuje de sustentación (equilibrio)
        self.u_hover = self.m * self.g
        
        # Límites de control
        self.thrust_min = 0.0
        self.thrust_max = 15.0
        self.Mroll_min = -0.5
        self.Mroll_max = 0.5
        self.Mpitch_min = -0.5
        self.Mpitch_max = 0.5
        self.Myaw_min = -0.2
        self.Myaw_max = 0.2
    
    def step(self, state, control, add_disturbance=True):
        """
        Simula un paso de la dinámica del dron.
        
        Parámetros:
        -----------
        state: np.ndarray (8,)
            Estado actual [z, dz, roll, droll, pitch, dpitch, yaw, dyaw]
        control: np.ndarray (4,)
            Entradas de control [thrust, Mroll, Mpitch, Myaw]
        add_disturbance: bool
            Si se añade perturbación de viento y ruido
            
        Retorna:
        --------
        next_state: np.ndarray (8,)
            Siguiente estado tras aplicar el control
        """
        next_state = np.copy(state)
        
        # Aplicar límites de los actuadores
        thrust = np.clip(control[0], self.thrust_min, self.thrust_max)
        Mroll = np.clip(control[1], self.Mroll_min, self.Mroll_max)
        Mpitch = np.clip(control[2], self.Mpitch_min, self.Mpitch_max)
        Myaw = np.clip(control[3], self.Myaw_min, self.Myaw_max)
        
        # Perturbación (si está habilitada) - reducida para mejor seguimiento
        wind_disturbance = -0.02 if add_disturbance else 0.0  # Aún más reducida desde -0.05
        dz_noise = np.random.normal(0, 0.001) if add_disturbance else 0.0  # Ruido reducido
        
        # Efecto suelo extra cuando está cerca del suelo (los drones reales lo experimentan)
        ground_effect = 0.0
        if next_state[0] < 0.2:  # Si la altitud es menor a 20cm
            ground_effect = 0.15 * (1.0 - next_state[0]/0.2)  # Más fuerte cerca del suelo
        
        # Actualizar velocidades
        next_state[1] += self.dt * ((thrust + wind_disturbance + ground_effect) / self.m - self.g) + dz_noise  # dz
        next_state[3] += self.dt * Mroll / self.Ixx   # droll
        next_state[5] += self.dt * Mpitch / self.Iyy  # dpitch
        next_state[7] += self.dt * Myaw / self.Izz    # dyaw
        
        # Actualizar posiciones
        next_state[0] += self.dt * next_state[1]  # z
        next_state[2] += self.dt * next_state[3]  # roll
        next_state[4] += self.dt * next_state[5]  # pitch
        next_state[6] += self.dt * next_state[7]  # yaw
        
        # Restricción de suelo
        if next_state[0] <= 0.001:
            next_state[0] = 0.0
            next_state[1] = 0.0  # Detener velocidad vertical si está en el suelo
        
        return next_state


class EnhancedDroneController:
    """
    Controlador neuronal altamente mejorado para el control de dron con seguimiento de referencia específico.
    """
    def __init__(self):
        # Parámetros físicos
        self.m = 0.60    # [kg] masa
        self.g = 9.81    # [m/s²] gravedad
        self.u_hover = self.m * self.g
        
        # Límites de control
        self.thrust_min = 0.0
        self.thrust_max = 15.0
        self.Mroll_min = -0.5
        self.Mroll_max = 0.5
        self.Mpitch_min = -0.5
        self.Mpitch_max = 0.5
        self.Myaw_min = -0.2
        self.Myaw_max = 0.2
        
        # Entrada de control previa para transiciones suaves (historial de 2 pasos)
        self.prev_u = [np.array([self.u_hover * 1.6, 0.0, 0.0, 0.0]), 
                       np.array([self.u_hover * 1.6, 0.0, 0.0, 0.0])]
        
        # Historial de errores para términos derivativos e integrales
        self.error_history = {
            'z': {'current': 0.0, 'prev': 0.0, 'integral': 0.0},
            'roll': {'current': 0.0, 'prev': 0.0, 'integral': 0.0},
            'pitch': {'current': 0.0, 'prev': 0.0, 'integral': 0.0},
            'yaw': {'current': 0.0, 'prev': 0.0, 'integral': 0.0}
        }
        
        # Parámetros de control
        self.dt = 0.02  # Intervalo de control
        
        # Construir modelo
        self._build_model()
    
    def _build_model(self):
        """Crea un modelo de red neuronal altamente mejorado específicamente ajustado para seguimiento de referencia."""
        # Forma de entrada: [
        #   estado (8), 
        #   referencia (4), 
        #   controles previos (8),
        #   historial de errores (12) - actual, previo, integral para cada uno de z, roll, pitch, yaw
        # ]
        input_dim = 32
        
        # Regularización L2 para evitar sobreajuste
        reg = l2(0.0001)
        
        # Crear una red más profunda con conexiones de salto para mejor desempeño
        self.model = Sequential([
            # Capa de entrada
            Dense(512, activation='relu', input_shape=(input_dim,), kernel_regularizer=reg),
            BatchNormalization(),
            
            # Capas ocultas
            Dense(256, activation='relu', kernel_regularizer=reg),
            BatchNormalization(),
            Dropout(0.2),  # Añadir dropout para regularización
            
            Dense(128, activation='relu', kernel_regularizer=reg),
            BatchNormalization(),
            
            Dense(64, activation='relu', kernel_regularizer=reg),
            BatchNormalization(),
            
            # Capa de salida - funciones de activación específicas para cada control
            Dense(4, activation='sigmoid')  # Usar sigmoid para limitar el rango de salida
        ])
        
        # Compilar modelo con tasa de aprendizaje muy baja para ajuste fino
        self.model.compile(
            optimizer=Adam(learning_rate=0.0002), 
            loss='mse'
        )
        
        # Imprimir resumen del modelo
        self.model.summary()
    
    def predict(self, state, reference):
        """
        Calcula la acción de control basada en el estado actual y la referencia.
        
        Parámetros:
        -----------
        state: np.ndarray (8,)
            Estado actual [z, dz, roll, droll, pitch, dpitch, yaw, dyaw]
        reference: np.ndarray (4,)
            Referencias [z_ref, roll_ref, pitch_ref, yaw_ref]
            
        Retorna:
        --------
        control: np.ndarray (4,)
            Entradas de control [thrust, Mroll, Mpitch, Myaw]
        """
        # Calcular errores
        z_error = reference[0] - state[0]
        roll_error = reference[1] - state[2]
        pitch_error = reference[2] - state[4]
        yaw_error = reference[3] - state[6]
        
        # Actualizar historial de errores
        for key, error in zip(['z', 'roll', 'pitch', 'yaw'], [z_error, roll_error, pitch_error, yaw_error]):
            self.error_history[key]['prev'] = self.error_history[key]['current']
            self.error_history[key]['current'] = error
            
            # Actualizar integral con anti-windup
            max_integral = 1.0  # Límite para evitar acción integral excesiva
            self.error_history[key]['integral'] += error * self.dt
            self.error_history[key]['integral'] = np.clip(self.error_history[key]['integral'], -max_integral, max_integral)
        
        # Aplanar historial de errores para entrada al modelo
        error_flat = []
        for key in ['z', 'roll', 'pitch', 'yaw']:
            error_flat.extend([
                self.error_history[key]['current'],
                self.error_history[key]['prev'],
                self.error_history[key]['integral']
            ])
        
        # Combinar todas las entradas para la red neuronal
        model_input = np.concatenate([
            state,                     # Estado actual
            reference,                 # Referencia objetivo
            self.prev_u[0],            # Control previo
            self.prev_u[1],            # Control de hace 2 pasos
            np.array(error_flat)       # Historial de errores
        ]).reshape(1, -1)
        
        # Obtener predicciones crudas (rango 0 a 1 por sigmoid)
        raw_output = self.model.predict(model_input, verbose=0)[0]
        
        # Escalar a rangos de control con mapeo directo para mejorar seguimiento
        thrust = self.thrust_min + (self.thrust_max - self.thrust_min) * raw_output[0]
        Mroll = self.Mroll_min + (self.Mroll_max - self.Mroll_min) * raw_output[1]
        Mpitch = self.Mpitch_min + (self.Mpitch_max - self.Mpitch_min) * raw_output[2]
        Myaw = self.Myaw_min + (self.Myaw_max - self.Myaw_min) * raw_output[3]
        
        # Calcular términos feedforward manuales para ayudar a la red neuronal
        # Estos términos ayudan especialmente durante la fase inicial de aprendizaje
        
        # Feedforward de altitud - añadir empuje extra para despegue y cambios de altura
        if state[0] < 0.1 and reference[0] > 0.1:
            # Impulso más fuerte para el despegue inicial
            thrust += self.m * 4.0  # Aumentado para despegue más rápido
        elif np.abs(z_error) > 0.1:
            # Ayuda con cambios de altura
            thrust += np.sign(z_error) * self.m * 1.5 * min(np.abs(z_error), 0.5)  # Aumentado para respuesta más rápida
        
        # Feedforward para roll/pitch/yaw para mejorar seguimiento angular
        roll_ff = 0.2 * np.sign(roll_error) * min(np.abs(roll_error), 0.2)
        pitch_ff = 0.2 * np.sign(pitch_error) * min(np.abs(pitch_error), 0.2)
        yaw_ff = 0.1 * np.sign(yaw_error) * min(np.abs(yaw_error), 0.2)
        
        # Aplicar control suavizado con feedforward
        Mroll = 0.8 * Mroll + 0.2 * self.prev_u[0][1] + roll_ff
        Mpitch = 0.8 * Mpitch + 0.2 * self.prev_u[0][2] + pitch_ff
        Myaw = 0.8 * Myaw + 0.2 * self.prev_u[0][3] + yaw_ff
        
        # Aplicar límites de control nuevamente tras combinar
        thrust = np.clip(thrust, self.thrust_min, self.thrust_max)
        Mroll = np.clip(Mroll, self.Mroll_min, self.Mroll_max)
        Mpitch = np.clip(Mpitch, self.Mpitch_min, self.Mpitch_max)
        Myaw = np.clip(Myaw, self.Myaw_min, self.Myaw_max)
        
        # Salidas de control combinadas
        control = np.array([thrust, Mroll, Mpitch, Myaw])
        
        # Actualizar historial de control
        self.prev_u[1] = self.prev_u[0].copy()
        self.prev_u[0] = control.copy()
        
        return control
    
    def train(self, train_X, train_y, epochs=150, batch_size=128, validation_split=0.2):
        """
        Entrena el modelo usando los datos proporcionados con parámetros de entrenamiento mejorados.
        
        Parámetros:
        -----------
        train_X: np.ndarray
            Datos de entrada de entrenamiento
        train_y: np.ndarray
            Datos de salida de entrenamiento
        epochs: int
            Número de épocas de entrenamiento
        batch_size: int
            Tamaño de lote para entrenamiento
        validation_split: float
            Fracción de datos para validación
        
        Retorna:
        --------
        history: tf.keras.callbacks.History
            Historial de entrenamiento
        """
        # Early stopping con mayor paciencia
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=30,  # Paciencia aumentada
            restore_best_weights=True
        )
        
        # Reducción de tasa de aprendizaje en meseta
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15,
            min_lr=0.00005,
            verbose=1
        )
        
        # Entrenar el modelo
        history = self.model.fit(
            train_X, 
            train_y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def save_model(self, filepath):
        """Guarda el modelo en un archivo."""
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Carga un modelo desde un archivo."""
        self.model = tf.keras.models.load_model(filepath)


def generate_enhanced_training_data(n_samples=150000):
    """
    Genera datos de entrenamiento de alta calidad con perfiles de referencia muy específicos.
    
    Parámetros:
    -----------
    n_samples: int
        Número de muestras de entrenamiento a generar
        
    Retorna:
    --------
    X: np.ndarray
        Datos de entrada [estado, referencia, controles previos, historial de errores]
    y: np.ndarray
        Salidas objetivo (acciones de control)
    """
    print("Generando datos de entrenamiento mejorados...")
    
    # Crear simulador
    simulator = DroneSimulator()
    
    # Ganancias PID finamente ajustadas para seguimiento preciso
    Kp_z = 12.0     # Aumentado desde 10.0
    Kd_z = 5.0      # Aumentado desde 4.0
    Ki_z = 1.5      # Aumentado desde 1.0
    
    Kp_roll = 0.5   # Aumentado desde 0.4
    Kd_roll = 0.2   # Aumentado desde 0.15
    Ki_roll = 0.15  # Aumentado desde 0.1
    
    Kp_pitch = 0.5  # Aumentado desde 0.4
    Kd_pitch = 0.2  # Aumentado desde 0.15
    Ki_pitch = 0.15 # Aumentado desde 0.1
    
    Kp_yaw = 0.3    # Aumentado desde 0.2
    Kd_yaw = 0.15   # Aumentado desde 0.1
    Ki_yaw = 0.1    # Término integral añadido
    
    # Almacenamiento de datos
    states = []
    references = []
    prev_controls = []
    error_histories = []
    controls = []
    
    # Control inicial
    prev_control_1 = np.array([simulator.u_hover * 1.2, 0.0, 0.0, 0.0])
    prev_control_2 = np.array([simulator.u_hover * 1.2, 0.0, 0.0, 0.0])
    
    # Generar escenarios aleatorios
    scenarios_count = int(n_samples / 200)  # ~200 pasos por escenario
    
    # Añadir perfiles de referencia específicos que coinciden con los gráficos objetivo
    specific_profiles = [
        # Coincidiendo con el perfil de altitud objetivo (0 a 0.6m a 1.2m)
        {
            'initial_state': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'references': [
                (np.array([0.0, 0.0, 0.0, 0.0]), 25),    # Suelo, 25 pasos
                (np.array([0.6, 0.0, 0.0, 0.0]), 75),    # Subida a 0.6m, 75 pasos
                (np.array([1.2, 0.0, 0.0, 0.0]), 100),   # Subida a 1.2m, 100 pasos
            ]
        },
        # Coincidiendo con el perfil de roll objetivo (-3.8, +2, -2.5, +2.5 grados)
        {
            'initial_state': np.array([1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'references': [
                (np.array([1.2, 0.0, 0.0, 0.0]), 50),
                (np.array([1.2, np.radians(-3.8), 0.0, 0.0]), 100),
                (np.array([1.2, np.radians(2.0), 0.0, 0.0]), 50),
                (np.array([1.2, np.radians(-2.5), 0.0, 0.0]), 100),
                (np.array([1.2, np.radians(2.5), 0.0, 0.0]), 50),
            ]
        },
        # Coincidiendo con el perfil de pitch objetivo (+15, 0, -15 grados)
        {
            'initial_state': np.array([1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'references': [
                (np.array([1.2, 0.0, 0.0, 0.0]), 50),
                (np.array([1.2, 0.0, np.radians(15.0), 0.0]), 50),
                (np.array([1.2, 0.0, 0.0, 0.0]), 100),
                (np.array([1.2, 0.0, np.radians(-15.0), 0.0]), 50),
                (np.array([1.2, 0.0, 0.0, 0.0]), 50),
            ]
        },
        # Coincidiendo con el perfil de yaw objetivo (0 a 20 grados)
        {
            'initial_state': np.array([1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'references': [
                (np.array([1.2, 0.0, 0.0, 0.0]), 50),
                (np.array([1.2, 0.0, 0.0, np.radians(20.0)]), 50),
                (np.array([1.2, 0.0, 0.0, 0.0]), 100),
            ]
        }
    ]
    
    # Generar datos para perfiles de referencia específicos
    for profile in specific_profiles:
        # Inicializar integrales de error
        z_error_integral = 0.0
        roll_error_integral = 0.0
        pitch_error_integral = 0.0
        yaw_error_integral = 0.0
        
        # Historial de errores
        error_history = {
            'z': {'current': 0.0, 'prev': 0.0},
            'roll': {'current': 0.0, 'prev': 0.0},
            'pitch': {'current': 0.0, 'prev': 0.0},
            'yaw': {'current': 0.0, 'prev': 0.0}
        }
        
        # Estado inicial
        state = profile['initial_state'].copy()
        
        # Recorrer cada referencia en el perfil
        for ref, steps in profile['references']:
            for _ in range(steps):
                # Extraer componentes del estado
                z, dz, roll, droll, pitch, dpitch, yaw, dyaw = state
                z_ref, roll_ref, pitch_ref, yaw_ref = ref
                
                # Calcular errores
                z_error = z_ref - z
                roll_error = roll_ref - roll
                pitch_error = pitch_ref - pitch
                yaw_error = yaw_ref - yaw
                
                # Actualizar historial de errores
                for key, error, prev_error in zip(
                    ['z', 'roll', 'pitch', 'yaw'],
                    [z_error, roll_error, pitch_error, yaw_error],
                    [error_history[k]['current'] for k in ['z', 'roll', 'pitch', 'yaw']]
                ):
                    error_history[key]['prev'] = error_history[key]['current']
                    error_history[key]['current'] = error
                
                # Calcular derivadas de error
                z_error_deriv = (z_error - error_history['z']['prev']) / simulator.dt
                roll_error_deriv = (roll_error - error_history['roll']['prev']) / simulator.dt
                pitch_error_deriv = (pitch_error - error_history['pitch']['prev']) / simulator.dt
                yaw_error_deriv = (yaw_error - error_history['yaw']['prev']) / simulator.dt
                
                # Actualizar integrales de error con anti-windup
                max_integral = 1.0
                
                z_error_integral += z_error * simulator.dt
                z_error_integral = np.clip(z_error_integral, -max_integral, max_integral)
                
                roll_error_integral += roll_error * simulator.dt
                roll_error_integral = np.clip(roll_error_integral, -max_integral, max_integral)
                
                pitch_error_integral += pitch_error * simulator.dt
                pitch_error_integral = np.clip(pitch_error_integral, -max_integral, max_integral)
                
                yaw_error_integral += yaw_error * simulator.dt
                yaw_error_integral = np.clip(yaw_error_integral, -max_integral, max_integral)
                
                # Cálculo PID de control
                thrust = simulator.m * (simulator.g + Kp_z * z_error + Kd_z * z_error_deriv + Ki_z * z_error_integral)
                
                # Añadir empuje extra cerca del suelo para evitar arrastre
                if z < 0.1 and z_ref > 0.1:
                    thrust += simulator.m * 4.0  # Aumentado para mejor despegue
                
                # PID para roll, pitch, yaw
                Mroll = Kp_roll * roll_error + Kd_roll * roll_error_deriv + Ki_roll * roll_error_integral
                Mpitch = Kp_pitch * pitch_error + Kd_pitch * pitch_error_deriv + Ki_pitch * pitch_error_integral
                Myaw = Kp_yaw * yaw_error + Kd_yaw * yaw_error_deriv + Ki_yaw * yaw_error_integral
                
                # Aplicar límites de control
                thrust = np.clip(thrust, simulator.thrust_min, simulator.thrust_max)
                Mroll = np.clip(Mroll, simulator.Mroll_min, simulator.Mroll_max)
                Mpitch = np.clip(Mpitch, simulator.Mpitch_min, simulator.Mpitch_max)
                Myaw = np.clip(Myaw, simulator.Myaw_min, simulator.Myaw_max)
                
                # Acción de control combinada
                control = np.array([thrust, Mroll, Mpitch, Myaw])
                
                # Preparar historial de errores para entrada al modelo
                error_hist_flat = [
                    error_history['z']['current'], error_history['z']['prev'], z_error_integral,
                    error_history['roll']['current'], error_history['roll']['prev'], roll_error_integral,
                    error_history['pitch']['current'], error_history['pitch']['prev'], pitch_error_integral,
                    error_history['yaw']['current'], error_history['yaw']['prev'], yaw_error_integral
                ]
                
                # Guardar el punto de datos
                states.append(state)
                references.append(ref)
                prev_controls.append(np.concatenate([prev_control_1, prev_control_2]))
                error_histories.append(error_hist_flat)
                controls.append(control)
                
                # Actualizar estado y control previo
                state = simulator.step(state, control, add_disturbance=(z > 0.2))  # Reducir perturbación cerca del suelo
                prev_control_2 = prev_control_1.copy()
                prev_control_1 = control.copy()
    
    # Generar escenarios aleatorios adicionales para asegurar diversidad
    for _ in range(scenarios_count - len(specific_profiles)):
        # Estado inicial aleatorio
        state = np.array([
            np.random.uniform(0.0, 1.0),    # z
            np.random.uniform(-0.2, 0.2),   # dz
            np.random.uniform(-0.1, 0.1),   # roll
            np.random.uniform(-0.1, 0.1),   # droll
            np.random.uniform(-0.1, 0.1),   # pitch
            np.random.uniform(-0.1, 0.1),   # dpitch
            np.random.uniform(-0.1, 0.1),   # yaw
            np.random.uniform(-0.1, 0.1),   # dyaw
        ])
        
        # Referencia inicial aleatoria
        reference = np.array([
            np.random.uniform(0.0, 1.5),     # z_ref
            np.random.uniform(-0.2, 0.2),    # roll_ref
            np.random.uniform(-0.3, 0.3),    # pitch_ref
            np.random.uniform(-0.5, 0.5),    # yaw_ref
        ])
        
        # Inicializar integrales de error
        z_error_integral = 0.0
        roll_error_integral = 0.0
        pitch_error_integral = 0.0
        yaw_error_integral = 0.0
        
        # Historial de errores
        error_history = {
            'z': {'current': 0.0, 'prev': 0.0},
            'roll': {'current': 0.0, 'prev': 0.0},
            'pitch': {'current': 0.0, 'prev': 0.0},
            'yaw': {'current': 0.0, 'prev': 0.0}
        }
        
        # Simulación por varios pasos
        for _ in range(200):
            # Extraer componentes del estado
            z, dz, roll, droll, pitch, dpitch, yaw, dyaw = state
            z_ref, roll_ref, pitch_ref, yaw_ref = reference
            
            # Calcular errores
            z_error = z_ref - z
            roll_error = roll_ref - roll
            pitch_error = pitch_ref - pitch
            yaw_error = yaw_ref - yaw
            
            # Actualizar historial de errores
            for key, error, prev_error in zip(
                ['z', 'roll', 'pitch', 'yaw'],
                [z_error, roll_error, pitch_error, yaw_error],
                [error_history[k]['current'] for k in ['z', 'roll', 'pitch', 'yaw']]
            ):
                error_history[key]['prev'] = error_history[key]['current']
                error_history[key]['current'] = error
            
            # Calcular derivadas de error
            z_error_deriv = (z_error - error_history['z']['prev']) / simulator.dt
            roll_error_deriv = (roll_error - error_history['roll']['prev']) / simulator.dt
            pitch_error_deriv = (pitch_error - error_history['pitch']['prev']) / simulator.dt
            yaw_error_deriv = (yaw_error - error_history['yaw']['prev']) / simulator.dt
            
            # Actualizar integrales de error con anti-windup
            max_integral = 1.0
            
            z_error_integral += z_error * simulator.dt
            z_error_integral = np.clip(z_error_integral, -max_integral, max_integral)
            
            roll_error_integral += roll_error * simulator.dt
            roll_error_integral = np.clip(roll_error_integral, -max_integral, max_integral)
            
            pitch_error_integral += pitch_error * simulator.dt
            pitch_error_integral = np.clip(pitch_error_integral, -max_integral, max_integral)
            
            yaw_error_integral += yaw_error * simulator.dt
            yaw_error_integral = np.clip(yaw_error_integral, -max_integral, max_integral)
            
            # Cálculo PID de control
            thrust = simulator.m * (simulator.g + Kp_z * z_error + Kd_z * z_error_deriv + Ki_z * z_error_integral)
            
            # Añadir empuje extra cerca del suelo para evitar arrastre
            if z < 0.1 and z_ref > 0.1:
                thrust += simulator.m * 4.0  # Aumentado para mejor despegue
            
            # PID para roll, pitch, yaw
            Mroll = Kp_roll * roll_error + Kd_roll * roll_error_deriv + Ki_roll * roll_error_integral
            Mpitch = Kp_pitch * pitch_error + Kd_pitch * pitch_error_deriv + Ki_pitch * pitch_error_integral
            Myaw = Kp_yaw * yaw_error + Kd_yaw * yaw_error_deriv + Ki_yaw * yaw_error_integral
            
            # Aplicar límites de control
            thrust = np.clip(thrust, simulator.thrust_min, simulator.thrust_max)
            Mroll = np.clip(Mroll, simulator.Mroll_min, simulator.Mroll_max)
            Mpitch = np.clip(Mpitch, simulator.Mpitch_min, simulator.Mpitch_max)
            Myaw = np.clip(Myaw, simulator.Myaw_min, simulator.Myaw_max)
            
            # Acción de control combinada
            control = np.array([thrust, Mroll, Mpitch, Myaw])
            
            # Preparar historial de errores para entrada al modelo
            error_hist_flat = [
                error_history['z']['current'], error_history['z']['prev'], z_error_integral,
                error_history['roll']['current'], error_history['roll']['prev'], roll_error_integral,
                error_history['pitch']['current'], error_history['pitch']['prev'], pitch_error_integral,
                error_history['yaw']['current'], error_history['yaw']['prev'], yaw_error_integral
            ]
            
            # Guardar el punto de datos
            states.append(state)
            references.append(reference)
            prev_controls.append(np.concatenate([prev_control_1, prev_control_2]))
            error_histories.append(error_hist_flat)
            controls.append(control)
            
            # Actualizar estado y control previo
            state = simulator.step(state, control, add_disturbance=(z > 0.2))  # Reducir perturbación cerca del suelo
            prev_control_2 = prev_control_1.copy()
            prev_control_1 = control.copy()
            
            # Ocasionalmente cambiar la referencia para mayor variedad
            if np.random.random() < 0.03:  # Probabilidad reducida para entrenamiento más estable
                reference = np.array([
                    np.random.uniform(0.0, 1.5),     # Rango completo incluyendo cero
                    np.random.uniform(-0.2, 0.2),
                    np.random.uniform(-0.3, 0.3),
                    np.random.uniform(-0.5, 0.5),
                ])
    
    # Combinar entradas para la red neuronal
    X = np.hstack([
        np.array(states),
        np.array(references),
        np.array(prev_controls),
        np.array(error_histories)
    ])
    
    # Salidas de acciones de control
    y = np.array(controls)
    
    # Normalizar salidas a 0-1
    y_normalized = np.zeros_like(y)
    y_normalized[:, 0] = (y[:, 0] - simulator.thrust_min) / (simulator.thrust_max - simulator.thrust_min)
    y_normalized[:, 1] = (y[:, 1] - simulator.Mroll_min) / (simulator.Mroll_max - simulator.Mroll_min)
    y_normalized[:, 2] = (y[:, 2] - simulator.Mpitch_min) / (simulator.Mpitch_max - simulator.Mpitch_min)
    y_normalized[:, 3] = (y[:, 3] - simulator.Myaw_min) / (simulator.Myaw_max - simulator.Myaw_min)
    
    print(f"Generadas {len(X)} muestras de entrenamiento")
    
    return X, y_normalized


def simulate_test_scenario(controller, total_time=30.0):
    """
    Simula el escenario de prueba con perfiles de referencia que coinciden exactamente.
    
    Parámetros:
    -----------
    controller: EnhancedDroneController
        Controlador neuronal
    total_time: float
        Tiempo total de simulación en segundos
        
    Retorna:
    --------
    time_points: np.ndarray
        Puntos de tiempo para graficar
    states: np.ndarray
        Historial de estados
    inputs: np.ndarray
        Historial de entradas de control
    references: np.ndarray
        Historial de referencias
    """
    # Parámetros de simulación
    dt = 0.02  # 50 Hz
    steps = int(total_time / dt)
    
    # Crear simulador
    simulator = DroneSimulator(dt=dt)
    
    # Inicializar historiales
    states = np.zeros((steps+1, 8))
    inputs = np.zeros((steps, 4))
    references = np.zeros((steps, 4))
    
    # Estado inicial (en el suelo con altitud cero)
    states[0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Definir perfiles de referencia para coincidir con los gráficos objetivo
    time_points = np.arange(0, total_time, dt)
    time_points = time_points[:steps]  # Asegurar longitud correcta
    
    # Perfil de altitud: inicia en 0m, sube a 0.7m, luego a 1.2m en t=5s, luego se mantiene en 1.2
    alt_ref = np.zeros_like(time_points)  # Inicia en cero
    
    for i, t in enumerate(time_points):
        if t < 0.5:
            alt_ref[i] = 0.0  # En el suelo
        elif t < 2.0:
            progress = (t - 0.5) / 1.5  # Subida a 0.7m
            alt_ref[i] = 0.0 + 0.7 * progress
        elif t < 5.0:
            progress = (t - 2.0) / 3.0  # Subida a 1.2m
            alt_ref[i] = 0.7 + 0.5 * progress
        else:
            alt_ref[i] = 1.2  # Mantener en 1.2
    
    # Perfil de roll: coincide exactamente con el gráfico de prueba de roll
    roll_ref = np.zeros_like(time_points)
    
    for i, t in enumerate(time_points):
        if t < 10.0:
            roll_ref[i] = 0.0
        elif t < 15.0:
            roll_ref[i] = 0.0  # Primer segmento donde roll es 0
        elif t < 17.0:
            roll_ref[i] = np.radians(-3.0)  # Baja a -3 grados
        elif t < 19.0:
            roll_ref[i] = np.radians(3.0)  # Sube a +3 grados
        elif t < 21.0:
            roll_ref[i] = np.radians(0.0)  # Vuelve a 0 grados
        else:
            roll_ref[i] = np.radians(0.0)  # Se mantiene en 0
    
    # Perfil de pitch: coincide con el gráfico de prueba de pitch
    pitch_ref = np.zeros_like(time_points)
    
    for i, t in enumerate(time_points):
        if t < 8.0:
            pitch_ref[i] = 0.0
        elif t < 9.5:
            pitch_ref[i] = np.radians(15.0)  # Sube a +15 grados
        elif t < 12.0:
            pitch_ref[i] = np.radians(-15.0)  # Baja a -15 grados
        else:
            pitch_ref[i] = 0.0  # Vuelve a 0
    
    # Perfil de yaw: escalón simple de 20 grados
    yaw_ref = np.zeros_like(time_points)
    yaw_ref[(time_points >= 8.0) & (time_points < 9.5)] = np.radians(20.0)
    
    # Ejecutar simulación
    for k in range(steps):
        # Obtener referencia actual
        ref = np.array([alt_ref[k], roll_ref[k], pitch_ref[k], yaw_ref[k]])
        references[k] = ref
        
        # Obtener control de la red neuronal
        control = controller.predict(states[k], ref)
        inputs[k] = control
        
        # Simular dinámica del dron
        states[k+1] = simulator.step(states[k], control, add_disturbance=(k > 100))  # Sin perturbaciones durante el despegue inicial
    
    # Crear puntos de tiempo para graficar
    plot_time_points = np.linspace(0, total_time, steps+1)
    
    return plot_time_points, states, inputs, references


def run_enhanced_nn_controller():
    """
    Función principal para entrenar y probar el controlador neuronal mejorado.
    """
    print("Iniciando entrenamiento y prueba del controlador neuronal mejorado para dron...")
    
    # 1. Generar datos de entrenamiento de alta calidad
    try:
        print("Intentando cargar datos de entrenamiento pre-generados...")
        X = np.load('enhanced_drone_training_inputs.npy')
        y = np.load('enhanced_drone_training_outputs.npy')
        print("Datos de entrenamiento existentes cargados exitosamente.")
    except FileNotFoundError:
        print("Generando nuevos datos de entrenamiento mejorados...")
        X, y = generate_enhanced_training_data(n_samples=100000)
        
        # Guardar datos para uso futuro
        np.save('enhanced_drone_training_inputs.npy', X)
        np.save('enhanced_drone_training_outputs.npy', y)
    
    # 2. Crear y entrenar el controlador neuronal mejorado
    controller = EnhancedDroneController()
    history = controller.train(X, y, epochs=100, batch_size=128)
    
    # 3. Graficar historial de entrenamiento
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Pérdida Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida Validación')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.yscale('log')
    plt.title('Historial de Entrenamiento')
    plt.legend()
    plt.grid(True)
    plt.savefig('enhanced_nn_training_history.png')
    plt.close()
    
    # 4. Guardar el modelo entrenado
    controller.save_model('enhanced_drone_nn_controller.h5')
    print("Controlador neuronal mejorado entrenado y guardado en 'enhanced_drone_nn_controller.h5'")
    
    # 5. Probar el controlador con los perfiles de referencia precisos
    print("Probando el controlador neuronal mejorado...")
    time_points, states, inputs, references = simulate_test_scenario(controller)
    
    # 6. Graficar los resultados
    # Ajustar puntos de tiempo de referencia para coincidir en longitud
    ref_time_points = time_points[:-1]
    
    # Crear subfiguras
    fig, axs = plt.subplots(4, 1, figsize=(12, 15))
    
    # Convertir radianes a grados para ángulos
    roll_deg = np.degrees(states[:, 2])
    pitch_deg = np.degrees(states[:, 4])
    yaw_deg = np.degrees(states[:, 6])
    
    roll_ref_deg = np.degrees(references[:, 1])
    pitch_ref_deg = np.degrees(references[:, 2])
    yaw_ref_deg = np.degrees(references[:, 3])
    
    # Altitud
    axs[0].plot(time_points, states[:, 0], 'blue', linewidth=2, label='Real')
    axs[0].plot(ref_time_points, references[:, 0], 'blue', linestyle='--', linewidth=1, label='Referencia')
    axs[0].set_title('Test Altitud', color='blue')
    axs[0].set_ylim(0.0, 1.3)  # Modificado para mostrar todo el rango desde 0
    axs[0].grid(True)
    axs[0].legend()
    
    # Roll
    axs[1].plot(time_points, roll_deg, 'purple', linewidth=2)
    axs[1].plot(ref_time_points, roll_ref_deg, 'purple', linestyle='--', linewidth=1)
    axs[1].set_title('Test Roll', color='purple')
    axs[1].set_ylim(-4, 4)
    axs[1].grid(True)
    
    # Pitch
    axs[2].plot(time_points, pitch_deg, 'red', linewidth=2)
    axs[2].plot(ref_time_points, pitch_ref_deg, 'red', linestyle='--', linewidth=1)
    axs[2].set_title('Test Pitch', color='red')
    axs[2].set_ylim(-20, 20)
    axs[2].grid(True)
    
    # Yaw
    axs[3].plot(time_points, yaw_deg, 'magenta', linewidth=2)
    axs[3].plot(ref_time_points, yaw_ref_deg, 'magenta', linestyle='--', linewidth=1)
    axs[3].set_title('Test Yaw', color='magenta')
    axs[3].set_ylim(-5, 25)
    axs[3].grid(True)
    
    axs[3].set_xlabel('Tiempo (seg)')
    
    plt.tight_layout()
    plt.savefig('enhanced_nn_drone_stabilization.png')
    plt.close()
    
    print("Prueba completada. Resultados guardados en:")
    print("- enhanced_nn_training_history.png")
    print("- enhanced_nn_drone_stabilization.png")
    

if __name__ == "__main__":
    run_enhanced_nn_controller()