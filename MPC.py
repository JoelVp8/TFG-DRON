import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

class DroneMPC:
    """
    Controlador Predictivo por Modelo (MPC) para un cuadricóptero que regula
    altitud (z) y actitud (roll, pitch, yaw).

    ------------------------------------------------------------
    Estados  (8) :  z,  dz,  roll, droll, pitch, dpitch, yaw, dyaw
    Entradas (4) :  thrust,  Mroll,  Mpitch,  Myaw
    Referencias (4) :  z_ref, roll_ref, pitch_ref, yaw_ref
    """

    def __init__(self, dt: float = 0.02, horizon: int = 20):
        self.dt = dt          # [s]  tiempo de muestreo de control (50 Hz)
        self.N = horizon      # horizonte de predicción (pasos)

        # === Parámetros físicos del vehículo ===
        self.m   = 0.60   # [kg]   masa
        self.Ixx = 0.02   # [kg·m²] inercia alrededor de x (roll)
        self.Iyy = 0.02   # [kg·m²] inercia alrededor de y (pitch)
        self.Izz = 0.04   # [kg·m²] inercia alrededor de z (yaw)
        self.g   = 9.81   # [m/s²] aceleración de la gravedad

        # === Límites de los actuadores (ajustar para tu vehículo) ===
        self.thrust_min = 0.0     # [N]
        self.thrust_max = 15.0    # [N]
        self.Mroll_min  = -0.5    # [N·m]
        self.Mroll_max  =  0.5
        self.Mpitch_min = -0.5
        self.Mpitch_max =  0.5
        self.Myaw_min   = -0.2
        self.Myaw_max   =  0.2
        
        # Empuje de hover - punto de equilibrio
        self.u_hover = self.m * self.g

        # Pre-construir el problema de optimización para que cada llamada sea rápida
        self._build_problem()

    def _build_problem(self):
        """Configura variables, parámetros y restricciones de CVXPY."""
        n_x = 8   # dimensión del estado
        n_u = 4   # dimensión de la entrada
        N   = self.N
        dt  = self.dt

        # Dinámica del sistema (discretizada)
        A = np.eye(n_x)
        A[0, 1] = dt
        A[2, 3] = dt
        A[4, 5] = dt
        A[6, 7] = dt

        B = np.zeros((n_x, n_u))
        B[1, 0] = dt / self.m
        B[3, 1] = dt / self.Ixx
        B[5, 2] = dt / self.Iyy
        B[7, 3] = dt / self.Izz

        # Vector de efecto de la gravedad (como perturbación externa)
        g_vec = np.zeros(n_x)
        g_vec[1] = -self.g * dt  # Efecto de la gravedad en la velocidad vertical

        self.x = cp.Variable((n_x, N + 1))
        self.u = cp.Variable((n_u, N))
        
        # Parámetros para el estado inicial y referencia
        self.x0_param  = cp.Parameter(n_x)
        self.ref_param = cp.Parameter(4)

        # Pesos de ajuste para la función de costo
        Qz = 15.0     # Peso para seguimiento de altitud (aumentado)
        Qr = 8.0      # Peso para seguimiento de roll
        Qp = 8.0      # Peso para seguimiento de pitch
        Qy = 5.0      # Peso para seguimiento de yaw
        
        R_thrust = 0.005  # Penalización por uso de empuje (reducido)
        R_moment = 0.05   # Penalización por uso de momento roll/pitch
        R_yaw    = 0.02   # Penalización por uso de momento yaw
        
        # Penalización por cambio de entrada
        R_delta = 0.1     # Penalización por cambios en la entrada de control

        cost = 0
        constr = [self.x[:, 0] == self.x0_param]

        # Entrada de control previa para la restricción de tasa en el primer paso
        u_prev = cp.Parameter(n_u)
        self.u_prev_param = u_prev
        
        for k in range(N):
            # Dinámica del sistema con gravedad
            constr += [self.x[:, k + 1] == A @ self.x[:, k] + B @ self.u[:, k] + g_vec]
            
            # Restricciones de entrada
            constr += [self.u[0, k] >= self.thrust_min,
                      self.u[0, k] <= self.thrust_max,
                      self.u[1, k] >= self.Mroll_min,
                      self.u[1, k] <= self.Mroll_max,
                      self.u[2, k] >= self.Mpitch_min,
                      self.u[2, k] <= self.Mpitch_max,
                      self.u[3, k] >= self.Myaw_min,
                      self.u[3, k] <= self.Myaw_max]
            
            # Restricciones de seguridad
            constr += [self.x[0, k] >= 0.05]  # Altitud mínima (margen de seguridad de 5cm)
            
            # Restricciones de velocidad para evitar movimientos bruscos
            constr += [self.x[1, k] >= -1.0]  # Velocidad máxima de descenso
            constr += [self.x[1, k] <= 1.5]   # Velocidad máxima de ascenso
            
            # Restricción de cambio de entrada en el primer paso
            if k == 0:
                constr += [cp.abs(self.u[:, k] - self.u_prev_param) <= np.array([2.0, 0.2, 0.2, 0.1])]
            
            # Restricciones de cambio de entrada entre pasos consecutivos
            if k > 0:
                constr += [cp.abs(self.u[:, k] - self.u[:, k-1]) <= np.array([2.0, 0.2, 0.2, 0.1])]

            # Error de seguimiento
            z_err    = self.x[0, k] - self.ref_param[0]
            roll_err = self.x[2, k] - self.ref_param[1]
            pitch_err = self.x[4, k] - self.ref_param[2]
            yaw_err  = self.x[6, k] - self.ref_param[3]
            
            # Costo de etapa
            cost += Qz * cp.square(z_err) \
                  + Qr * cp.square(roll_err) \
                  + Qp * cp.square(pitch_err) \
                  + Qy * cp.square(yaw_err) \
                  + R_thrust * cp.square(self.u[0, k] - self.u_hover) \
                  + R_moment * cp.square(self.u[1, k]) \
                  + R_moment * cp.square(self.u[2, k]) \
                  + R_yaw    * cp.square(self.u[3, k])
            
            # Agregar penalización por cambio de entrada para todos menos el primer paso
            if k > 0:
                for i in range(n_u):
                    cost += R_delta * cp.square(self.u[i, k] - self.u[i, k-1])

        # Costo terminal
        z_err_T    = self.x[0, N] - self.ref_param[0]
        roll_err_T = self.x[2, N] - self.ref_param[1]
        pitch_err_T = self.x[4, N] - self.ref_param[2]
        yaw_err_T  = self.x[6, N] - self.ref_param[3]

        # Pesos terminales mayores para mejor seguimiento de referencia
        cost += 2 * Qz * cp.square(z_err_T) \
              + 2 * Qr * cp.square(roll_err_T) \
              + 2 * Qp * cp.square(pitch_err_T) \
              + 2 * Qy * cp.square(yaw_err_T)

        self._prob = cp.Problem(cp.Minimize(cost), constr)
        
        # Inicializar entrada de control previa
        self.prev_u = np.zeros(n_u)
        self.prev_u[0] = self.u_hover  # Inicializar con empuje de hover

    def _solve_mpc(self):
        try:
            self._prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            solved = True
        except cp.SolverError:
            print("Error del solver, intentando de nuevo sin warm start")
            try:
                self._prob.solve(solver=cp.OSQP, verbose=False)
                solved = True
            except cp.SolverError:
                print("El solver falló de nuevo, usando control de respaldo")
                solved = False
        
        return solved

    def predict(self, entrada: np.ndarray) -> np.ndarray:
        """
        Calcula la primera acción de control óptima.

        Parámetros
        ----------
        entrada : array-like, shape (12,)
            Concatenación de la estimación del estado (8,) y referencias (4,).

        Retorna
        -------
        u0 : ndarray, shape (4,)
            Entrada de control a aplicar en el paso actual.
        """
        entrada = np.asarray(entrada).flatten()
        x_hat = entrada[:8]
        ref = entrada[8:]

        self.x0_param.value = x_hat
        self.ref_param.value = ref
        self.u_prev_param.value = self.prev_u

        solved = self._solve_mpc()
        
        if solved and self.u[:, 0].value is not None:
            control = self.u[:, 0].value
        else:
            # Control de respaldo: mantener empuje de hover y momentos cero
            control = np.zeros(4)
            control[0] = self.u_hover
            print("Usando control de respaldo")
        
        # Guardar para la siguiente iteración
        self.prev_u = control
        
        return control


def simulate_test_scenario(total_time=30.0):
    """
    Simula el escenario de prueba con un despegue y maniobras seguras.
    """
    # Crear el controlador MPC
    dt = 0.02  # 50 Hz
    mpc = DroneMPC(dt=dt, horizon=20)  # Reducido a 20 pasos
    
    # Calcular el número de pasos
    steps = int(total_time / dt)
    
    # Inicializar historia de estados, entradas y referencias
    states = np.zeros((steps+1, 8))
    inputs = np.zeros((steps, 4))
    references = np.zeros((steps, 4))
    
    # Estado inicial (en el suelo, con pequeña altura segura)
    states[0] = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Inicializar el empuje con valor de hover para evitar caída
    mpc.prev_u[0] = mpc.u_hover
    
    # Definir perfiles de referencia 
    time_points = np.arange(0, total_time, dt)
    time_points = time_points[:steps]  # Asegurar longitud correcta
    
    # Altitud: Transición suave desde 0.1m a 1.2m
    alt_ref = np.ones_like(time_points) * 0.1  # Altura inicial segura
    
    # Rampa suave de subida en lugar de escalón
    t_start = 1.0   # Tiempo de inicio de subida
    t_end = 3.0     # Tiempo en que se alcanza altura final
    
    for i, t in enumerate(time_points):
        if t < t_start:
            alt_ref[i] = 0.1
        elif t < t_end:
            # Transición suave (rampa)
            progress = (t - t_start) / (t_end - t_start)
            alt_ref[i] = 0.1 + progress * (1.2 - 0.1)
        else:
            alt_ref[i] = 1.2
    
    # Roll: Normalmente 0, pero con cambios de +3 a -3 grados entre 15-18s
    roll_ref = np.zeros_like(time_points)
    roll_ref[(time_points >= 15.0) & (time_points < 16.5)] = np.radians(3)
    roll_ref[(time_points >= 16.5) & (time_points < 18.0)] = np.radians(-3)
    
    # Pitch: Cambios entre 0, +15, -15 entre 8-12s
    pitch_ref = np.zeros_like(time_points)
    pitch_ref[(time_points >= 8.0) & (time_points < 9.5)] = np.radians(15)
    pitch_ref[(time_points >= 9.5) & (time_points < 12.0)] = np.radians(-15)
    
    # Yaw: Cambio a +20 grados en t=8s
    yaw_ref = np.zeros_like(time_points)
    yaw_ref[(time_points >= 8.0) & (time_points < 9.5)] = np.radians(20)
    
    # Asignar referencias para cada paso
    for k in range(steps):
        references[k] = [alt_ref[k], roll_ref[k], pitch_ref[k], yaw_ref[k]]
    
    # Simulación del modelo dinámico
    for k in range(steps):
        # Obtener referencias actuales
        ref = references[k]
        
        # Concatenar estado y referencia para el MPC
        mpc_input = np.concatenate([states[k], ref])
        
        # Calcular acción de control óptima
        control = mpc.predict(mpc_input)
        inputs[k] = control
        
        # Simular la dinámica del dron
        next_state = np.copy(states[k])
        
        # Perturbación simulada (viento descendente constante)
        wind_disturbance = -0.15  # [N] de empuje negativo

        # Ruido leve en la estimación de velocidad vertical
        dz_noise = np.random.normal(0, 0.02)  # media 0, desviación 0.02 m/s

        # Actualizar velocidad vertical con perturbación y ruido
        next_state[1] += dt * ((control[0] + wind_disturbance) / mpc.m - mpc.g) + dz_noise

        next_state[3] += dt * control[1] / mpc.Ixx  # droll
        next_state[5] += dt * control[2] / mpc.Iyy  # dpitch
        next_state[7] += dt * control[3] / mpc.Izz  # dyaw
        
        # Actualizar posiciones
        next_state[0] += dt * next_state[1]  # z
        next_state[2] += dt * next_state[3]  # roll
        next_state[4] += dt * next_state[5]  # pitch
        next_state[6] += dt * next_state[7]  # yaw
        
        # Restricción física: el dron no puede estar por debajo del suelo
        next_state[0] = max(0.0, next_state[0])
        
        # Si el dron toca el suelo, detener la velocidad vertical
        if next_state[0] <= 0.001:
            next_state[1] = 0.0
        
        # Guardar el estado para el siguiente paso
        states[k+1] = next_state
    
    # Crear el vector de tiempo para graficar
    plot_time_points = np.linspace(0, total_time, steps+1)
    
    return plot_time_points, states, inputs, references


def plot_results(time_points, states, inputs, references):
    """
    Genera gráficas detalladas del comportamiento del dron.
    """
    # Asegurarse de que references tenga la misma longitud que time_points para graficar
    ref_time_points = time_points[:-1]  # Quitar el último punto para que coincida con references
    
    # Crear subfiguras
    fig, axs = plt.subplots(5, 1, figsize=(12, 15))
    
    # Convertir radianes a grados para ángulos
    roll_deg = np.degrees(states[:, 2])
    pitch_deg = np.degrees(states[:, 4])
    yaw_deg = np.degrees(states[:, 6])
    
    roll_ref_deg = np.degrees(references[:, 1])
    pitch_ref_deg = np.degrees(references[:, 2])
    yaw_ref_deg = np.degrees(references[:, 3])
    
    # Altitud (primera para ver claramente)
    axs[0].plot(time_points, states[:, 0], 'blue', linewidth=2, label='Actual')
    axs[0].plot(ref_time_points, references[:, 0], 'blue', linestyle='--', linewidth=1, label='Reference')
    axs[0].set_title('Test Alt', color='blue')
    axs[0].set_ylim(-0.1, 1.3)
    axs[0].grid(True)
    axs[0].legend()
    
    # Velocidad vertical
    axs[1].plot(time_points, states[:, 1], 'green', linewidth=2)
    axs[1].set_title('Vertical Velocity (dz)', color='green')
    axs[1].set_ylim(-1.5, 1.5)
    axs[1].grid(True)
    
    # Roll
    axs[2].plot(time_points, roll_deg, 'purple', linewidth=2)
    axs[2].plot(ref_time_points, roll_ref_deg, 'purple', linestyle='--', linewidth=1)
    axs[2].set_title('Test Roll', color='purple')
    axs[2].set_ylim(-5, 5)
    axs[2].grid(True)
    
    # Pitch
    axs[3].plot(time_points, pitch_deg, 'red', linewidth=2)
    axs[3].plot(ref_time_points, pitch_ref_deg, 'red', linestyle='--', linewidth=1)
    axs[3].set_title('Test Pitch', color='red')
    axs[3].set_ylim(-20, 20)
    axs[3].grid(True)
    
    # Yaw
    axs[4].plot(time_points, yaw_deg, 'magenta', linewidth=2)
    axs[4].plot(ref_time_points, yaw_ref_deg, 'magenta', linestyle='--', linewidth=1)
    axs[4].set_title('Test Yaw', color='magenta')
    axs[4].set_ylim(-5, 25)
    axs[4].grid(True)
    
    axs[4].set_xlabel('Time (sec)')
    
    plt.tight_layout()
    plt.savefig('drone_stabilization_test.png')
    
    # Graficar los controles también
    fig2, axs2 = plt.subplots(4, 1, figsize=(12, 10))
    
    # Thrust
    axs2[0].plot(ref_time_points, inputs[:, 0], 'blue', linewidth=2)
    axs2[0].axhline(y=0.6*9.81, color='r', linestyle='--', label='Hover thrust')
    axs2[0].set_title('Thrust')
    axs2[0].grid(True)
    axs2[0].legend()
    
    # Roll moment
    axs2[1].plot(ref_time_points, inputs[:, 1], 'purple', linewidth=2)
    axs2[1].set_title('Roll Moment')
    axs2[1].grid(True)
    
    # Pitch moment
    axs2[2].plot(ref_time_points, inputs[:, 2], 'red', linewidth=2)
    axs2[2].set_title('Pitch Moment')
    axs2[2].grid(True)
    
    # Yaw moment
    axs2[3].plot(ref_time_points, inputs[:, 3], 'magenta', linewidth=2)
    axs2[3].set_title('Yaw Moment')
    axs2[3].grid(True)
    
    axs2[3].set_xlabel('Time (sec)')
    
    plt.tight_layout()
    plt.savefig('drone_control_inputs.png')
    
    plt.show()


def run_test():
    """
    Ejecuta la simulación y muestra los resultados.
    """
    print("Ejecutando simulación de estabilización del dron...")
    time_points, states, inputs, references = simulate_test_scenario(total_time=30.0)
    
    print("Simulación completada. Mostrando resultados...")
    plot_results(time_points, states, inputs, references)
    
    print("Test finalizado. Los resultados se han guardado en 'drone_stabilization_test.png' y 'drone_control_inputs.png'")
    

if __name__ == "__main__":
    run_test()