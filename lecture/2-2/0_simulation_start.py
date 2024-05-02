# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day2. 
# 2-2.0 Basic simulation
# ---- ---- ---- ----

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})  # 해당 코드 실행 시, 시뮬레이션이 시작됨
print(simulation_app.DEFAULT_LAUNCHER_CONFIG)

time = 0
while simulation_app.is_running():
    print(f"Simulation Application is Running... {time}", end="\r")
    time += 1
    
    
simulation_app.close()
print("Simulation is Closed")

