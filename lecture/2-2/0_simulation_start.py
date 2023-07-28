# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day2. 
# 2-2.0 Basic simulation
# ---- ---- ---- ----

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})  # 해당 코드 실행 시, 시뮬레이션이 시작됨
print(simulation_app.DEFAULT_LAUNCHER_CONFIG)

i = 0

while simulation_app.is_running():
    i += 1
    print(i)
    
simulation_app.close()
