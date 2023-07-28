# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day2. 
# 2-2.1 Basic simulation loop
# ---- ---- ---- ----
import time

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})
print(simulation_app.DEFAULT_LAUNCHER_CONFIG)

from omni.isaac.core import World                       # isaac sim 라이브러리 내 함수 import


my_world = World(stage_units_in_meters=1.0)             # World 선언 및 생성


sim_step = 0                                            
max_sim_step = 1000                                     # max_step 설정
start = time.time()
while simulation_app.is_running():                      
    my_world.step(render=True)                          # 한 step 실행
    print("Simulation Step: ", sim_step)                # 매 step 마다 수행할 동작
    
    sim_step += 1                                       
    if sim_step >= max_sim_step:                        
        print("Total Time"+str(time.time()-start))
        simulation_app.close()                          # 10000 step 실행 후, 시뮬레이션 종료
simulation_app.close()