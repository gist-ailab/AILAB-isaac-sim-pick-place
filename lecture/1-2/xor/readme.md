## 비선형 함수의 필요성 시각화 코드
**AIlab Project for GSH - GIST Creative Research Program**  

### How to Run
```
python show_xor.py
# 마우스 좌우 클릭으로 레이블 생성 (Binary Classification) -> 창 닫기
# 각 layer마다 데이터가 시각화됨
```

### Code Overview
1. **generate_array.py** : Generating Raw dataset(array) for training with pygame.<br />
2. **vis_dataset.py** : preprocessing raw array to dataset<br />
3. **model.py** : FCN model structure<br />
4. **visualize.py** : Visualizing code with PyQt5<br />
5. **get_train.py** : Training code<br />
6. **show_xor.py** : Visualize all latent space<br />