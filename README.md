*   python execOnePipe.py 

*   templateRatio 변수에 있는 비율 대로 동작이 구분된다고 할 때, 

    test 비디오도 동작의 시간적인 비율이 크게 달라지지는 않는 다는 가정 후, 

    feature kmeans clustering &  lstmAL (feature의 변화 급격한 변화 감지 기반 동작 경계 나누기 - cvpr 논문)

    을 앙상블 느낌으로 활용하여 semantic한 동작 경계 후보들을 추천하고, templateRatio와 시간 상 가장 
 
    가까운 프레임을 최종 test 비디오의 동작경계로 제시 
    

*   단점: 매우 느리고, 반드시 같은 작업이라도 동작의 시간 비율이 유지되는 것도 아니며, 
 
          lstmAL은 동작 경계 제시를 몇 프레임을 할지 알 수 없음 (1개도 제시하지 않는 경우도 존재) 
