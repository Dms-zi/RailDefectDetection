from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


if __name__ == '__main__':
    # 모델 경로와 이름 정의
    model_paths = {
        'YOLOv8n': r'C:\Users\user\yolov8_rail\yolov8_rail\yolo_result\train\yolov8n_train\weights\best.pt',
        'YOLOv8m': r'C:\Users\user\yolov8_rail\yolov8_rail\yolo_result\train\yolov8m_train\weights\best.pt',
        'YOLOv8l': r'C:\Users\user\yolov8_rail\yolov8_rail\yolo_result\train\yolov8l_train\weights\best.pt',
        'YOLOv9c': r'C:\Users\user\yolov8_rail\yolov8_rail\yolo_result\train\yolov9c_train\weights\best.pt',
    }

    # 빈 데이터프레임 초기화
    df_total = pd.DataFrame()

    # 각 모델에 대해 평가 수행 및 데이터프레임 생성
    for model_name, model_path in model_paths.items():
        # 모델 로드 및 평가 수행
        model = YOLO(model_path)
        print(f"현재 진행되는 모델: {model_name}")
        data_path = r"C:\Users\user\yolov8_rail\yolov8_rail\data.yaml"
        # if model_name.split('_')[-1] == 'all':
        #     data_path = "data.yaml"
        results = model.val(data=data_path) 
            
        # 필요한 평가 지표 추출 (클래스별)
        data = {
            #'Precision': results.box.p.tolist(),
            #'Recall': results.box.r.tolist(),
            #'F1_Score': results.box.f1.tolist(),
            'AP_50': results.box.ap50.tolist(),
            'AP_50_95': results.box.ap.tolist(),
        }
        
        # 데이터 프레임 생성
        df = pd.DataFrame(data)
        df['Model'] = model_name
        
        # 총 데이터프레임에 추가
        df_total = pd.concat([df_total, df], ignore_index=True)



    # Melt the DataFrame for easier plotting with seaborn
    df_melted = df_total.melt(id_vars='Model', var_name='Metric', value_name='Value')

    # Box plot 생성
    plt.figure(figsize=(16, 8))
    sns.boxplot(x='Metric', y='Value', hue='Model', data=df_melted)

    # 그래프 제목 설정
    plt.title('Rail Dataset')

    # 그래프 출력
    plt.show()
