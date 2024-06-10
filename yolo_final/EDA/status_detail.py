import os
import json

def process_json_files(json_folder):
    # 결과를 저장할 딕셔너리 생성
    abnormal_details = {}

    # JSON 폴더 안의 모든 파일에 대해 반복
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            json_file_path = os.path.join(json_folder, filename)

            # JSON 파일 열기
            with open(json_file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)

                # 각 annotation에 대해 반복
                for annotation in data["annotations"]:
                    status = annotation.get("status")
                    status_detail = annotation.get("status_detail")
                    
                    # status가 abnormal이고 status_detail이 있는 경우 처리
                    if status == "abnormal" and status_detail:
                        #if status_detail == "마모, 절손":
                                    #print(f"File: {filename}")
                        # status_detail이 리스트인 경우 처리
                        if isinstance(status_detail, list):
                            for detail in status_detail:
                                if detail not in abnormal_details:
                                    abnormal_details[detail] = 0
                                abnormal_details[detail] += 1
                        else:
                            # 딕셔너리에 새로운 status_detail 키 추가
                            if status_detail not in abnormal_details:
                                abnormal_details[status_detail] = 0

                            # 해당 status_detail의 값을 1 증가
                            abnormal_details[status_detail] += 1
                            
                              
    return abnormal_details

# 예시로 경로를 지정하여 함수 호출

json_folder_path1 = r'E:\225.철도 선로 상태 인식 데이터\01-1.정식개방데이터\Validation\Labeling\VL_일반철도_이상'
json_folder_path2 = r'E:\225.철도 선로 상태 인식 데이터\01-1.정식개방데이터\Validation\Labeling\VL_고속철도_이상'
json_folder_path3 = r'E:\225.철도 선로 상태 인식 데이터\01-1.정식개방데이터\Validation\Labeling\VL_경전철_이상'
json_folder_path4 = r'E:\225.철도 선로 상태 인식 데이터\01-1.정식개방데이터\Validation\Labeling\VL_도시철도_이상'
abnormal_details1 = process_json_files(json_folder_path1)
abnormal_details2 = process_json_files(json_folder_path2)
abnormal_details3 = process_json_files(json_folder_path3)
abnormal_details4 = process_json_files(json_folder_path4)
print(f'일반철도 비정상 디테일 : {abnormal_details1} \n 고속철도 비정상 디테일 : {abnormal_details2} \n 경전철 비정상 디테일 : {abnormal_details3} \n 도시철도 비정상 디테일 : {abnormal_details4}')


