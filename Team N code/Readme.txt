실행 개요
경진대회는 실시간 데이터 저장 및 진단 결과 전송을 다음과 같은 구조로 수행합니다.

./py/UDP_read.py
	UDP를 통해 수신된 데이터를 data/test{n}/test{n}_sec{t}.csv 형태로 1초 단위로 저장합니다.
main.py
	data/test{n} 폴더에 생성되는 sec{t} 파일을 순차적으로 읽습니다.
	매 초 전처리 → 추론 → 진단 확정 로직을 수행합니다.
	진단 결과를 UDP를 통해 전송합니다.

2. 데이터 저장 구조
data/
 └── test1/
      ├── test1_sec1.csv
      ├── test1_sec2.csv
      └── ...
각 CSV 파일은 1초 단위 실시간 데이터를 포함합니다.
파일 생성 후 즉시 파이프라인에서 처리됩니다.

3. 진단 처리 절차
각 초(sec)마다 다음 과정을 수행합니다:

파일 완성 확인 후 로드
전처리 수행
모델 추론
진단 확정 로직 적용 (팀별 기준)
진단 확정 및 시간 계산
	진단 확정 시간 = 데이터 시간(sec) + 파이프라인 runtime
	소수 첫째 자리까지 반올림
UDP를 통해 결과 전송

4. 전송 데이터 형식
test{n} sec{t}|diagnostic_results,diagnostic_time,prob_1,prob_2,...,prob_9
diagnostic_results: 진단 확정 사고(str)
diagnostic_time: 진단 확정 시간
prob: 각 레이블별 예측값(합 1)

5. 제출 파일
아래 모든 Python 파일 및 참조 파일을 함께 제출합니다:
main.py
./py/UDP_read.py
모델 관련 파일
참조 데이터 파일