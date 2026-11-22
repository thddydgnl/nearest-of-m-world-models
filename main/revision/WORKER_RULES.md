# Revision r1 작업 규칙 (작업자는 시작 전에 끝까지 읽는다)

- 저장소 루트: `/Users/song-yonghwi/Documents/IMLE-WM` (브랜치 `revision-r1`)
- 기준 설계: `main/revision/DESIGN_r1.md` v1.0 (고정판, SHA-256과 커밋은 `main/revision/LOG.md`). 가설, 지표, 문턱,
  run, seed는 바꾸지 않는다. 설계대로 할 수 없거나 바꿔야 할 것 같으면 멈추고 조정자(메인 세션)에게 보고한다.
- 문서(PLAN, LOG, REPORT)는 한국어, 코드와 주석은 영어. 시각은 UTC(`2026-11-22T18:00Z`).

## 1. 하지 않는 것

- 맥북에서 계산하지 않는다. 맥북에서는 파일 편집, 문법 검사, 몇 초 안에 끝나는 점검, 결과 읽기만 한다.
- git commit과 push를 하지 않는다. 조정자가 한다.
- `nom-` 아닌 pod, 자기에게 맡겨지지 않은 pod는 건드리지 않는다. pod 생성과 삭제는 조정자가 한다.
- 원래 run(`~/imle-wm/runs/` 중 `r1_`로 시작하지 않는 것)과 그 파일은 읽기만 한다. 새 결과는 새 파일에 쓴다.
- Google Drive 백업은 읽기 전용이다(`rclone copy/cat/ls*`만). rclone 설정과 토큰을 출력하거나 복사하지 않는다.
- 사용자의 파일을 지우지 않는다.

## 2. RunPod 사용법 (`runpod/`)

- 원격 명령: `bash /Users/song-yonghwi/Documents/IMLE-WM/runpod/rsh.sh <tag> '<cmd>'`.
  동기화: `bash .../runpod/sync.sh <tag> push|pull <src> <dst> [rsync args]`.
- 오래 도는 일은 `bash rsh.sh <tag> 'setsid nohup bash -c "<명령>" > /workspace/nom/logs/<이름>.log 2>&1 < /dev/null &'`로
  띄운다. 앞에 `cd … &&`를 붙이지 않는다(ssh가 붙잡혀 시간 초과).
- 맥북 쪽 ssh·rsync는 `perl -e 'alarm 600; exec @ARGV' …`로 감싼다(macOS에는 `timeout`이 없다). 기다릴 때 `sleep`을 쓰지
  않는다. 원격 명령이 끝나기를 기다리거나 Monitor 도구를 쓴다.
- Bash 도구는 zsh다. 따옴표 없는 `$VAR`는 단어로 나뉘지 않는다. 절대 경로를 쓴다.
- 비용: `python3 .../runpod/budget.py status`. 분석 그룹 상한은 6달러이고, 80%에 닿을 것 같으면 멈추고 보고한다.

## 3. pod 안의 배치

- `/workspace/imle-wm` = `~/imle-wm`: `code/`(맥북 `main/code/`와 같아야 한다), `data/`, `runs/`, `logs/`, `.venv`, `.hf`.
  `source ~/imle-wm/env.sh`로 환경을 켠다.
- `/workspace/nom/out/<분석>/`: 작은 결과 파일만(파일 하나 20MB 미만). 조정자가 `pull_all.sh`로
  `main/revision/an_analyses/raw/<tag>/out/`에 당겨 온다.
- `/workspace/nom/big/`: 큰 중간 파일. `/workspace/nom/logs/`: 로그.
- 결과 파일은 원자적으로 쓴다(임시 파일 → rename). 이미 있으면 건너뛴다.

## 4. 기록

- `main/revision/an_analyses/PLAN.md`: 본 실행 전에 쓴다(구현 선택, 점검 결과, 예상 시간과 비용).
- `main/revision/an_analyses/LOG.md`: 시간순 기록.
- `main/revision/an_analyses/REPORT.md`: 분석별 결과, 원자료 파일, 설계의 판정 규칙에 따른 결론, 원고에 미치는 영향.
  보고서의 숫자는 결과 JSON과 대조한 것만 쓴다. Write 도구가 막히면 Bash heredoc으로 쓴다.
