"""제출 전 자가 점검 — 내 test.py를 운영진 검토와 같은 방식으로 여러 번 실행해 비교합니다.

사용법 (레포 폴더, 즉 test.py가 있는 폴더에서 실행)
    python check_submission.py --data_dir <battery_data_light 경로> --final <DAKER에 낸 최종 CSV>

--final 을 빼면 최종 파일 재현 비교만 건너뜁니다. 모두 ✓이면 제출해도 됩니다.
test.py를 7번 실행하므로 추론 시간의 약 7배가 걸립니다.
"""
import argparse, shutil, subprocess, sys, tempfile
from pathlib import Path
import pandas as pd
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--data_dir', type=Path, required=True, help='test/test.csv가 들어 있는 battery_data_light 폴더')
ap.add_argument('--final', type=Path, help='DAKER에 제출한 최종 선택 CSV')
args = ap.parse_args()

if not Path('test.py').is_file():
    sys.exit('이 폴더에 test.py가 없습니다. test.py가 있는 레포 폴더에서 실행하세요.')
test_csv = args.data_dir / 'test' / 'test.csv'
if not test_csv.is_file():
    sys.exit(f'{test_csv} 가 없습니다. --data_dir 경로를 확인하세요.')

test = pd.read_csv(test_csv)
image_root = args.data_dir / 'test'
tmp = Path(tempfile.mkdtemp())


def run(name, frame, root=image_root):
    inp, out = tmp / f'{name}_in.csv', tmp / f'{name}_out.csv'
    frame.to_csv(inp, index=False)
    print(f'  test.py 실행 중: {name} ({len(frame)}행)', flush=True)
    r = subprocess.run([sys.executable, 'test.py', '--input', str(inp), '--image_root', str(root),
                        '--output', str(out)], capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        print(f'[{name}] 실행 실패\n{r.stderr[-2000:]}')
        sys.exit(1)
    return pd.read_csv(out)


def diff_rate(a, b):
    m = a.merge(b, on='sample_id')
    return len(m), float((m.label_x != m.label_y).mean())


def blank_image_root():
    """test 이미지와 같은 경로·크기의 검은 이미지 폴더를 만듭니다."""
    root = tmp / 'blank_images'
    made = {}
    for rel in test.image_path.astype(str).unique():
        src = image_root / rel
        if not src.is_file():
            sys.exit(f'이미지를 찾지 못했습니다: {src}')
        with Image.open(src) as im:
            key = (im.mode, im.size)
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if key not in made:
            made[key] = tmp / f'blank_{len(made)}.png'
            Image.new(*key).save(made[key])
        shutil.copyfile(made[key], dst)
    return root


full = run('full', test)
fmt_ok = list(full.columns) == ['sample_id', 'label'] and len(full) == len(test) and full.label.isin(range(1, 7)).all()
print(f'{"✓" if fmt_ok else "✗"} 형식: sample_id·label 두 열, {len(test)}행, 라벨 1~6 정수')

n, noise = diff_rate(full, run('repeat', test))
print(f'{"✓" if noise == 0 else "⚠"} 같은 입력 다시 실행: {n}행 중 {noise:.2%} 다름')
if noise > 0:
    print('  → 같은 입력에도 결과가 흔들립니다. GPU 수치 오차일 수 있으니 시드·결정적 연산 설정을 확인하세요.')

checks = []
if args.final:
    checks.append(('최종 선택 파일 재현', pd.read_csv(args.final)))
checks.append(('행 순서 섞기', run('shuffle', test.sample(frac=1, random_state=0))))
exps = sorted(test.experiment_id.unique())
checks.append(('실험 나눠 넣기', pd.concat([run(f'split{i}', test[test.experiment_id.isin(exps[i::2])]) for i in range(2)])))
s = test.sort_values(['experiment_id', 'relative_time_s'], kind='mergesort')
head = s[s.groupby('experiment_id').cumcount() < s.groupby('experiment_id').sample_id.transform('size') * 0.6]
checks.append(('미래 행 빼고 넣기 (실험마다 앞 60%)', run('past', test[test.sample_id.isin(head.sample_id)])))

failed, warned = not fmt_ok, False
for name, pred in checks:
    n, d = diff_rate(full, pred)
    if d == 0:
        mark = '✓'
    elif noise > 0:
        mark, warned = '⚠', True   # 반복 실행도 흔들리므로 수치 오차와 구별 불가 → 운영진 검토에서 판단
    else:
        mark, failed = '✗', True   # 반복 실행은 같은데 이 검사만 다름 → 입력 구성에 따라 예측이 바뀜
    print(f'{mark} {name}: {n}행 중 {d:.2%} 다름')

# D: 이미지를 모두 검은 이미지로 바꾸면 예측이 달라져야 합니다 (이미지·센서 모두 사용).
n, d = diff_rate(full, run('blank_image', test, root=blank_image_root()))
img_ok = d > noise
failed |= not img_ok
print(f'{"✓" if img_ok else "✗"} 이미지 사용: 모든 이미지를 검은 이미지로 바꾸면 {n}행 중 {d:.2%} 예측이 바뀜'
      + ('' if img_ok else ' — 이미지가 예측에 쓰이지 않습니다'))

if failed:
    print('\n✗ 표시가 있습니다. 미래 관측, 실험·Test 전체 통계, 캐시, test.py에서 다시 fit하는 전처리가 흔한 원인입니다. '
          '이미지 사용이 ✗이면 test.py가 --image_root의 이미지를 실제로 읽는지 확인하세요. '
          'GPU를 쓰는데 차이가 아주 작다면 수치 오차일 수 있으니 한 번 더 실행해 반복 실행에도 차이가 나는지 보세요. '
          '모르겠으면 팀 담당 운영진에게 화면을 보여 주세요.')
elif warned:
    print('\n⚠ 같은 입력에도 결과가 조금씩 달라서, 위 차이가 GPU 수치 오차인지 이 도구로는 구별할 수 없습니다. '
          '차이가 반복 실행과 비슷한 수준이면 대개 수치 오차입니다. 불안하면 팀 담당 운영진에게 화면을 보여 주세요.')
else:
    print('\n모두 통과')
