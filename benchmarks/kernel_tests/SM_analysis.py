from collections import Counter

def count_sm_ids(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    sm_ids = []
    for line in lines:
        if line.startswith("SM"):
            parts = line.strip().split()
            if len(parts) == 2 and parts[1].isdigit():
                sm_ids.append(int(parts[1]))
                
    counter = Counter(sm_ids)

    print("SM ID Counts:")
    for sm_id, count in sorted(counter.items()):
        print(f"SM {sm_id}: {count} times")
    print(f"total SM used count : {len(sm_ids)}")
# 사용 예:
count_sm_ids("shrink_sm.txt")
