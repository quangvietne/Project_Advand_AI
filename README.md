# Tối ưu hóa Điều khiển Đèn Tín hiệu Giao thông bằng DQN

Ứng dụng **Double Dueling DQN** (Deep Reinforcement Learning) để điều khiển đèn tín hiệu tại ngã tư đô thị Hà Nội, mô phỏng bằng SUMO. Agent học cách thích ứng theo lưu lượng xe thực tế, vượt trội hơn điều khiển thời gian cố định (Fixed-Time).

**Kết quả thực nghiệm:** DQN giảm hàng đợi **51.6%**, giảm thời gian chờ **74.8%**, tăng tốc độ lưu thông **16.0%** so với baseline Fixed-Time 160s.

---

## Yêu cầu môi trường

- Python 3.9+
- [SUMO](https://sumo.dlr.de/docs/Downloads.php) (Windows MSI Installer)
- PyTorch 2.2+

---

## Cài đặt

**1. Cài SUMO** và thiết lập biến môi trường:
```powershell
[Environment]::SetEnvironmentVariable("SUMO_HOME", "C:\Program Files (x86)\Eclipse\Sumo", "User")
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\Program Files (x86)\Eclipse\Sumo\bin", "User")
```

**2. Cài Python packages:**
```powershell
pip install -r requirements.txt
pip install scipy matplotlib openpyxl jupyter
```

**3. Kiểm tra cài đặt:**
```powershell
python tests/test_imports.py
python tests/test_env.py
```

---

## Chạy project

```powershell
# Huấn luyện DQN (~30–60 phút)
python scripts/train.py

# So sánh DQN vs Fixed-Time (tuần tự)
python scripts/compare_strategies.py

# So sánh song song (nhanh hơn)
python scripts/parallel_comparison.py

# Xem trực quan bằng SUMO-GUI
python scripts/gui.py dqn       # DQN agent (mặc định)
python scripts/gui.py baseline  # Fixed-Time (chu kỳ 160s bất đối xứng)

# Phân tích kết quả (Notebook)
jupyter notebook
```

---

## Cấu trúc thư mục

```
Project_Advand_AI/
├── config.yaml                     # Cấu hình trung tâm (hyperparameter, SUMO, reward)
├── requirements.txt
│
├── src/
│   ├── env/sumo_env.py             # Môi trường MDP + TraCI API
│   ├── dqn/
│   │   ├── model.py                # Dueling DQN network (MLP + LayerNorm)
│   │   ├── agent.py                # Double DQN agent
│   │   └── replay_buffer.py        # Experience replay
│   ├── baseline/
│   │   └── fixed_time_controller.py # Fixed-Time baseline (chu kỳ 160s)
│   └── utils/
│       ├── schedules.py            # LinearEpsilon schedule
│       ├── plotting.py             # Xuất CSV và bảng so sánh
│       └── generate_scenario.py   # Sinh kịch bản SUMO mẫu
│
├── scripts/
│   ├── train.py                    # Vòng lặp huấn luyện chính
│   ├── compare_strategies.py       # So sánh DQN vs Fixed-Time
│   ├── parallel_comparison.py      # So sánh song song → JSON/TXT
│   ├── gui.py                      # SUMO-GUI demo
│   ├── validate.py                 # Kiểm tra môi trường & cài đặt
│   └── common.py                   # Tiện ích dùng chung
│
├── data/scenarios/hn_sample/       # Kịch bản ngã tư Hà Nội (XML SUMO)
│
├── outputs/
│   ├── dqn_vn_tls.pt               # Model weights cuối
│   ├── dqn_vn_tls_best.pt          # Model weights tốt nhất
│   ├── chapter4/                   # Kết quả đánh giá notebook
│   ├── comparison/                 # Kết quả compare_strategies
│   └── parallel_comparison/        # Kết quả parallel_comparison
│
├── chapter4_evaluation.ipynb       # Phân tích thống kê kết quả
├── chapter4_visualization.ipynb    # Biểu đồ so sánh (PNG 300DPI + Excel)
├── tests/                          # Unit tests
└── report/                         # Báo cáo LaTeX (Chương 1–5)
```

---

## Cấu hình quan trọng (`config.yaml`)

```yaml
sumo:
  tls_id: c                 # ID đèn giao thông
  phases: [0, 1, 2, 3]      # 4 pha: NS-xanh, NS-vàng, EW-xanh, EW-vàng
  action_duration: 5         # 5 giây mỗi quyết định MDP
  max_steps: 3600            # 1 giờ mô phỏng / episode
  phase_green_min: {0: 30, 2: 60}   # NS min 30s, EW min 60s
  phase_green_max: {0: 70, 2: 140}  # NS max 70s, EW max 140s
  fixed_time_phase_schedule:        # Baseline 160s bất đối xứng
    - [2, 100]  # EW xanh 100s
    - [3, 5]    # EW vàng 5s
    - [0, 50]   # NS xanh 50s
    - [1, 5]    # NS vàng 5s

vn_weights:                  # Hệ số PCU Việt Nam
  motorcycle: 0.5
  car: 1.5
  bus: 2.0
  truck: 2.0

train:
  total_steps: 200000
  gamma: 0.99
  lr: 0.0005
  double_dqn: true
```

---

## Xử lý lỗi thường gặp

| Lỗi | Cách sửa |
|-----|----------|
| `SUMO_HOME not set` | Xem bước cài đặt SUMO ở trên |
| `ModuleNotFoundError: traci` | `pip install traci sumolib` |
| `FileNotFoundError: config.sumocfg` | `python src/utils/generate_scenario.py` |
| Model không load được | Chạy `python scripts/train.py` để tạo model |

---

## Thông tin dự án

| Mục | Chi tiết |
|-----|----------|
| Đề tài | Tối ưu hóa điều khiển đèn tín hiệu giao thông dựa trên MDP |
| Thuật toán | Double DQN + Dueling Network + Experience Replay |
| Môi trường | SUMO + TraCI API |
| Kịch bản | Ngã tư 4 hướng, lưu lượng EW:NS = 2:1, 60% xe máy |
| Trường | ĐHQGHN – Trường ĐH Khoa học Tự nhiên |
