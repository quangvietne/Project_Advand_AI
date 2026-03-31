# Tối ưu hóa Điều khiển Đèn Tín hiệu Giao thông bằng DQN

Dự án sử dụng **Deep Q-Network (DQN)** để tối ưu hóa điều khiển đèn tín hiệu giao thông tại các nút giao đô thị Việt Nam, mô phỏng bằng phần mềm SUMO.

**Đặc điểm nổi bật:**
- Thuật toán Double DQN kết hợp Dueling Network
- Phân bố phương tiện đặc thù Việt Nam (60% xe máy, 30% ô tô, 10% xe buýt/xe tải)
- Bộ điều khiển thời gian cố định (Fixed-Time) làm baseline so sánh
- Notebook Jupyter đánh giá và trực quan hóa kết quả (Chương 4)

---

## Hướng dẫn cài đặt và chạy từ đầu

> Dành cho người mới clone code về lần đầu, chưa cài gì cả.

---

### Bước 1 — Cài đặt Python

Yêu cầu **Python 3.9 trở lên**.

Kiểm tra:
```powershell
python --version
```

Nếu chưa có → tải tại: https://www.python.org/downloads/

---

### Bước 2 — Cài đặt SUMO (bắt buộc)

SUMO là phần mềm mô phỏng giao thông, bắt buộc phải có.

**Windows:**
1. Tải installer tại: https://sumo.dlr.de/docs/Downloads.php → chọn **Windows MSI Installer**
2. Cài đặt bình thường (Next → Next → Finish)
3. Mặc định SUMO cài vào: `C:\Program Files (x86)\Eclipse\Sumo`

**Sau khi cài**, set biến môi trường (mở PowerShell và chạy):
```powershell
# Thêm vào PowerShell profile (chạy 1 lần)
$sumoPath = "C:\Program Files (x86)\Eclipse\Sumo"
[Environment]::SetEnvironmentVariable("SUMO_HOME", $sumoPath, "User")
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";$sumoPath\bin", "User")
```

Sau đó **đóng và mở lại PowerShell**, kiểm tra:
```powershell
echo $env:SUMO_HOME
sumo --version
```

---

### Bước 3 — Clone/tải code về

```powershell
# Nếu dùng git
git clone <link-repo>
cd Final

# Hoặc giải nén file zip vào thư mục
cd "d:\Study\Advand AI\Final\Final"
```

---

### Bước 4 — Cài đặt Python packages

```powershell
cd "d:\Study\Advand AI\Final\Final"
pip install -r requirements.txt
```

Các package được cài:
- `torch>=2.2` — Mạng nơ-ron DQN
- `numpy>=1.24` — Tính toán mảng số
- `pyyaml>=6.0.1` — Đọc file config
- `tqdm>=4.66` — Thanh tiến trình
- `traci>=1.20.0` — Giao tiếp với SUMO
- `sumolib>=1.20.0` — Tiện ích SUMO

---

### Bước 5 — Kiểm tra cài đặt

```powershell
python scripts/validate.py
```

Output mong đợi:
```
✓ torch
✓ numpy
✓ traci
✓ SUMO_HOME: C:\Program Files (x86)\Eclipse\Sumo
✓ config.sumocfg
✓ intersection.net.xml
✓ routes.rou.xml
```

Nếu thiếu file scenario → chạy:
```powershell
python src/utils/generate_scenario.py
```

---

### Bước 6 — Huấn luyện DQN (nếu chưa có model)

> Nếu đã có file `outputs/dqn_vn_tls.pt` thì bỏ qua bước này.

```powershell
python scripts/train.py
```

- Thời gian: ~30-60 phút tùy máy
- Kết quả lưu tại: `outputs/dqn_vn_tls.pt`
- Theo dõi tiến trình qua thanh `tqdm` hiện trên terminal

---

### Bước 7 — Chạy demo và xem kết quả

**Chạy DQN agent (AI điều khiển đèn):**
```powershell
python scripts/gui.py dqn
```

**Chạy Fixed-Time controller (đèn cố định 55s xanh + 5s vàng, chu kỳ 120s):**
```powershell
python scripts/gui.py baseline
```

**So sánh DQN vs Fixed-Time (không cần GUI, xem số):**
```powershell
python scripts/parallel_comparison.py
```

---

## Cấu trúc thư mục

```
Final/
├── config.yaml                     # Cấu hình huấn luyện và môi trường
├── requirements.txt                # Danh sách package cần cài
├── README.md                       # File này
│
├── src/                            # Mã nguồn thư viện lõi
│   ├── dqn/
│   │   ├── model.py               # Mạng Dueling DQN
│   │   ├── agent.py               # Double DQN agent
│   │   └── replay_buffer.py       # Experience replay buffer
│   ├── env/
│   │   └── sumo_env.py            # Wrapper môi trường SUMO (Gym-style)
│   ├── baseline/
│   │   └── fixed_time_controller.py  # Đèn cố định 55s xanh + 5s vàng (120s chu kỳ)
│   └── utils/
│       ├── schedules.py           # Lịch giảm epsilon
│       ├── generate_scenario.py   # Tạo file kịch bản SUMO
│       └── plotting.py            # Xuất CSV và bảng .txt
│
├── scripts/                        # Các script chạy chính
│   ├── train.py                   # Huấn luyện DQN
│   ├── gui.py                     # Mở SUMO-GUI (dqn / baseline)
│   ├── compare_strategies.py      # So sánh chi tiết
│   ├── parallel_comparison.py     # So sánh song song → JSON
│   ├── dual_simulation_gui.py     # Chạy cả 2 chiến lược tuần tự
│   ├── validate.py                # Kiểm tra cài đặt
│   └── common.py                  # Tiện ích dùng chung
│
├── data/
│   └── scenarios/hn_sample/       # Kịch bản ngã tư Hà Nội (file XML SUMO)
│
├── outputs/
│   ├── dqn_vn_tls.pt              # Trọng số model đã train
│   ├── chapter4/                  # Kết quả đánh giá chương 4
│   ├── comparison/                # Kết quả so sánh tiêu chuẩn
│   └── parallel_comparison/       # Kết quả so sánh song song
│
├── chapter4_evaluation.ipynb      # Notebook đánh giá chương 4
├── chapter4_visualization.ipynb   # Notebook biểu đồ kết quả
└── QuickStart.ipynb               # Notebook demo nhanh
```

---

## Cấu hình chính (config.yaml)

```yaml
sumo:
  tls_id: c               # ID đèn giao thông trong SUMO
  phases: [0, 1, 2, 3]   # 4 pha: NS-thẳng, NS-trái, ĐT-thẳng, ĐT-trái
  action_duration: 5      # Giữ mỗi pha bao nhiêu giây
  max_steps: 3600         # 1 giờ mô phỏng

vn_weights:               # Trọng số PCU theo chuẩn Việt Nam
  motorcycle: 0.5         # 1 ô tô ≈ 3 xe máy
  car: 1.5
  bus: 2.0
  truck: 2.0

train:
  total_steps: 200000     # Số bước huấn luyện
  gamma: 0.99             # Hệ số chiết khấu
  lr: 0.0005              # Learning rate
  double_dqn: true        # Dùng Double DQN
```

---

## Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|-----|-------------|----------|
| `ModuleNotFoundError: traci` | Chưa cài SUMO hoặc pip package | `pip install traci sumolib` |
| `SUMO_HOME not set` | Chưa set biến môi trường | Xem Bước 2 |
| `sumo: command not found` | Chưa thêm SUMO vào PATH | Thêm `C:\...\Sumo\bin` vào PATH |
| `FileNotFoundError: config.sumocfg` | Thiếu file kịch bản | `python src/utils/generate_scenario.py` |
| `KeyError: dqn_vn_tls.pt` | Chưa train model | `python scripts/train.py` |

---

## Thông tin dự án

| Mục | Chi tiết |
|-----|----------|
| Tên đề tài | Tối ưu hóa điều khiển đèn tín hiệu giao thông dựa trên MDP |
| Thuật toán | Double DQN + Dueling Network |
| Môi trường | SUMO (Simulation of Urban MObility) |
| Kịch bản | Ngã tư đô thị Hà Nội mẫu |
| Sinh viên | Nguyễn Quang Việt (22001659) · Phùng Hữu Uy (22001654) |
| Python | 3.9+ |
| PyTorch | 2.2+ |
   - `__pycache__/` - Python caches

3. **When adding code:**
   - Add docstrings to functions
   - Update config.yaml for new hyperparameters
   - Create tests for new features

---

## Support

For issues or questions:
1. Check the relevant script docstring: `python -c "import scripts.train; help(scripts.train.main)"`
2. Review test files for usage examples
3. Check notebook cells for implementation examples
4. See `run.sh` for all available commands

**Last Updated:** January 2026  
**Status:** Ready for thesis submission
