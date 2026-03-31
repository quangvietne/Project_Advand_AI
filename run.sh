#!/bin/bash
# DQN Traffic Light Control - Unified Launcher
# 
# Main Commands (SUMO GUI):
#   ./run.sh dqn                # Run DQN Agent with SUMO GUI + Live Metrics
#   ./run.sh baseline           # Run Fixed-Time Controller with SUMO GUI + Live Metrics
#   ./run.sh dual-gui           # Compare DQN vs Fixed-Time (sequential)
#
# Other Commands:
#   ./run.sh train              # Train DQN agent
#   ./run.sh compare [episodes] # Detailed comparison report (terminal)
#   ./run.sh dual [episodes]    # Side-by-side comparison (terminal)
#   ./run.sh metrics-parallel   # Parallel metrics comparison (terminal)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_menu() {
    echo -e "${BLUE}DQN Traffic Light Control - Launcher${NC}"
    echo "===================================="
    echo ""
    echo -e "${GREEN}SUMO GUI Commands (For Demonstrations):${NC}"
    echo "  dqn                Run DQN Agent with SUMO GUI + Live Metrics"
    echo "  baseline           Run Fixed-Time Controller with SUMO GUI + Live Metrics"
    echo "  dual-gui           Compare DQN vs Fixed-Time sequentially"
    echo ""
    echo -e "${GREEN}Other Commands:${NC}"
    echo "  train              Train DQN agent (no GUI)"
    echo "  compare [n]        Detailed comparison (terminal, default: 5 episodes)"
    echo "  dual [n]           Side-by-side comparison (terminal, default: 1 episode)"
    echo "  metrics-parallel   Parallel metrics comparison (terminal)"
    echo "  validate           Validate environment setup"
    echo "  docker-train       Train DQN in Docker container"
    echo ""
}

# Ensure SUMO is available for GUI commands
ensure_sumo() {
    if ! command -v sumo &> /dev/null && ! command -v sumo-gui &> /dev/null; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo -e "${YELLOW}⚠️  SUMO not found. Install with: brew install sumo${NC}"
        fi
    fi
}

# Start XQuartz on macOS if needed
start_xquartz() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if ! pgrep -q Xquartz; then
            echo "🖥️  Starting XQuartz..."
            open -a XQuartz 2>/dev/null || true
            sleep 2
        fi
    fi
}

case "${1:-help}" in
    train)
        echo -e "${BLUE}🚀 Starting DQN Training${NC}"
        python scripts/train.py
        ;;
    
    validate)
        echo -e "${BLUE}✓ Validating Environment${NC}"
        python scripts/validate.py
        ;;
    
    dqn)
        ensure_sumo
        start_xquartz
        echo -e "${BLUE}🤖 Running DQN Agent with GUI Metrics${NC}"
        python scripts/gui_metrics_window.py dqn 3600
        ;;
    
    baseline)
        ensure_sumo
        start_xquartz
        echo -e "${BLUE}🚦 Running Fixed-Time Baseline with GUI Metrics${NC}"
        python scripts/gui_metrics_window.py fixed 3600
        ;;
    
    compare)
        EPISODES="${2:-5}"
        echo -e "${BLUE}📊 Comparing DQN vs Baseline (${EPISODES} episodes)${NC}"
        python scripts/compare_strategies.py --num-episodes "$EPISODES"
        ;;
    

    metrics-parallel)
        EPISODES="${2:-1}"
        echo -e "${BLUE}📊 Parallel Comparison (${EPISODES} episodes)${NC}"
        python scripts/parallel_comparison.py --model-path outputs/dqn_vn_tls.pt --episodes "$EPISODES"
        ;;
    
    dual)
        EPISODES="${2:-1}"
        echo -e "${BLUE}🎬 Dual Simulation with Live Metrics (${EPISODES} episodes)${NC}"
        python scripts/dual_simulation_gui.py --model-path outputs/dqn_vn_tls.pt --episodes "$EPISODES"
        ;;
    
    dual-gui)
        ensure_sumo
        start_xquartz
        echo -e "${BLUE}🎬 Dual GUI Comparison${NC}"
        echo ""
        echo "This will run two simulations sequentially:"
        echo "  1. DQN Agent (with GUI + Metrics Window)"
        echo "  2. Fixed-Time Controller (with GUI + Metrics Window)"
        echo ""
        read -p "Press Enter to start DQN simulation..." 
        python scripts/gui_metrics_window.py dqn 3600
        echo ""
        read -p "Press Enter to start Fixed-Time simulation..." 
        python scripts/gui_metrics_window.py fixed 3600
        echo ""
        echo -e "${GREEN}✓ Both simulations completed${NC}"
        echo -e "${BLUE}📊 View detailed comparison:${NC} ./run.sh compare 1"
        ;;
    
    docker-train)
        echo -e "${BLUE}🐳 Training in Docker${NC}"
        docker build -t dqn-traffic-sumo .
        docker run --rm \
            -v "$(pwd)/outputs:/workspace/outputs" \
            -v "$(pwd)/data:/workspace/data" \
            dqn-traffic-sumo \
            python scripts/train.py
        ;;
    
    help|--help|-h|"")
        print_menu
        ;;
    
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        echo ""
        print_menu
        exit 1
        ;;
esac
