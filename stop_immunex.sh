#!/bin/bash
echo "Stopping IMMUNEX..."
pkill -f "layer1_detection/server.py" 2>/dev/null && echo "  ✅ Layer 1 stopped" || true
pkill -f "orchestrator/server.py"     2>/dev/null && echo "  ✅ Orchestrator stopped" || true
pkill -f "layer2"                     2>/dev/null && echo "  ✅ Layer 2 stopped" || true
pkill -f "layer3"                     2>/dev/null && echo "  ✅ Layer 3 stopped" || true
pkill -f "layer4"                     2>/dev/null && echo "  ✅ Layer 4 stopped" || true
<<<<<<< HEAD

=======
pkill -f "layer5"                     2>/dev/null && echo "  ✅ Layer 5 stopped" || true
>>>>>>> 2b0972f24f02f6df454050c626cf8a1556f12d69
docker compose down
echo "  ✅ Docker stopped"
echo "IMMUNEX stopped."
