"""
Patch blind_spot.py and server.py to match new lora_retrain.py architecture
Run: python patch_all.py
"""

# ── Patch blind_spot.py ───────────────────────────────────────────────────────
print("Patching blind_spot.py...")
with open('blind_spot.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

old_class = '''class IMMUNEXLayer4(nn.Module):
    def __init__(self, input_dim=25):
        super().__init__()
        self.base_encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(0.2),
        )
        self.lora_head = nn.Sequential(
            LoRALayer(64, 32, rank=8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.lora_head(self.base_encoder(x))'''

new_class = '''class IMMUNEXLayer4(nn.Module):
    def __init__(self, input_dim=25, rank=8):
        super().__init__()
        self.base_encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(0.2),
        )
        self.lora_layer = LoRALayer(64, 32, rank=rank)
        self.head = nn.Sequential(
            nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.head(self.lora_layer(self.base_encoder(x)))'''

old_load = '''    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model      = IMMUNEXLayer4(input_dim=25).to(device)'''

new_load = '''    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    rank  = checkpoint.get("lora_rank", 8)
    model = IMMUNEXLayer4(input_dim=25, rank=rank).to(device)'''

c1 = content.replace(old_class, new_class)
c2 = c1.replace(old_load, new_load)

if c2 != content:
    with open('blind_spot.py', 'w', encoding='utf-8') as f:
        f.write(c2)
    print("  blind_spot.py patched successfully")
else:
    print("  blind_spot.py already up to date or pattern not found")

# ── Patch server.py ───────────────────────────────────────────────────────────
print("Patching server.py...")
with open('server.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

old_srv_class = '''class IMMUNEXLayer4(nn.Module):
    def __init__(self, input_dim=25):
        super().__init__()
        self.base_encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(0.2),
        )
        self.lora_head = nn.Sequential(
            LoRALayer(64, 32, rank=8), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.lora_head(self.base_encoder(x))'''

new_srv_class = '''class IMMUNEXLayer4(nn.Module):
    def __init__(self, input_dim=25, rank=8):
        super().__init__()
        self.base_encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(0.2),
        )
        self.lora_layer = LoRALayer(64, 32, rank=rank)
        self.head = nn.Sequential(
            nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.head(self.lora_layer(self.base_encoder(x)))'''

old_mgr = '        self.model  = IMMUNEXLayer4(25).to(self.device)'
new_mgr  = '        rank = ckpt.get("lora_rank", 8)\n        self.model  = IMMUNEXLayer4(25, rank=rank).to(self.device)'

c1 = content.replace(old_srv_class, new_srv_class)
c2 = c1.replace(old_mgr, new_mgr)

if c2 != content:
    with open('server.py', 'w', encoding='utf-8') as f:
        f.write(c2)
    print("  server.py patched successfully")
else:
    print("  server.py already up to date or pattern not found")

print("\nAll done! Now run:")
print("  python lora_retrain.py 2>&1")
