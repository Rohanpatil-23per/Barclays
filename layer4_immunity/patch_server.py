"""
Patch server.py to match new lora_retrain.py architecture
Run: python patch_server.py
"""

with open('server.py', 'r') as f:
    content = f.read()

# Fix 1: Update IMMUNEXLayer4 class architecture
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
            LoRALayer(64, 32, rank=8), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(32, 2)
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

if old_class in content:
    content = content.replace(old_class, new_class)
    print("✅ Fix 1: IMMUNEXLayer4 architecture updated")
else:
    print("⚠️  Fix 1: class block not found — may already be patched")

# Fix 2: Update ModelManager.load() to read rank
old_load = '''        self.model  = IMMUNEXLayer4(25).to(self.device)'''
new_load  = '''        rank = ckpt.get("lora_rank", 8)
        self.model  = IMMUNEXLayer4(25, rank=rank).to(self.device)'''

if old_load in content:
    content = content.replace(old_load, new_load)
    print("✅ Fix 2: ModelManager.load() updated to read lora_rank")
else:
    print("⚠️  Fix 2: ModelManager.load() block not found — may already be patched")

with open('server.py', 'w') as f:
    f.write(content)

print("\n✅ server.py patched successfully!")
