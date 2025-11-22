import sys

def show_type(value):
    print(f"▶ TYPE: {type(value).__name__}")
    print(value)

# override display hook
sys.displayhook = show_type
